"""Experimental cross-image face matching over the real event photos (Stage 4).

This module REUSES the working detection -> size -> quality -> pose -> embedding
pipeline from :mod:`ml.inference.predict` unchanged. It adds, as an *experiment
only*, a cosine-similarity comparison of the kept faces across images and a
human-readable face-tracking / grouping report.

Scope (deliberately limited)
----------------------------
* Face embeddings -> cosine similarity -> EXPERIMENTAL groups. Nothing more.
* NO final clustering (no DBSCAN / K-Means / Agglomerative), no Person IDs,
  no Merge Sort, no web UI, no database, no training.
* The groups and every similarity value printed/written here are produced from
  the actual images in the target directory. Nothing is fabricated.
* The grouping threshold is a configurable, clearly-labelled EXPERIMENTAL value
  used only to visualise candidate matches; the point of the experiment is to
  see whether similarity values naturally separate same- vs different-person
  faces.

Pipeline per image (all reused from predict.py)
-----------------------------------------------
    detect_face_regions -> evaluate_faces (size -> quality -> pose)
        -> embed_faces (kept + uncertain only)

Every detected face in an image gets a TEMPORARY index (``Face 0``, ``Face 1``,
...). These indices identify a face within one image for this experiment only;
they are NOT Person IDs.

Usage
-----
    python ml/inference/match_faces.py
    python ml/inference/match_faces.py --dir ml/datasets/test --glob "event*"
    python ml/inference/match_faces.py --match-threshold 0.45
    python ml/inference/match_faces.py --provider CPUExecutionProvider
    python ml/inference/match_faces.py --dir ml/datasets/test --glob "event*" --debug-filtering

The ``--debug-filtering`` flag prints the full per-face metrics (relative area,
relative-face-size, detection confidence, quality/blur, yaw/pitch) and the
KEEP/FILTERED/UNCERTAIN decision for every detected face before matching runs.
An invariant check at the end asserts that only KEEP/UNCERTAIN faces were
embedded (embeddings == kept), which is what catches a filter bypass.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Allow running this file directly (``python ml/inference/match_faces.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.inference import predict as P  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration  (all EXPERIMENTAL / overridable)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
CSV_PATH = DATA_OUTPUT_DIR / "face_matching_results.csv"
UNCERTAIN_CSV_PATH = DATA_OUTPUT_DIR / "confirmed_uncertain_matches.csv"

DEFAULT_DIR = PROJECT_ROOT / "ml" / "datasets" / "test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Cosine-similarity threshold used to *visualise* candidate matches. It is NOT a
# claim that 0.5 (or anything else) means "same person" — see --match-threshold.
# Default reuses the project's existing display threshold (predict.py).
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", str(P.FACE_SIMILARITY_THRESHOLD)))

# A cross-image pair whose similarity falls within ``[MATCH_THRESHOLD - margin,
# MATCH_THRESHOLD)`` is reported separately as a POTENTIALLY AMBIGUOUS match
# instead of being forced into a group.
AMBIGUOUS_MARGIN = float(os.getenv("FACE_MATCH_AMBIGUOUS_MARGIN", "0.05"))

# How many strongest cross-image pairs to list.
TOP_MATCHES_LIMIT = int(os.getenv("FACE_MATCH_TOP_N", "12"))

# --- Uncertain-face recovery (FINAL, separate from the grouping threshold) -----
# Every UNCERTAIN face already has an embedding (predict.py embeds KEEP +
# UNCERTAIN faces). If such a face has a STRONG cosine-similarity match with a
# KEEP face from a DIFFERENT image, it is recovered/promoted to
# CONFIRMED_FROM_MATCH. This uses its OWN configurable threshold and must NOT
# reuse the 0.50 experimental grouping threshold above.
UNCERTAIN_PROMOTION_THRESHOLD = float(
    os.getenv("UNCERTAIN_PROMOTION_THRESHOLD", "0.70")
)

# Status assigned to an UNCERTAIN face that is promoted by a strong match.
STATUS_CONFIRMED_FROM_MATCH = "CONFIRMED_FROM_MATCH"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class FaceRecord:
    """One embedded face: its image, temporary index and experimental metrics.

    ``eq=False`` keeps identity semantics — the dataclass must never compare its
    numpy ``embedding`` field with ``==`` (which would return an array).
    """

    image_filename: str
    face_index: int  # temporary index over ALL detections in that image (0-based)
    status: str
    det_score: float
    relative_area: float
    blur_score: float
    yaw: float | None
    pitch: float | None
    bbox: list[float]
    embedding: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.float32))
    # --- Uncertain-face recovery bookkeeping ---------------------------------
    # ``status`` (above) is the ORIGINAL filtering decision. ``final_status``
    # starts equal to it and may be promoted to CONFIRMED_FROM_MATCH by a strong
    # cross-image match with a KEEP face. Both are preserved side by side.
    final_status: str = ""
    uncertainty_reason: str = ""
    matched_image_filename: str = ""
    matched_face_index: int = -1
    promotion_similarity: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.image_filename} -> Face {self.face_index}"

    @property
    def original_status(self) -> str:
        """The original filtering decision (alias of ``status``)."""
        return self.status

    @property
    def promoted(self) -> bool:
        return self.final_status == STATUS_CONFIRMED_FROM_MATCH


@dataclass
class ImageResult:
    """Per-image outcome of the reused pipeline, kept for the report."""

    image_filename: str
    image_shape: tuple[int, ...]
    total_detected: int
    faces: list  # raw predict.DetectedFace objects (ALL detections, for reporting)
    records: list[FaceRecord]  # embedded faces only (KEEP + UNCERTAIN)
    detection_seconds: float
    filter_seconds: float
    embedding_seconds: float
    # Duplicate boxes the detection stage suppressed for this photo (the same
    # physical face returned twice by the detector). 0 for a clean photo.
    duplicates_removed: int = 0

    @property
    def embedded(self) -> int:
        return len(self.records)

    @property
    def after_filtering(self) -> int:
        return len(self.records)

    @property
    def filtered(self) -> int:
        return self.total_detected - self.after_filtering


# ---------------------------------------------------------------------------
# Union-Find (connected components) for experimental grouping
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


# ---------------------------------------------------------------------------
# Per-image processing (reuses predict.py)
# ---------------------------------------------------------------------------


def list_images(directory: Path, glob_pattern: str = "*") -> list[Path]:
    """Return image files in ``directory`` matching ``glob_pattern``, sorted."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = [
        p
        for p in sorted(directory.glob(glob_pattern))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def process_image(image_path: Path, app, **filter_kwargs) -> ImageResult:
    """Run the existing (predict.py) pipeline and collect an :class:`ImageResult`.

    This delegates **entirely** to :func:`ml.inference.predict.analyze_image`,
    which is the single authoritative implementation of

        detect -> relative-size filter -> quality filter -> pose filter
               -> embed (KEEP + UNCERTAIN only)

    No filtering logic is duplicated or re-implemented here. A face only has an
    ``embedding`` if it survived the filters, so collecting faces with a
    non-``None`` embedding is exactly "collected the kept faces".
    """
    image = P.read_image(image_path)
    analysis = P.analyze_image(image, image_path=image_path, app=app, **filter_kwargs)

    records: list[FaceRecord] = []
    for index, face in enumerate(analysis.all_faces):
        if face.embedding is None:  # filtered face -> never embedded -> skipped
            continue
        records.append(
            FaceRecord(
                image_filename=image_path.name,
                face_index=index,
                status=face.status,
                det_score=float(face.det_score),
                relative_area=float(face.relative_area),
                blur_score=float(face.blur_score),
                yaw=face.yaw,
                pitch=face.pitch,
                bbox=[float(v) for v in face.bbox],
                embedding=np.asarray(face.embedding, dtype=np.float32),
                final_status=face.status,
                uncertainty_reason=face.reason,
            )
        )

    return ImageResult(
        image_filename=image_path.name,
        image_shape=image.shape,
        total_detected=analysis.total_faces,
        faces=list(analysis.all_faces),
        records=records,
        detection_seconds=analysis.detection_seconds,
        filter_seconds=analysis.filter_seconds,
        embedding_seconds=analysis.embedding_seconds,
        duplicates_removed=analysis.duplicate_count,
    )


# ---------------------------------------------------------------------------
# Cross-image similarity + experimental grouping
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Pair:
    a: FaceRecord
    b: FaceRecord
    similarity: float


def cross_image_pairs(records: list[FaceRecord]) -> list[Pair]:
    """All pairs of embedded faces that come from DIFFERENT images."""
    pairs: list[Pair] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if a.image_filename == b.image_filename:
                continue
            pairs.append(Pair(a, b, P.compare_embeddings(a.embedding, b.embedding)))
    return pairs


def build_groups(
    records: list[FaceRecord], pairs: list[Pair], threshold: float
) -> list[list[int]]:
    """Connected components over cross-image edges with ``similarity >= threshold``.

    Returns a list of groups, each a list of indices into ``records``.
    """
    index_of = {id(record): i for i, record in enumerate(records)}
    uf = UnionFind(len(records))
    for pair in pairs:
        if pair.similarity >= threshold:
            uf.union(index_of[id(pair.a)], index_of[id(pair.b)])
    buckets: dict[int, list[int]] = {}
    for idx in range(len(records)):
        buckets.setdefault(uf.find(idx), []).append(idx)
    groups = list(buckets.values())
    # Stable, readable order: larger groups first, then by first occurrence.
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def pick_reference(members: list[FaceRecord]) -> FaceRecord:
    """Reference face of a group: highest detection confidence (ties: first)."""
    return max(members, key=lambda r: (r.det_score, -r.face_index))


# ---------------------------------------------------------------------------
# Uncertain-face recovery (FINAL, separate threshold)
# ---------------------------------------------------------------------------


def recover_uncertain_faces(
    records: list[FaceRecord], threshold: float = UNCERTAIN_PROMOTION_THRESHOLD
) -> list[FaceRecord]:
    """Promote UNCERTAIN faces that strongly match a KEEP face from another image.

    Rules enforced here (and only here — no filtering threshold is touched):

    1. only ``original_status == UNCERTAIN`` faces are considered;
    2. their embedding must already exist (guaranteed: they were embedded but
       this re-checks the vector is non-empty);
    3. they are compared against ``KEEP`` faces ONLY;
    4. the KEEP face must come from a DIFFERENT image (same-image pairs skipped);
    5. the strongest eligible KEEP match is selected per UNCERTAIN face;
    6. promotion happens only when ``similarity >= threshold``;
    7. otherwise ``final_status`` stays ``UNCERTAIN``;
    8. FILTERED faces can never take part — they are neither promoted nor used as
       a match target (they have no embedding and are excluded by status).

    Mutates the matched records in place and returns the list of promoted records.
    """
    keeps = [r for r in records if r.status == P.STATUS_KEEP]
    promotions: list[FaceRecord] = []
    for record in records:
        if record.status != P.STATUS_UNCERTAIN:
            continue
        if record.embedding.size == 0:  # rule 2: an embedding must already exist
            continue
        best_match: FaceRecord | None = None
        best_similarity = -1.0
        for keep in keeps:  # rules 3 + 8 (KEEP only; FILTERED never embedded)
            if keep.image_filename == record.image_filename:  # rule 4
                continue
            similarity = P.compare_embeddings(record.embedding, keep.embedding)
            if similarity > best_similarity:  # rule 5
                best_similarity = similarity
                best_match = keep
        if best_match is not None:
            # Record the strongest eligible KEEP candidate (useful evidence even
            # when it does not clear the threshold).
            record.matched_image_filename = best_match.image_filename
            record.matched_face_index = best_match.face_index
            record.promotion_similarity = best_similarity
        if best_match is not None and best_similarity >= threshold:  # rule 6
            record.final_status = STATUS_CONFIRMED_FROM_MATCH
            promotions.append(record)
        # rule 7: no match above threshold -> final_status stays UNCERTAIN.
    return promotions


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".1f") -> str:
    return "n/a" if value is None else format(value, spec)


def verify_pipeline_invariants(results: list[ImageResult]) -> bool:
    """Assert that only kept/uncertain faces were embedded.

    This is the guard that would have caught a filter bypass: for every image the
    number of embeddings must equal the number of KEEP+UNCERTAIN faces, and **no
    FILTERED face may carry an embedding**.
    """
    print("\n" + "=" * 70)
    print("FILTERING INVARIANT CHECK (embeddings must equal kept + uncertain)")
    print("=" * 70)
    ok = True
    for result in results:
        kept = sum(1 for f in result.faces if f.status in (P.STATUS_KEEP, P.STATUS_UNCERTAIN))
        embedded = sum(1 for f in result.faces if f.embedding is not None)
        filtered_embedded = sum(
            1 for f in result.faces
            if f.status == P.STATUS_FILTERED and f.embedding is not None
        )
        good = kept == embedded == result.embedded and filtered_embedded == 0
        ok = ok and good
        print(f"  {result.image_filename:16s} detected={result.total_detected:2d}  "
              f"duplicates={result.duplicates_removed:d}  "
              f"filtered={result.filtered:2d}  kept={kept:2d}  embeddings={embedded:2d}  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\n  Overall: {'PASS' if ok else 'FAIL'} — no FILTERED face received an embedding.")
    return ok


def print_image_blocks(results: list[ImageResult], debug: bool = False) -> None:
    print("\n" + "=" * 70)
    print("PER-IMAGE DETECTION / FILTERING RESULTS")
    if debug:
        print("(debug mode: full per-face metrics shown)")
    print("=" * 70)
    for result in results:
        print("\n" + "=" * 70)
        print(f"IMAGE: {result.image_filename}")
        print("=" * 70)
        print(f"  Original faces detected: {result.total_detected}")
        if result.duplicates_removed:
            print(f"  Duplicate boxes suppressed: {result.duplicates_removed}")
        for index, face in enumerate(result.faces):
            if debug:
                print(f"\n  Face {index}:")
                print(f"    Width: {face.width:.1f}   Height: {face.height:.1f} px")
                print(f"    Face area: {face.area:.0f} px  "
                      f"(image area: {face.image_area:.0f} px)")
                print(f"    Relative image area: {face.relative_area:.5f} "
                      f"({face.relative_area * 100:.3f}%)")
                print(f"    Largest face area: {face.largest_face_area:.0f} px")
                print(f"    Ratio to largest face: {face.ratio_to_largest_face:.3f}")
                print(f"    Detection confidence: {face.det_score:.3f}")
                print(f"    Quality score (blur): {face.blur_score:.1f}")
                print(f"    Yaw: {_fmt(face.yaw)}   Pitch: {_fmt(face.pitch)}")
                print(f"    Decision: {face.status}")
                if face.reason:
                    print(f"    Reason: {face.reason}")
            else:
                line = f"  Face {index} -> {face.status}"
                if face.reason:
                    line += f"   ({face.reason})"
                print(line)
        print(f"\n  Filtered out: {result.filtered}")
        print(f"  Final relevant faces (kept + uncertain): {result.after_filtering}")
        print(f"  Embeddings generated: {result.embedded}")


def print_tracking_report(groups: list[list[FaceRecord]], pairs: list[Pair], threshold: float) -> None:
    print("\n" + "=" * 70)
    print("FACE TRACKING VERIFICATION  [EXPERIMENTAL GROUPS]")
    print("=" * 70)
    print(f"Grouping method: connected components over cross-image cosine similarity")
    print(f"EXPERIMENTAL threshold = {threshold:.2f} (configurable; NOT a claim about")
    print("what value means 'same person' — the goal is to inspect the separation).")

    for gid, members in enumerate(groups, start=1):
        print(f"\nExperimental Face Group {gid}" + ("  (single-occurrence)" if len(members) == 1 else ""))
        print("  Appears in:")
        for member in members:
            print(f"    {member.label}   [{member.status}]")
        if len(members) == 1:
            print("  Status: No matching face found in other images.")
            continue
        reference = pick_reference(members)
        print(f"  Reference face: {reference.label} (highest detection confidence)")
        print("  Similarity evidence (similarity to group reference):")
        for member in members:
            if member is reference:
                continue
            sim = P.compare_embeddings(member.embedding, reference.embedding)
            print(f"    {member.label} <-> {reference.label} = {sim:.4f}")
        # Strongest edges that actually caused this grouping.
        member_set = set(id(m) for m in members)
        internal = sorted(
            (p for p in pairs if id(p.a) in member_set and id(p.b) in member_set),
            key=lambda p: p.similarity,
            reverse=True,
        )
        if internal:
            print("  Strongest internal pairs (>= threshold):")
            for p in internal:
                if p.similarity >= threshold:
                    print(f"    {p.a.label} <-> {p.b.label} = {p.similarity:.4f}")


def print_top_matches(pairs: list[Pair], limit: int) -> None:
    print("\n" + "=" * 70)
    print(f"TOP {limit} STRONGEST CROSS-IMAGE MATCHES")
    print("=" * 70)
    if not pairs:
        print("  (no cross-image pairs available)")
        return
    for rank, pair in enumerate(sorted(pairs, key=lambda p: p.similarity, reverse=True)[:limit], start=1):
        print(f"\n{rank}. {pair.a.label}")
        print(f"   <-> {pair.b.label}")
        print(f"   Similarity: {pair.similarity:.4f}")


def print_ambiguous_matches(pairs: list[Pair], threshold: float, margin: float) -> None:
    low = threshold - margin
    ambiguous = [p for p in pairs if low <= p.similarity < threshold]
    ambiguous.sort(key=lambda p: p.similarity, reverse=True)
    print("\n" + "=" * 70)
    print("POTENTIALLY AMBIGUOUS MATCHES")
    print("=" * 70)
    print(f"Band: {low:.2f} <= similarity < {threshold:.2f} (just below the experimental")
    print("grouping threshold). These are NOT forced into any group.")
    if not ambiguous:
        print("  (none — no cross-image pair falls in the ambiguous band)")
        return
    for pair in ambiguous:
        print(f"\n  {pair.a.label}")
        print(f"  <-> {pair.b.label}")
        print(f"  Similarity: {pair.similarity:.4f}")
        print("  Reason: similarity is not sufficiently separated from the grouping boundary.")


def print_single_occurrence(groups: list[list[FaceRecord]]) -> None:
    singles = [g[0] for g in groups if len(g) == 1]
    print("\n" + "=" * 70)
    print("SINGLE-OCCURRENCE FACES (retained — never dropped for appearing once)")
    print("=" * 70)
    if not singles:
        print("  (none)")
        return
    for record in singles:
        print(f"\n  {record.label}")
        print("  Status: No matching face found in other images.")


def print_uncertain_recovery_report(
    records: list[FaceRecord], promotions: list[FaceRecord], threshold: float
) -> None:
    """Print the uncertain-face recovery outcome: examined / promoted / remaining."""
    uncertain = [r for r in records if r.status == P.STATUS_UNCERTAIN]
    remaining = [r for r in uncertain if not r.promoted]
    print("\n" + "=" * 70)
    print("UNCERTAIN FACE RECOVERY  [EXPERIMENTAL]")
    print("=" * 70)
    print(f"Promotion threshold (separate from grouping) = {threshold:.2f}")
    print("An UNCERTAIN face is promoted to CONFIRMED_FROM_MATCH only if its")
    print("strongest cross-image KEEP match is >= this threshold.")
    print(f"\n  Uncertain faces examined: {len(uncertain)}")
    print(f"  Uncertain faces promoted: {len(promotions)}")
    print(f"  Uncertain faces remaining: {len(remaining)}")

    if promotions:
        print("\n  PROMOTIONS (every one, with its similarity score):")
        for record in promotions:
            print(f"\n    {record.image_filename} -> Face {record.face_index}")
            print(f"      original_status: {record.original_status}")
            print(f"      uncertainty_reason: {record.uncertainty_reason or 'n/a'}")
            print(f"      matched KEEP: {record.matched_image_filename} -> "
                  f"Face {record.matched_face_index}")
            print(f"      similarity: {record.promotion_similarity:.4f} "
                  f"(threshold {threshold:.2f})")
            print(f"      final_status: {record.final_status}")
    else:
        print("\n  (no UNCERTAIN face reached the promotion threshold)")

    if remaining:
        print("\n  REMAINING UNCERTAIN (no cross-image KEEP match reached the threshold):")
        for record in remaining:
            print(f"\n    {record.image_filename} -> Face {record.face_index}")
            print(f"      uncertainty_reason: {record.uncertainty_reason or 'n/a'}")
            if record.matched_image_filename:
                print(f"      strongest cross-image KEEP candidate: "
                      f"{record.matched_image_filename} -> Face {record.matched_face_index}"
                      f"  similarity {record.promotion_similarity:.4f} "
                      f"(< {threshold:.2f})")
            else:
                print("      no eligible cross-image KEEP candidate")
            print(f"      final_status: {record.final_status}")


def print_similarity_distribution(pairs: list[Pair]) -> None:
    print("\n" + "=" * 70)
    print("CROSS-IMAGE SIMILARITY DISTRIBUTION (all pairs, no threshold applied)")
    print("=" * 70)
    if not pairs:
        print("  (no cross-image pairs)")
        return
    scores = np.array([p.similarity for p in pairs])
    print(f"  Cross-image pairs total: {len(scores)}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Median: {float(np.median(scores)):.4f}")
    print(f"  Mean:   {float(scores.mean()):.4f}")
    print(f"  Max:    {scores.max():.4f}")
    edges = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    print("  Histogram (bins of similarity):")
    for low, high in zip(edges, edges[1:] + [1.0]):
        count = int(np.sum((scores >= low) & (scores < high)))
        bar = "#" * count
        print(f"    [{low:.2f}, {high:.2f}): {count:3d}  {bar}")


def print_threshold_sensitivity(records: list[FaceRecord], pairs: list[Pair]) -> None:
    print("\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY (how grouping changes with the EXPERIMENTAL cutoff)")
    print("=" * 70)
    print("  Shows whether the embedding scores separate cleanly, or whether the")
    print("  result is an artefact of one arbitrary threshold.")
    print("  threshold | cross-image edges | multi-face groups | single-occurrence")
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        groups = build_groups(records, pairs, threshold)
        edges = sum(1 for p in pairs if p.similarity >= threshold)
        multi = sum(1 for g in groups if len(g) > 1)
        singles = sum(1 for g in groups if len(g) == 1)
        marker = "  <- default" if abs(threshold - MATCH_THRESHOLD) < 1e-9 else ""
        print(f"  {threshold:.2f}      | {edges:17d} | {multi:17d} | {singles:15d}{marker}")


def print_embedding_info(records: list[FaceRecord]) -> int:
    print("\n" + "=" * 70)
    print("EMBEDDING INFORMATION")
    print("=" * 70)
    if not records:
        print("  (no embeddings generated)")
        return 0
    dim = int(records[0].embedding.shape[0])
    # Verify the actual InsightFace behaviour instead of assuming normalisation.
    norms = np.array([float(np.linalg.norm(r.embedding)) for r in records])
    normalised = bool(np.allclose(norms, 1.0, atol=1e-3))
    print(f"  Embeddings generated: {len(records)}")
    print(f"  Embedding dimension: {dim}")
    print(f"  Embedding dtype: {records[0].embedding.dtype}")
    print(f"  Observed L2 norm range: {norms.min():.4f} .. {norms.max():.4f}")
    print(f"  Already unit-normalised (norm ~= 1)? {normalised}")
    print("  Cosine similarity is computed defensively as dot(a,b)/(||a||*||b||),")
    print("  so it is valid whether or not the vectors are pre-normalised.")
    return dim


def print_final_summary(
    results: list[ImageResult],
    records: list[FaceRecord],
    groups: list[list[FaceRecord]],
    pairs: list[Pair],
    threshold: float,
    margin: float,
    embedding_dim: int,
) -> None:
    total_detected = sum(r.total_detected for r in results)
    total_after = sum(r.after_filtering for r in results)
    duplicates_removed = sum(r.duplicates_removed for r in results)
    total_embeddings = len(records)
    single_groups = sum(1 for g in groups if len(g) == 1)
    low = threshold - margin
    ambiguous_count = sum(1 for p in pairs if low <= p.similarity < threshold)

    print("\n" + "=" * 70)
    print("FINAL FACE MATCHING SUMMARY")
    print("=" * 70)
    kept_count = sum(1 for r in records if r.status == P.STATUS_KEEP)
    uncertain_count = sum(1 for r in records if r.status == P.STATUS_UNCERTAIN)
    promoted_count = sum(1 for r in records if r.promoted)
    print(f"Photos processed: {len(results)}")
    print(f"Total faces detected: {total_detected}")
    print(f"Duplicate detections suppressed (same face twice): {duplicates_removed}")
    print(f"Total faces after filtering: {total_after}")
    print(f"Total embeddings generated: {total_embeddings}")
    print(f"KEEP faces: {kept_count}")
    print(f"UNCERTAIN faces: {uncertain_count} "
          f"(promoted to CONFIRMED_FROM_MATCH: {promoted_count})")
    print(f"Experimental face groups: {len(groups)}")
    print(f"Single-occurrence groups: {single_groups}")
    print(f"Potential ambiguous matches: {ambiguous_count}")
    print(f"Embedding dimension: {embedding_dim}")
    print("\nEXPERIMENTAL GROUPS")
    for gid, members in enumerate(groups, start=1):
        suffix = "  [single-occurrence: no match in other images]" if len(members) == 1 else ""
        print(f"\nGroup {gid}:{suffix}")
        for member in members:
            print(f"  {member.image_filename} -> Face {member.face_index}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(
    groups: list[list[FaceRecord]], threshold: float, csv_path: Path = CSV_PATH
) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_id",
                "image_filename",
                "face_index",
                "similarity_to_group_reference",
                "detection_confidence",
                "relative_face_area",
                "quality_score",
                "yaw",
                "pitch",
            ]
        )
        for gid, members in enumerate(groups, start=1):
            reference = pick_reference(members)
            for member in members:
                if member is reference:
                    similarity = 1.0
                else:
                    similarity = P.compare_embeddings(member.embedding, reference.embedding)
                writer.writerow(
                    [
                        f"Group_{gid}",
                        member.image_filename,
                        member.face_index,
                        f"{similarity:.6f}",
                        f"{member.det_score:.6f}",
                        f"{member.relative_area:.6f}",
                        f"{member.blur_score:.4f}",
                        "" if member.yaw is None else f"{member.yaw:.4f}",
                        "" if member.pitch is None else f"{member.pitch:.4f}",
                    ]
                )
    return csv_path


def write_uncertain_recovery_csv(
    records: list[FaceRecord],
    threshold: float = UNCERTAIN_PROMOTION_THRESHOLD,
    csv_path: Path = UNCERTAIN_CSV_PATH,
) -> Path:
    """Write one row per examined UNCERTAIN face to ``confirmed_uncertain_matches.csv``.

    Promoted faces carry their matched KEEP photo/face and similarity; faces that
    did not clear the threshold keep empty match fields and ``final_status``
    ``UNCERTAIN``. FILTERED faces never appear (they are not UNCERTAIN).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    uncertain = [r for r in records if r.status == P.STATUS_UNCERTAIN]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "photo",
                "face_id",
                "original_status",
                "uncertainty_reason",
                "matched_photo",
                "matched_face_id",
                "similarity",
                "promotion_threshold",
                "final_status",
            ]
        )
        for record in uncertain:
            if record.matched_image_filename:
                # Strongest eligible cross-image KEEP candidate (promoted or not).
                matched_photo = record.matched_image_filename
                matched_face_id = str(record.matched_face_index)
                similarity = f"{record.promotion_similarity:.6f}"
            else:
                matched_photo = ""
                matched_face_id = ""
                similarity = ""
            writer.writerow(
                [
                    record.image_filename,
                    record.face_index,
                    record.original_status,
                    record.uncertainty_reason,
                    matched_photo,
                    matched_face_id,
                    similarity,
                    f"{threshold:.2f}",
                    record.final_status,
                ]
            )
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="Directory of event images.")
    parser.add_argument(
        "--glob",
        default="*",
        dest="glob_pattern",
        help="Glob pattern for images inside --dir (default: all images).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="ONNX Runtime provider (default: CoreML if available, else CPU).",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=P.DEFAULT_DET_SIZE,
        help="Detector input resolution (edge length). Default 640 matches the "
             "InsightFace default and preserves current behaviour; use e.g. 1280 "
             "to detect smaller background faces (default: %(default)s).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help="EXPERIMENTAL cosine-similarity threshold used to group faces "
             "(default: %(default)s).",
    )
    parser.add_argument("--ambiguous-margin", type=float,
        default=AMBIGUOUS_MARGIN,
        help="Band below the match threshold reported as ambiguous (default: %(default)s).",
    )
    parser.add_argument(
        "--uncertain-promotion-threshold",
        type=float,
        default=UNCERTAIN_PROMOTION_THRESHOLD,
        help="Separate cosine-similarity threshold used ONLY to promote an "
             "UNCERTAIN face to CONFIRMED_FROM_MATCH when it strongly matches a "
             "KEEP face from another image (default: %(default)s). Independent "
             "of --match-threshold.",
    )
    parser.add_argument("--top-n", type=int, default=TOP_MATCHES_LIMIT,
                        help="Number of strongest cross-image matches to list.")
    parser.add_argument("--min-relative-area", type=float, default=P.MIN_RELATIVE_FACE_AREA)
    parser.add_argument("--min-detection-confidence", type=float, default=P.MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--min-blur-score", type=float, default=P.MIN_BLUR_SCORE)
    parser.add_argument("--max-yaw", type=float, default=P.MAX_YAW)
    parser.add_argument("--max-pitch", type=float, default=P.MAX_PITCH)
    parser.add_argument("--min-ratio-to-largest-face", type=float,
                        default=P.MIN_RATIO_TO_LARGEST_FACE,
                        help="Perspective gate: minimum face area as a fraction of "
                             "the largest detected face (default: %(default)s).")
    parser.add_argument("--scale-keep-blur", type=float, default=P.SCALE_KEEP_BLUR)
    parser.add_argument("--scale-keep-confidence", type=float,
                        default=P.SCALE_KEEP_CONFIDENCE)
    parser.add_argument(
        "--debug-filtering",
        "--debug",
        dest="debug_filtering",
        action="store_true",
        help="Print full per-face metrics (relative area, detection confidence, "
             "quality/blur, yaw/pitch) and the KEEP/FILTERED decision for every "
             "detected face before matching runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    directory = Path(args.dir).expanduser()
    images = list_images(directory, args.glob_pattern)
    if not images:
        print(f"No images found in {directory} matching {args.glob_pattern!r}", file=sys.stderr)
        return 1

    providers = P.resolve_providers(args.provider) if args.provider else None
    t0 = time.perf_counter()
    app = P.get_model(providers=providers, det_size=args.det_size)
    load_seconds = time.perf_counter() - t0

    print("=" * 70)
    print("FACE MATCHING EXPERIMENT (EXPERIMENTAL — no final clustering)")
    print("=" * 70)
    print(f"Directory: {directory}")
    print(f"Glob: {args.glob_pattern}")
    print(f"Images found: {len(images)}")
    for image in images:
        print(f"  - {image.name}")
    print(f"Model: {P.MODEL_NAME}")
    print(f"Providers: {providers or P.DEFAULT_PROVIDERS}")
    print(f"Detection size: {args.det_size}x{args.det_size}")
    print(f"Model load time: {load_seconds:.2f}s")
    print("Filter thresholds (reused from predict.py, EXPERIMENTAL):")
    print(f"  MIN_RELATIVE_FACE_AREA   = {args.min_relative_area:.5f}")
    print(f"  MIN_DETECTION_CONFIDENCE = {args.min_detection_confidence:.2f}")
    print(f"  MIN_BLUR_SCORE           = {args.min_blur_score:.1f}")
    print(f"  MAX_YAW                  = {args.max_yaw:.1f} deg")
    print(f"  MAX_PITCH                = {args.max_pitch:.1f} deg")
    print(f"  MIN_RATIO_TO_LARGEST_FACE= {args.min_ratio_to_largest_face:.2f} "
          f"(perspective gate; SCALE_KEEP_BLUR={args.scale_keep_blur:.0f}, "
          f"SCALE_KEEP_CONFIDENCE={args.scale_keep_confidence:.2f})")
    print(f"Match threshold [EXPERIMENTAL] = {args.match_threshold:.2f}")

    filter_kwargs = dict(
        min_relative_area=args.min_relative_area,
        min_detection_confidence=args.min_detection_confidence,
        min_blur_score=args.min_blur_score,
        max_yaw=args.max_yaw,
        max_pitch=args.max_pitch,
        min_ratio_to_largest=args.min_ratio_to_largest_face,
        scale_keep_blur=args.scale_keep_blur,
        scale_keep_confidence=args.scale_keep_confidence,
    )

    results: list[ImageResult] = []
    for image_path in images:
        results.append(process_image(image_path, app, **filter_kwargs))

    records: list[FaceRecord] = [r for result in results for r in result.records]
    pairs = cross_image_pairs(records)
    groups_idx = build_groups(records, pairs, args.match_threshold)
    groups = [[records[i] for i in group] for group in groups_idx]

    # Uncertain-face recovery: promote UNCERTAIN faces that strongly match a KEEP
    # face from another image (separate threshold, applied AFTER grouping so it
    # never influences the experimental groups above).
    promotions = recover_uncertain_faces(records, args.uncertain_promotion_threshold)

    print_image_blocks(results, debug=args.debug_filtering)
    invariants_ok = verify_pipeline_invariants(results)
    if not invariants_ok:
        print("WARNING: filtering invariant FAILED — a filtered face received an "
              "embedding. Matching results below are NOT trustworthy.", file=sys.stderr)
    embedding_dim = print_embedding_info(records)
    print_similarity_distribution(pairs)
    print_threshold_sensitivity(records, pairs)
    print_tracking_report(groups, pairs, args.match_threshold)
    print_top_matches(pairs, args.top_n)
    print_ambiguous_matches(pairs, args.match_threshold, args.ambiguous_margin)
    print_single_occurrence(groups)
    print_uncertain_recovery_report(records, promotions, args.uncertain_promotion_threshold)

    # Recovery safety checks (report the ACTUAL outcome, never assume).
    same_image_matches = sum(
        1 for r in records
        if r.promoted and r.matched_image_filename == r.image_filename
    )
    filtered_embedded = sum(
        1 for r in records
        if r.status == P.STATUS_FILTERED and r.embedding is not None
    )
    print("\n  Recovery invariants:")
    print(f"    same-image matches used: {same_image_matches} "
          f"({'PASS' if same_image_matches == 0 else 'FAIL'})")
    print(f"    FILTERED faces with embeddings: {filtered_embedded} "
          f"({'PASS' if filtered_embedded == 0 else 'FAIL'})")

    csv_path = write_csv(groups, args.match_threshold)
    print(f"\nCSV written to: {csv_path.relative_to(PROJECT_ROOT)}")
    uncertain_csv_path = write_uncertain_recovery_csv(
        records, args.uncertain_promotion_threshold
    )
    print(f"Uncertain recovery CSV written to: "
          f"{uncertain_csv_path.relative_to(PROJECT_ROOT)}")

    print_final_summary(
        results, records, groups, pairs, args.match_threshold, args.ambiguous_margin, embedding_dim
    )
    print("\nSTOP — this is the end of the MATCHING experiment (grouping only). "
          "The FINAL stage (person clustering, anonymous Person IDs, Person-Photo "
          "records and the custom Merge Sort) is implemented separately — run "
          "`python run.py` or `ml/inference/cluster_people.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
