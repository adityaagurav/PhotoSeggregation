"""Coordinates the FULL final pipeline across a batch of photographs.

This is the application-level orchestrator. It REUSES the verified ML pipeline
(:mod:`ml.inference.match_faces`, which itself reuses
:mod:`ml.inference.predict`) unchanged and adds the final DAA stages:

    detection -> filtering -> embeddings -> uncertain recovery
              -> centroid person clustering -> anonymous Person IDs
              -> Person-Photo records -> custom Merge Sort
              -> person-wise galleries + machine-readable outputs

Nothing about detection, filtering, embedding or uncertain recovery is
re-implemented here; those functions are imported from the verified modules.

Entry points:
* ``python run.py`` / ``python app/main.py``  (application entry point)
* ``python ml/inference/cluster_people.py``   (ML-side convenience runner)

Both delegate to :func:`run_cli` here, so there is a single implementation.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow running this module directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import match_faces as M  # noqa: E402
from ml.inference import predict as P  # noqa: E402

from app.ai.face_cluster import (  # noqa: E402
    FINAL_CLUSTER_THRESHOLD,
    ClusteringResult,
    cluster_faces,
    select_usable_faces,
)
from app.ai.singleton_qualifier import (  # noqa: E402
    DEFAULT_SINGLETON_THRESHOLDS,
    SingletonQualification,
    qualify_singletons,
)
from app.algorithms.merge_sort import merge_sort, person_id_sort_key  # noqa: E402
from app.processing.person_manager import PersonManager, PersonIdentity  # noqa: E402
from app.processing.record_generator import (  # noqa: E402
    PersonPhotoRecord,
    build_photo_index,
    build_photo_people_index,
    generate_records,
    write_person_clusters_json,
    write_person_photo_records_csv,
    write_singleton_validation_csv,
)

DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
PERSON_CLUSTERS_JSON = DATA_OUTPUT_DIR / "person_clusters.json"
PERSON_PHOTO_RECORDS_CSV = DATA_OUTPUT_DIR / "person_photo_records.csv"
CONFIRMED_UNCERTAIN_CSV = DATA_OUTPUT_DIR / "confirmed_uncertain_matches.csv"
SINGLETON_VALIDATION_CSV = DATA_OUTPUT_DIR / "singleton_validation.csv"

ANONYMOUS_ID_RE = re.compile(r"^Person #\d+$")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Everything the full pipeline produced, for reporting and reuse."""

    directory: Path
    glob_pattern: str
    image_count: int
    providers: list[str]
    det_size: int
    load_seconds: float
    cluster_threshold: float

    results: list[Any] = field(default_factory=list)          # ImageResult list
    records: list[Any] = field(default_factory=list)          # FaceRecord list
    promotions: list[Any] = field(default_factory=list)
    usable: list[Any] = field(default_factory=list)
    clustering: ClusteringResult | None = None
    persons: list[PersonIdentity] = field(default_factory=list)

    records_before: list[PersonPhotoRecord] = field(default_factory=list)
    records_sorted: list[PersonPhotoRecord] = field(default_factory=list)
    photo_index: dict[str, list[str]] = field(default_factory=dict)
    photo_people: dict[str, list[str]] = field(default_factory=dict)
    unresolved: list[dict[str, Any]] = field(default_factory=list)

    singleton: SingletonQualification | None = None
    json_path: Path = PERSON_CLUSTERS_JSON
    csv_path: Path = PERSON_PHOTO_RECORDS_CSV
    confirmed_csv_path: Path = CONFIRMED_UNCERTAIN_CSV
    singleton_csv_path: Path = SINGLETON_VALIDATION_CSV

    cluster_seconds: float = 0.0
    sort_seconds: float = 0.0
    filtered_with_embeddings: int = 0

    # Convenience counts
    @property
    def photos(self) -> int:
        return len(self.results)

    @property
    def detected(self) -> int:
        return sum(result.total_detected for result in self.results)

    @property
    def retained(self) -> int:
        return len(self.records)

    @property
    def kept(self) -> int:
        return sum(1 for r in self.records if r.status == P.STATUS_KEEP)

    @property
    def confirmed(self) -> int:
        return sum(
            1 for r in self.records if r.final_status == M.STATUS_CONFIRMED_FROM_MATCH
        )

    @property
    def uncertain(self) -> list[Any]:
        return [r for r in self.records if r.status == P.STATUS_UNCERTAIN]

    @property
    def multi_photo_persons(self) -> int:
        """Persons whose gallery spans more than one photo (cross-photo evidence)."""
        return sum(1 for photos in self.photo_index.values() if len(photos) > 1)

    @property
    def accepted_singletons(self) -> int:
        return 0 if self.singleton is None else len(self.singleton.accepted)

    @property
    def rejected_singletons(self) -> int:
        return 0 if self.singleton is None else len(self.singleton.rejected)

    @property
    def duplicates_removed(self) -> int:
        """Boxes dropped at detection time as the same physical face seen twice."""
        return sum(result.duplicates_removed for result in self.results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo-seggregation",
        description="Run the final photo pipeline: detection -> filtering -> "
                    "embeddings -> uncertain recovery -> person clustering -> "
                    "anonymous Person IDs -> Merge Sort -> galleries.",
    )
    parser.add_argument("--dir", default=str(M.DEFAULT_DIR),
                        help="Directory of event images (default: ml/datasets/test).")
    parser.add_argument("--glob", default="*", dest="glob_pattern",
                        help="Glob pattern for images inside --dir (default: all).")
    parser.add_argument("--provider", default=None,
                        help="ONNX Runtime provider override.")
    parser.add_argument("--det-size", type=int, default=P.DEFAULT_DET_SIZE,
                        help="Detector input resolution (default: %(default)s).")
    parser.add_argument("--cluster-threshold", type=float, default=FINAL_CLUSTER_THRESHOLD,
                        help="Cosine-similarity threshold for FINAL person clustering "
                             "(configurable, EXPERIMENTAL; default: %(default)s).")
    parser.add_argument("--out-dir", default=str(DATA_OUTPUT_DIR),
                        help="Directory for pipeline outputs (default: data/output). "
                             "Pass a job-specific directory to keep runs isolated.")
    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    """Execute the full pipeline for already-parsed ``args`` and write outputs.

    Raises ``FileNotFoundError`` when no images match.
    """
    directory = Path(args.dir).expanduser()
    images = M.list_images(directory, args.glob_pattern)
    if not images:
        raise FileNotFoundError(
            f"No images found in {directory} matching {args.glob_pattern!r}"
        )

    # Output paths: default to data/output, or a job-specific directory when
    # ``--out-dir`` is supplied (used by the local API for job isolation).
    out_dir = Path(getattr(args, "out_dir", DATA_OUTPUT_DIR)).expanduser()
    if not out_dir.is_absolute():
        # Keep outputs inside the project (and make print_report's relative_to work).
        out_dir = PROJECT_ROOT / out_dir
    json_path = out_dir / "person_clusters.json"
    csv_path = out_dir / "person_photo_records.csv"
    confirmed_csv_path = out_dir / "confirmed_uncertain_matches.csv"
    singleton_csv_path = out_dir / "singleton_validation.csv"

    providers = P.resolve_providers(args.provider) if args.provider else None
    t0 = time.perf_counter()
    app = P.get_model(providers=providers, det_size=args.det_size)
    load_seconds = time.perf_counter() - t0

    # Reuse the verified pipeline exactly as match_faces.py does.
    filter_kwargs = dict(
        min_relative_area=P.MIN_RELATIVE_FACE_AREA,
        min_detection_confidence=P.MIN_DETECTION_CONFIDENCE,
        min_blur_score=P.MIN_BLUR_SCORE,
        max_yaw=P.MAX_YAW,
        max_pitch=P.MAX_PITCH,
        min_ratio_to_largest=P.MIN_RATIO_TO_LARGEST_FACE,
        scale_keep_blur=P.SCALE_KEEP_BLUR,
        scale_keep_confidence=P.SCALE_KEEP_CONFIDENCE,
    )
    results = [M.process_image(image_path, app, **filter_kwargs) for image_path in images]
    records = [record for result in results for record in result.records]

    # Uncertain-face recovery (existing, unchanged; owns its own CSV).
    promotions = M.recover_uncertain_faces(records, M.UNCERTAIN_PROMOTION_THRESHOLD)
    confirmed_csv_path = M.write_uncertain_recovery_csv(
        records, M.UNCERTAIN_PROMOTION_THRESHOLD, confirmed_csv_path
    )

    # Select usable faces: KEEP + CONFIRMED_FROM_MATCH only.
    usable = select_usable_faces(records)

    # Final centroid-based clustering.
    t_cluster = time.perf_counter()
    clustering = cluster_faces(usable, args.cluster_threshold)
    cluster_seconds = time.perf_counter() - t_cluster

    # Strict SINGLETON qualification (post-clustering; does not change clustering).
    # A cluster from a single photo must pass a stricter quality gate before it
    # becomes a Person; multi-photo clusters are left untouched.
    singleton = qualify_singletons(clustering.clusters)
    persons = PersonManager().assign(singleton.retained_clusters)
    retained_ids = {id(cluster) for cluster in singleton.retained_clusters}
    kept_assignments = [
        assignment
        for assignment in clustering.assignments
        if id(assignment.cluster) in retained_ids
    ]

    # Person-Photo records (face-processing order) then the custom Merge Sort.
    records_before = generate_records(persons, kept_assignments)
    t_sort = time.perf_counter()
    records_sorted = merge_sort(records_before)
    sort_seconds = time.perf_counter() - t_sort

    photo_index = build_photo_index(records_sorted)
    photo_people = build_photo_people_index(records_sorted)

    # Unresolved UNCERTAIN faces are preserved (never silently dropped).
    unresolved = [
        {
            "photo": record.image_filename,
            "face_id": record.face_index,
            "original_status": record.original_status,
            "reason": record.uncertainty_reason,
            "final_status": record.final_status,
        }
        for record in records
        if record.status == P.STATUS_UNCERTAIN
    ]

    metadata = {
        "cluster_threshold": args.cluster_threshold,
        "duplicate_detections_removed": sum(
            result.duplicates_removed for result in results
        ),
        "uncertain_promotion_threshold": M.UNCERTAIN_PROMOTION_THRESHOLD,
        "singleton_thresholds": {
            "min_relative_face_area": DEFAULT_SINGLETON_THRESHOLDS.min_relative_face_area,
            "min_detection_confidence": DEFAULT_SINGLETON_THRESHOLDS.min_detection_confidence,
            "min_blur_score": DEFAULT_SINGLETON_THRESHOLDS.min_blur_score,
            "max_yaw": DEFAULT_SINGLETON_THRESHOLDS.max_yaw,
            "max_pitch": DEFAULT_SINGLETON_THRESHOLDS.max_pitch,
        },
        "det_size": args.det_size,
        "photos": len(results),
    }
    json_path = write_person_clusters_json(
        persons, photo_index, json_path,
        unresolved_uncertain=unresolved, metadata=metadata,
    )
    csv_path = write_person_photo_records_csv(records_sorted, csv_path)
    singleton_csv_path = write_singleton_validation_csv(
        singleton.decisions, singleton_csv_path
    )

    filtered_with_embeddings = sum(
        1 for result in results for face in result.faces
        if face.status == P.STATUS_FILTERED and face.embedding is not None
    )

    return PipelineResult(
        directory=directory,
        glob_pattern=args.glob_pattern,
        image_count=len(images),
        providers=providers or P.DEFAULT_PROVIDERS,
        det_size=args.det_size,
        load_seconds=load_seconds,
        cluster_threshold=args.cluster_threshold,
        results=results,
        records=records,
        promotions=promotions,
        usable=usable,
        clustering=clustering,
        singleton=singleton,
        persons=persons,
        records_before=records_before,
        records_sorted=records_sorted,
        photo_index=photo_index,
        photo_people=photo_people,
        unresolved=unresolved,
        json_path=json_path,
        csv_path=csv_path,
        confirmed_csv_path=confirmed_csv_path,
        singleton_csv_path=singleton_csv_path,
        cluster_seconds=cluster_seconds,
        sort_seconds=sort_seconds,
        filtered_with_embeddings=filtered_with_embeddings,
    )


# ---------------------------------------------------------------------------
# Reporting (required tests)
# ---------------------------------------------------------------------------


def print_report(result: PipelineResult) -> None:
    """Print the pipeline header, the seven required checks and the final summary."""
    clustering = result.clustering

    print("=" * 72)
    print("FINAL PERSON CLUSTERING + ANONYMOUS PERSON IDs + MERGE SORT")
    print("=" * 72)
    print(f"Directory: {result.directory}")
    print(f"Glob: {result.glob_pattern}")
    print(f"Images found: {result.image_count}")
    print(f"Model: {P.MODEL_NAME}   Providers: {result.providers}")
    print(f"Detection size: {result.det_size}x{result.det_size}   "
          f"Model load: {result.load_seconds:.2f}s")
    print(f"Final cluster threshold [EXPERIMENTAL, configurable] = "
          f"{result.cluster_threshold:.2f}")
    print(f"Uncertain recovery threshold (unchanged) = "
          f"{M.UNCERTAIN_PROMOTION_THRESHOLD:.2f}")
    st = DEFAULT_SINGLETON_THRESHOLDS
    print("Singleton gate [strict; only when no cross-photo evidence] = "
          f"area>={st.min_relative_face_area:.5f}, "
          f"conf>={st.min_detection_confidence:.2f}, "
          f"blur>={st.min_blur_score:.1f}, "
          f"|yaw|<={st.max_yaw:.0f}, |pitch|<={st.max_pitch:.0f}")

    print("\n" + "=" * 72)
    print("TEST 1 — FINAL CLUSTERING")
    print("=" * 72)
    print(f"Usable faces: {len(result.usable)}")
    print(f"Persons identified: {len(result.persons)}")
    print(f"Unresolved uncertain faces: {len(result.unresolved)}")
    print(f"Clustering time: {result.cluster_seconds * 1000:.1f} ms")
    for person in result.persons:
        print(f"\n{person.person_id}:")
        for photo in result.photo_index.get(person.person_id, []):
            print(f"    {photo}")

    print("\n" + "=" * 72)
    print("TEST 2 — MULTI-PERSON PHOTO CHECK")
    print("=" * 72)
    multi = {p: people for p, people in result.photo_people.items() if len(people) > 1}
    if multi:
        for photo, people in multi.items():
            print(f"{photo}:")
            for person_id in people:
                print(f"    {person_id}")
    else:
        print("No multi-person photo assignment observed in current test data.")

    print("\n" + "=" * 72)
    print("TEST 3 — SAME-PHOTO MATCHING")
    print("=" * 72)
    print(f"Same-photo comparisons used: {clustering.same_photo_comparisons}")
    print("PASS" if clustering.same_photo_comparisons == 0 else "FAIL")
    print(f"(clusters skipped because they only had same-photo members: "
          f"{clustering.skipped_same_photo_only})")

    print("\n" + "=" * 72)
    print("TEST 4 — MERGE SORT (custom recursive; no sorted()/list.sort)")
    print("=" * 72)
    print(f"Records before Merge Sort: {len(result.records_before)}")
    for record in result.records_before:
        print(f"    {record.person_id:>10}  ->  {record.photo}  (face {record.face_id})")
    print("\nAfter Merge Sort:")
    for record in result.records_sorted:
        print(f"    {record.person_id:>10}  ->  {record.photo}  (face {record.face_id})")
    ordered = all(
        person_id_sort_key(result.records_sorted[i].person_id)
        <= person_id_sort_key(result.records_sorted[i + 1].person_id)
        for i in range(len(result.records_sorted) - 1)
    )
    print(f"\nRecords ordered by Person ID: {'PASS' if ordered else 'FAIL'}")
    print(f"Sort time: {result.sort_seconds * 1000:.3f} ms")
    print(f"Complexity: time O(n log n), auxiliary space O(n), "
          f"n = {len(result.records_sorted)}")

    print("\n" + "=" * 72)
    print("TEST 5 — FILTERED FACE INTEGRITY")
    print("=" * 72)
    print(f"Filtered faces with embeddings: {result.filtered_with_embeddings}")
    print("PASS" if result.filtered_with_embeddings == 0 else "FAIL")

    print("\n" + "=" * 72)
    print("TEST 6 — UNCERTAIN RECOVERY")
    print("=" * 72)
    print(f"Uncertain faces examined: {len(result.uncertain)}")
    print(f"Promoted: {len(result.promotions)}")
    print(f"Remaining uncertain: {len(result.uncertain) - len(result.promotions)}")
    for record in result.uncertain:
        if record.promoted:
            print(f"  {record.image_filename} Face {record.face_index}: "
                  f"PROMOTED -> CONFIRMED_FROM_MATCH "
                  f"(match {record.matched_image_filename} Face {record.matched_face_index}, "
                  f"similarity {record.promotion_similarity:.4f})")
        else:
            print(f"  {record.image_filename} Face {record.face_index}: "
                  f"remains UNCERTAIN (best candidate "
                  f"{record.matched_image_filename or 'n/a'} similarity "
                  f"{record.promotion_similarity:.4f})")

    print("\n" + "=" * 72)
    print("TEST 7 — NO HARDCODED PERSON IDENTITIES")
    print("=" * 72)
    ids_ok = all(ANONYMOUS_ID_RE.match(person.person_id) for person in result.persons)
    print(f"All Person IDs dynamically generated as 'Person #N': "
          f"{'PASS' if ids_ok else 'FAIL'}")
    print("No names, filenames-as-identities or known test identities are used.")

    singleton = result.singleton
    print("\n" + "=" * 72)
    print("SINGLETON PERSON QUALIFICATION")
    print("=" * 72)
    print(f"Singleton clusters examined: {singleton.examined}")
    print(f"Accepted singleton persons: {len(singleton.accepted)}")
    print(f"Rejected singleton faces: {len(singleton.rejected)}")
    print("Strict gate applies ONLY when there is no cross-photo evidence; "
          "multi-photo persons keep all their faces as-is.")
    if singleton.examined == 0:
        print("\nNo singleton clusters observed in current test data.")
    person_by_cluster = {id(person.cluster): person.person_id for person in result.persons}
    for decision in singleton.decisions:
        person_id = person_by_cluster.get(id(decision.cluster))
        print(f"\n{person_id if person_id else decision.photo}:")
        if person_id:
            print(f"    {decision.photo}")
        print(f"    Face {decision.face_id}")
        print(f"    area: {decision.relative_area:.6f}")
        print(f"    confidence: {decision.det_score:.4f}")
        print(f"    blur: {decision.blur_score:.2f}")
        print(f"    yaw: {'n/a' if decision.yaw is None else format(decision.yaw, '.2f')}")
        print(f"    pitch: {'n/a' if decision.pitch is None else format(decision.pitch, '.2f')}")
        print(f"    status: {decision.status}")
        if decision.reasons:
            print("    reasons:")
            for reason in decision.reasons:
                print(f"        {reason}")

    print("\n" + "=" * 40)
    print("FINAL PERSON CLUSTERING RESULT")
    print("=" * 40)
    print(f"Photos processed: {result.photos}")
    print(f"Faces detected: {result.detected}")
    print(f"Duplicate detections suppressed: {result.duplicates_removed}")
    print(f"Faces retained: {result.retained}")
    print(f"Embeddings: {result.retained}")
    print()
    print(f"KEEP: {result.kept}")
    print(f"CONFIRMED_FROM_MATCH: {result.confirmed}")
    print(f"UNCERTAIN: {len(result.uncertain)}")
    print()
    print(f"Persons identified: {len(result.persons)}")
    print(f"Multi-photo persons: {result.multi_photo_persons}")
    print(f"Accepted singleton persons: {result.accepted_singletons}")
    print(f"Rejected singleton faces: {result.rejected_singletons}")
    print()
    print(f"Person-Photo records: {len(result.records_sorted)}")
    print(f"Final Person IDs: {len(result.persons)}")
    print()
    print(f"Same-photo comparisons: {clustering.same_photo_comparisons}")
    print(f"Filtered faces with embeddings: {result.filtered_with_embeddings}")
    print()
    ordered = all(
        person_id_sort_key(result.records_sorted[i].person_id)
        <= person_id_sort_key(result.records_sorted[i + 1].person_id)
        for i in range(len(result.records_sorted) - 1)
    )
    print("Merge Sort:")
    print(f"    Records sorted: {len(result.records_sorted)}")
    print("    Complexity: O(n log n)")
    print(f"    Status: {'PASS' if ordered else 'FAIL'}")
    print()
    print("Output:")
    print(f"    {result.json_path.relative_to(PROJECT_ROOT)}")
    print(f"    {result.csv_path.relative_to(PROJECT_ROOT)}")
    print(f"    {result.confirmed_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"    {result.singleton_csv_path.relative_to(PROJECT_ROOT)}")
    print("=" * 40)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_cli(argv: list[str] | None = None) -> int:
    """Parse ``argv``, run the pipeline, print the report; return an exit code."""
    args = build_parser().parse_args(argv)
    try:
        result = run_pipeline(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print_report(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Application entry point used by ``run.py`` / ``app/main.py``."""
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
