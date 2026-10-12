"""Final person clustering using a representative/centroid-based approach.

Why NOT connected components / single linkage
---------------------------------------------
The earlier *experimental* grouping used connected components over cross-image
edges >= 0.50. That over-merges through transitive chaining::

    A <-> B = 0.82      (edge kept)
    B <-> C = 0.76      (edge kept)
    A <-> C = 0.48      (weak, but A and C are still connected through B)

A single-linkage / connected-component method therefore puts A, B and C in one
person even though A and C are barely similar. The final system instead groups
faces by their similarity to a **cluster representative (centroid)**, which is a
stricter, non-chaining rule.

Algorithm (incremental)
-----------------------
1. Start with no person clusters.
2. Process each usable face embedding in a deterministic order.
3. Compare the face against the representative of every eligible cluster.
4. If the best similarity is ``>= FINAL_CLUSTER_THRESHOLD`` -> join that cluster.
5. Otherwise -> start a new anonymous cluster.

Same-photo rule
---------------
Two faces from the **same photo** must never be used to decide that they are the
same person. When evaluating a face from photo *P*, a cluster is only eligible if
it already contains a member from a **different** photo, and the representative
is recomputed **excluding** photo *P*. A cluster holding only same-photo members
is skipped entirely, so same-photo comparisons are never used as evidence
(``same_photo_comparisons`` stays 0).

Centroid drift (known limitation, documented NOT hidden)
--------------------------------------------------------
The representative is the L2-normalised mean of its members' (already
unit-length) embeddings. Adding a borderline face nudges the centroid toward
that face; with small clusters a single questionable face can move it noticeably.
Re-normalising keeps the representative unit-length, but it does **not** prevent
drift. Two consequences to be aware of:

* a face can occasionally join a cluster even though it is only weakly similar to
  the founding member, because the centroid has drifted toward it; and
* the result depends on the processing order of faces.

This is the trade-off accepted for a non-chaining rule; the threshold is
configurable and must be validated against manually verified groups before being
frozen. See ``docs/results/person_clustering_and_merge_sort.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.ai.face_matcher import cosine_similarity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Final clustering cosine-similarity threshold. EXPERIMENTAL / configurable, NOT
# claimed to be optimal. Initial value 0.60 (was 0.50 only in the experimental
# connected-component grouping). Rationale: on the 10-photo test set the number
# of cross-image edges falls 44 (0.50) -> 36 (0.60) -> 25 (0.65), i.e. 0.60 keeps
# the clearly-separated matches while discarding most of the 0.40-0.55 band that
# contained the ambiguous pairs. Override with --cluster-threshold or the
# FINAL_CLUSTER_THRESHOLD env var.
FINAL_CLUSTER_THRESHOLD = float(os.getenv("FINAL_CLUSTER_THRESHOLD", "0.60"))

# Statuses allowed to take part in final clustering. Remaining UNCERTAIN faces
# (not promoted by the recovery step) are deliberately excluded from forming a
# person. FILTERED faces never have an embedding at all.
STATUS_KEEP = "KEEP"
STATUS_CONFIRMED_FROM_MATCH = "CONFIRMED_FROM_MATCH"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_FILTERED = "FILTERED"
PARTICIPATING_STATUSES = (STATUS_KEEP, STATUS_CONFIRMED_FROM_MATCH)


# ---------------------------------------------------------------------------
# Generic accessors (work with FaceRecord and with lightweight test objects)
# ---------------------------------------------------------------------------


def face_photo(face: Any) -> str:
    """Return the photo/filename a face belongs to."""
    photo = getattr(face, "photo", None)
    if photo is None:
        photo = getattr(face, "image_filename", None)
    return photo


def face_face_id(face: Any) -> int:
    """Return the temporary per-image face index of a face."""
    for attr in ("face_id", "face_index"):
        value = getattr(face, attr, None)
        if value is not None:
            return value
    return -1


def face_status(face: Any) -> str:
    """Return the effective (``final_status`` if set, else ``status``) label."""
    final = getattr(face, "final_status", None)
    if final:
        return final
    return getattr(face, "status", "")


def face_embedding(face: Any) -> np.ndarray:
    """Return the face embedding as a flat ``float32`` vector."""
    return np.asarray(face.embedding, dtype=np.float32).ravel()


def select_usable_faces(faces: list[Any]) -> list[Any]:
    """Faces allowed to participate in clustering (KEEP + CONFIRMED_FROM_MATCH).

    A face is usable when its effective status is one of
    :data:`PARTICIPATING_STATUSES`. Remaining ``UNCERTAIN`` faces are excluded
    from forming a person (but are preserved in the reports), and ``FILTERED``
    faces never have an embedding.
    """
    return [face for face in faces if face_status(face) in PARTICIPATING_STATUSES]


# ---------------------------------------------------------------------------
# Cluster + result data structures
# ---------------------------------------------------------------------------


@dataclass
class PersonCluster:
    """One anonymous person: the faces grouped into it so far."""

    members: list[Any] = field(default_factory=list)

    def members_from_other_photo(self, photo: str) -> list[Any]:
        return [m for m in self.members if face_photo(m) != photo]

    def has_cross_photo_member(self, photo: str) -> bool:
        """True when the cluster has at least one member from another photo."""
        return any(face_photo(m) != photo for m in self.members)

    def representative(self, exclude_photo: str | None = None) -> np.ndarray | None:
        """Unit-normalised centroid of the cluster members.

        ``exclude_photo`` removes every member from that photo before averaging,
        which is how same-photo evidence is kept out of the decision. Returns
        ``None`` when no members remain.
        """
        vectors = [
            face_embedding(m)
            for m in self.members
            if exclude_photo is None or face_photo(m) != exclude_photo
        ]
        if not vectors:
            return None
        centroid = np.mean(np.stack(vectors, axis=0), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0.0:
            return centroid
        return centroid / norm

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class FaceAssignment:
    """The decision made for one face during clustering."""

    face: Any
    cluster: PersonCluster
    similarity: float | None  # similarity to the chosen representative (None = new)
    is_new_cluster: bool


@dataclass
class ClusteringResult:
    """Everything the clustering produced, plus comparison statistics."""

    clusters: list[PersonCluster]
    assignments: list[FaceAssignment]
    comparisons: int = 0
    same_photo_comparisons: int = 0
    skipped_same_photo_only: int = 0


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_faces(
    faces: list[Any], threshold: float = FINAL_CLUSTER_THRESHOLD
) -> ClusteringResult:
    """Group usable faces into anonymous person clusters (incremental centroid).

    ``faces`` are compared across DIFFERENT photos only; see the module docstring
    for the same-photo rule. The order of ``faces`` affects the result (documented
    limitation), so callers should pass a deterministic ordering.
    """
    clusters: list[PersonCluster] = []
    assignments: list[FaceAssignment] = []
    comparisons = 0
    same_photo_comparisons = 0
    skipped_same_photo_only = 0

    for face in faces:
        photo = face_photo(face)
        embedding = face_embedding(face)

        best_cluster: PersonCluster | None = None
        best_similarity = -1.0

        for cluster in clusters:
            # Same-photo rule: only clusters with a member from a DIFFERENT photo
            # can provide evidence, and the representative excludes this photo.
            if not cluster.has_cross_photo_member(photo):
                skipped_same_photo_only += 1
                continue
            representative = cluster.representative(exclude_photo=photo)
            if representative is None:
                skipped_same_photo_only += 1
                continue
            similarity = cosine_similarity(embedding, representative)
            comparisons += 1
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None and best_similarity >= threshold:
            cluster = best_cluster
            similarity: float | None = best_similarity
            is_new_cluster = False
        else:
            cluster = PersonCluster()
            clusters.append(cluster)
            similarity = None
            is_new_cluster = True

        cluster.members.append(face)
        assignments.append(
            FaceAssignment(
                face=face,
                cluster=cluster,
                similarity=similarity,
                is_new_cluster=is_new_cluster,
            )
        )

    # ``same_photo_comparisons`` is never incremented: same-photo pairs are
    # structurally excluded above (kept as an explicit, reportable invariant).
    return ClusteringResult(
        clusters=clusters,
        assignments=assignments,
        comparisons=comparisons,
        same_photo_comparisons=same_photo_comparisons,
        skipped_same_photo_only=skipped_same_photo_only,
    )
