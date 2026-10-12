"""Strict qualification for SINGLETON person clusters (a person in one photo only).

Runs AFTER final clustering. It does not change the clustering algorithm: it only
decides whether a cluster that has **no cross-photo evidence** (all of its faces
come from a single photo) is reliable enough to become a final ``Person #N``.

Rules
-----
* A cluster with faces from **more than one photo** is left untouched — the strong
  cross-photo evidence is what makes a multi-photo person reliable, so the strict
  singleton rules are NEVER applied to the individual faces of a multi-photo
  person.
* A cluster whose faces all come from **one photo** (a singleton) must pass a
  strict quality gate, because there is no cross-photo evidence backing it.
* Remaining ``UNCERTAIN`` faces never reach this stage: they are already excluded
  from clustering, and only a ``CONFIRMED_FROM_MATCH`` face (promoted by the
  existing recovery mechanism) may participate.

Only metrics that already exist in the pipeline are used (all measured by the
verified detection/filtering stage and carried on each face record):
``relative_area``, ``det_score``, ``blur_score`` (Laplacian variance), ``yaw`` and
``pitch`` (InsightFace ``landmark_3d_68`` pose).

Face visibility: InsightFace's ``buffalo_l`` pack does NOT expose a direct
face-visibility / occlusion attribute, so none is invented here. Visibility is
approximated only by the available size, confidence, blur and pose signals, and
the absence of a true visibility metric is stated explicitly.

If a singleton fails the gate it is classified ``SINGLETON_REJECTED`` with the
exact reason(s). It is NOT deleted (the photo and face data are untouched) and it
does NOT receive a ``Person #N``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app.ai.face_cluster import face_face_id, face_photo

STATUS_ACCEPTED_SINGLETON = "ACCEPTED_SINGLETON"
STATUS_SINGLETON_REJECTED = "SINGLETON_REJECTED"

# Reason labels (exact strings recorded in singleton_validation.csv).
REASON_TOO_SMALL = "face_too_small"
REASON_LOW_CONFIDENCE = "low_detection_confidence"
REASON_BLUR = "excessive_blur"
REASON_YAW = "excessive_yaw"
REASON_PITCH = "excessive_pitch"
REASON_MULTIPLE = "multiple_quality_failures"


# ---------------------------------------------------------------------------
# Configurable thresholds (all EXPERIMENTAL, env-overridable)
# ---------------------------------------------------------------------------

# Singleton size floor: the face must occupy a sufficiently large fraction of
# the image (resolution-independent; never a pixel size). The spec's suggested
# starting value was 0.01, but inspecting this dataset showed 0.01 rejects three
# clearly-good foreground faces (event04.JPG Face 0 @0.0084, event04_9 Face 0
# @0.0023 and Face 1 @0.0029) that are frontal, sharp and high-confidence — they
# are only "small" in relative terms because those photos are ~12 MP. The floor
# is therefore derived from the EXISTING pipeline floor rather than curve-fit:
# it is 2x MIN_RELATIVE_FACE_AREA (0.001), i.e. strictly stricter than what the
# verified pipeline already accepts, while not discarding real foreground faces.
SINGLETON_MIN_RELATIVE_FACE_AREA = float(
    os.getenv("SINGLETON_MIN_RELATIVE_FACE_AREA", "0.002")
)
SINGLETON_MIN_DETECTION_CONFIDENCE = float(
    os.getenv("SINGLETON_MIN_DETECTION_CONFIDENCE", "0.80")
)
SINGLETON_MIN_BLUR_SCORE = float(os.getenv("SINGLETON_MIN_BLUR_SCORE", "80.0"))
SINGLETON_MAX_YAW = float(os.getenv("SINGLETON_MAX_YAW", "45.0"))
SINGLETON_MAX_PITCH = float(os.getenv("SINGLETON_MAX_PITCH", "30.0"))


@dataclass(frozen=True)
class SingletonThresholds:
    """Strict quality thresholds applied to singleton faces only."""

    min_relative_face_area: float = SINGLETON_MIN_RELATIVE_FACE_AREA
    min_detection_confidence: float = SINGLETON_MIN_DETECTION_CONFIDENCE
    min_blur_score: float = SINGLETON_MIN_BLUR_SCORE
    max_yaw: float = SINGLETON_MAX_YAW
    max_pitch: float = SINGLETON_MAX_PITCH


DEFAULT_SINGLETON_THRESHOLDS = SingletonThresholds()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SingletonDecision:
    """The qualification outcome for one singleton cluster (one photo)."""

    photo: str
    face_id: int
    relative_area: float
    det_score: float
    blur_score: float
    yaw: float | None
    pitch: float | None
    status: str
    reasons: list[str] = field(default_factory=list)
    cluster: Any = None

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED_SINGLETON

    @property
    def rejection_reason(self) -> str:
        """Semicolon-joined reason string for the CSV (empty when accepted)."""
        return ";".join(self.reasons)


@dataclass
class SingletonQualification:
    """Result of qualifying every cluster after final clustering."""

    retained_clusters: list[Any] = field(default_factory=list)
    multi_photo_clusters: list[Any] = field(default_factory=list)
    decisions: list[SingletonDecision] = field(default_factory=list)

    @property
    def accepted(self) -> list[SingletonDecision]:
        return [d for d in self.decisions if d.accepted]

    @property
    def rejected(self) -> list[SingletonDecision]:
        return [d for d in self.decisions if not d.accepted]

    @property
    def examined(self) -> int:
        return len(self.decisions)


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------


def cluster_photos(cluster: Any) -> list[str]:
    """Distinct photos a cluster appears in (in first-seen order)."""
    photos: list[str] = []
    for member in cluster.members:
        photo = face_photo(member)
        if photo not in photos:
            photos.append(photo)
    return photos


def is_singleton_cluster(cluster: Any) -> bool:
    """True when every face in the cluster comes from the same photo."""
    return len(cluster_photos(cluster)) <= 1


def evaluate_singleton_face(
    face: Any, thresholds: SingletonThresholds = DEFAULT_SINGLETON_THRESHOLDS
) -> SingletonDecision:
    """Apply the strict singleton gate to a single face; return the decision."""
    relative_area = float(getattr(face, "relative_area", 0.0) or 0.0)
    det_score = float(getattr(face, "det_score", 0.0) or 0.0)
    blur_score = float(getattr(face, "blur_score", 0.0) or 0.0)
    yaw = getattr(face, "yaw", None)
    pitch = getattr(face, "pitch", None)

    reasons: list[str] = []
    if relative_area < thresholds.min_relative_face_area:
        reasons.append(REASON_TOO_SMALL)
    if det_score < thresholds.min_detection_confidence:
        reasons.append(REASON_LOW_CONFIDENCE)
    if blur_score < thresholds.min_blur_score:
        reasons.append(REASON_BLUR)
    # Pose may be unavailable (``None``); it is only checked when measured, so an
    # unmeasured angle never fails a face on its own.
    if yaw is not None and abs(float(yaw)) > thresholds.max_yaw:
        reasons.append(REASON_YAW)
    if pitch is not None and abs(float(pitch)) > thresholds.max_pitch:
        reasons.append(REASON_PITCH)

    if reasons:
        if len(reasons) > 1:
            reasons.append(REASON_MULTIPLE)
        status = STATUS_SINGLETON_REJECTED
    else:
        status = STATUS_ACCEPTED_SINGLETON

    return SingletonDecision(
        photo=face_photo(face),
        face_id=face_face_id(face),
        relative_area=relative_area,
        det_score=det_score,
        blur_score=blur_score,
        yaw=None if yaw is None else float(yaw),
        pitch=None if pitch is None else float(pitch),
        status=status,
        reasons=reasons,
        cluster=None,
    )


def qualify_singletons(
    clusters: list[Any], thresholds: SingletonThresholds = DEFAULT_SINGLETON_THRESHOLDS
) -> SingletonQualification:
    """Split clusters into retained (multi-photo + accepted singletons) and rejected.

    ``retained_clusters`` preserves the original cluster order so that, after the
    rejected singletons are dropped, anonymous Person IDs stay contiguous and
    deterministic. Multi-photo clusters are always retained and their individual
    faces are never individually re-checked.
    """
    result = SingletonQualification()
    for cluster in clusters:
        if not is_singleton_cluster(cluster):
            result.multi_photo_clusters.append(cluster)
            result.retained_clusters.append(cluster)
            continue

        # Singleton: evaluate its single face (all faces come from one photo).
        face = cluster.members[0]
        decision = evaluate_singleton_face(face, thresholds)
        decision.cluster = cluster
        result.decisions.append(decision)
        if decision.accepted:
            result.retained_clusters.append(cluster)
        # Rejected singletons are intentionally NOT retained.
    return result
