"""Reusable face detection / embedding helpers built on InsightFace Buffalo_L.

This module is the ML foundation used by the application code and can also be
run directly as a command-line experiment:

    python ml/inference/predict.py ml/datasets/test/single_face.png
    python ml/inference/predict.py ml/datasets/test/group_faces.jpg
    python ml/inference/predict.py img_a.jpg img_b.jpg
    python ml/inference/predict.py img.jpg --min-relative-area 0.002
    python ml/inference/predict.py img.jpg --provider CPUExecutionProvider

Face relevance pipeline implemented here
----------------------------------------
    detect faces                     (detect_face_regions — no embeddings yet)
      -> duplicate-box suppression   (deduplicate_faces)          [same physical face]
      -> face-size filter            (filter_faces_by_size)      [resolution independent]
      -> face-quality filter         (filter_faces_by_quality)   [blur + detection confidence]
      -> head-orientation check      (filter_faces_by_pose)      [yaw/pitch — a WEAK signal]
      -> perspective/relative-scale  (filter_faces_by_relative_scale)
                                     [face area vs the largest detected face]
      -> KEEP / UNCERTAIN / FILTERED decision
      -> embeddings for KEEP (and, by default, UNCERTAIN) faces only
      -> report

Detection, filtering and embedding are separate steps on purpose: the
recognition model is only run on faces that survive filtering.

* **Size** uses each face's area relative to the whole image (never a fixed
  pixel size), because event photos come in many resolutions.
* **Quality** uses the Laplacian variance of a resolution-normalised face crop
  plus the detection confidence.
* **Head pose** is a genuinely available InsightFace signal: the ``buffalo_l``
  pack contains the ``landmark_3d_68`` model, whose ``get()`` sets
  ``face['pose'] = [pitch, yaw, roll]`` (degrees). It is a *weak* signal — a
  large, high-quality face that is looking sideways is still kept; only an
  extreme angle on a non-large face is filtered.

IMPORTANT: a person who appears in only one photograph is never removed for
that reason. Filtering only ever looks at a single face's own properties.

Design notes
------------
* Only a pretrained model pack (Buffalo_L) is used; nothing is trained here.
* Execution provider: CoreML is used when available, with CPU as the fallback
  (override with ``--provider`` or the ``INSIGHTFACE_PROVIDERS`` env var).
* All thresholds live in one configuration block below and are overridable via
  environment variables or CLI flags. They are EXPERIMENTAL and must be tuned
  on real event photographs (see docs/results/face_detection_embedding_test.md).
* Full embedding vectors are never printed; only dimensions and a few values.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.app.common import Face

# ---------------------------------------------------------------------------
# Configuration  (every threshold is overridable + experimental)
# ---------------------------------------------------------------------------

# Pretrained model pack and its on-disk cache root (kept outside the repo).
MODEL_NAME = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
MODEL_ROOT = os.getenv("INSIGHTFACE_ROOT", "~/.insightface")


def resolve_providers(override: str | list[str] | None = None) -> list[str]:
    """Pick ONNX Runtime providers: explicit override > env var > CoreML > CPU.

    CoreML is preferred when the runtime exposes it (Apple Silicon); CPU is the
    always-available fallback.
    """
    if override:
        return [override] if isinstance(override, str) else list(override)
    env = [p.strip() for p in os.getenv("INSIGHTFACE_PROVIDERS", "").split(",") if p.strip()]
    if env:
        return env
    available = onnxruntime.get_available_providers()
    if "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


DEFAULT_PROVIDERS = resolve_providers()

# --- Detection -----------------------------------------------------------------
DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", "0.5"))

# Detector input resolution (edge length). 640 matches the InsightFace default,
# so leaving it unset preserves the exact previous behaviour. A larger value
# (e.g. 1280) lets the detector find smaller/background faces in high-resolution
# photos at the cost of speed. Override with --det-size or INSIGHTFACE_DET_SIZE.
DEFAULT_DET_SIZE = int(os.getenv("INSIGHTFACE_DET_SIZE", "640"))

# Cosine-similarity threshold for "same person" (tuned experimentally later).
FACE_SIMILARITY_THRESHOLD = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.5"))

# --- Size gate (kept from the previous stage) ----------------------------------
# Minimum face area as a FRACTION OF THE WHOLE IMAGE for a face to be treated as
# a main subject. Never a fixed pixel size: photos come in many resolutions, so
# relative area is used instead.
#
#   relative_area = (face_width * face_height) / (image_width * image_height)
#
# EXPERIMENTAL — a conservative starting point, not scientifically optimal.
MIN_RELATIVE_FACE_AREA = float(os.getenv("MIN_RELATIVE_FACE_AREA", "0.001"))

# --- Quality gate (new) --------------------------------------------------------
# Minimum detection confidence. Detection already runs at DETECTION_THRESHOLD,
# so this is an extra gate that drops the weakest detections.
#   observed: kept faces 0.74-0.92, filtered faces 0.57-0.72
# EXPERIMENTAL.
MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.60"))

# Minimum blur score (Laplacian variance) of a face crop. The crop is resized to
# BLUR_CROP_SIZE before measuring so the score is comparable across resolutions
# (a raw-crop variance is dominated by the crop's pixel size).
# EXPERIMENTAL — must be tuned on real photographs.
MIN_BLUR_SCORE = float(os.getenv("MIN_BLUR_SCORE", "15.0"))
BLUR_CROP_SIZE = int(os.getenv("BLUR_CROP_SIZE", "112"))

# --- Head-orientation gate (new, deliberately WEAK) ----------------------------
# Absurd-angle guard only: a face turned beyond these angles is treated as a
# strongly side-facing / incidental face. It is NOT "must look at the camera".
#   observed: a legitimate standing person in the test group photo has yaw -54.8,
#   so MAX_YAW is deliberately generous to avoid removing real participants.
MAX_YAW = float(os.getenv("MAX_YAW", "75.0"))
MAX_PITCH = float(os.getenv("MAX_PITCH", "40.0"))

# A face at/above this relative area is considered "clearly large". When such a
# face has an extreme pose it is marked UNCERTAIN (retained, flagged) rather
# than FILTERED, so a large legitimate person is never hard-dropped on pose.
STRONG_RELATIVE_FACE_AREA = float(os.getenv("STRONG_RELATIVE_FACE_AREA", "0.01"))

# Whether UNCERTAIN faces are embedded. Default True = keep them (conservative:
# the cost of losing a real person is higher than embedding one borderline face).
EMBED_UNCERTAIN = os.getenv("EMBED_UNCERTAIN", "1").lower() not in ("0", "false", "no")

# --- Perspective / relative-scale gate (new, ADDITIONAL signal) ----------------
# The size gate above compares a face to the WHOLE IMAGE. That is not enough for
# strong-perspective photos (a couple of dominant foreground people plus many
# small background faces): a small background face can still clear the
# image-relative floor. This gate adds a SECOND, perspective-aware size signal:
#
#     ratio_to_largest_face = face_area / max(face_area over all detected faces)
#
# It is deliberately NOT a hard "smaller than X% of the largest face => reject"
# rule. A face below the ratio is only dropped when it ALSO lacks the quality
# and/or pose evidence to be a real participant; a small-but-sharp, front-facing
# face becomes UNCERTAIN (kept + flagged) instead of being removed.
#
# EXPERIMENTAL — defaults chosen from measurements on the test event photos: all
# genuine foreground/group faces scored ratio >= 0.38, while only clearly
# incidental faces scored near 0.05.
MIN_RATIO_TO_LARGEST_FACE = float(os.getenv("MIN_RATIO_TO_LARGEST_FACE", "0.15"))
# A small face at/above these is still a plausible participant even though it is
# much smaller than the dominant face (kept as UNCERTAIN, not filtered).
SCALE_KEEP_BLUR = float(os.getenv("SCALE_KEEP_BLUR", "100.0"))
SCALE_KEEP_CONFIDENCE = float(os.getenv("SCALE_KEEP_CONFIDENCE", "0.80"))
# A small face turned beyond these is treated as strongly incidental.
SCALE_MAX_YAW = float(os.getenv("SCALE_MAX_YAW", "45.0"))
SCALE_MAX_PITCH = float(os.getenv("SCALE_MAX_PITCH", "30.0"))

# --- Duplicate / overlapping detection suppression (inference-time) ------------
# The RetinaFace detector already runs NMS inside the model, but that NMS is a
# fixed IoU (0.4) inside the ONNX wrapper and cannot be tightened from here.
# Any box it leaves behind for one physical face (an offset or re-scaled
# response of the SAME face) would otherwise be carried through as its own
# "face" — and, because faces from one photo are deliberately never compared
# during clustering, such a box can never be merged with the real one later.
# It would become a second Person (and a second Person-Photo record) for one
# person.
#
# These four numbers are geometry only (resolution independent) and fixed — they
# are NOT detection/filtering thresholds and are intentionally not exposed as
# CLI flags:
#   * FACE_DUPLICATE_IOU        mutual overlap that means "the same face" (two
#                               boxes sharing most of their area cannot be two
#                               different people standing side by side);
#   * FACE_DUPLICATE_MIN_IOU    a weaker overlap that still counts as "the same
#                               face" when the centres nearly coincide;
#   * FACE_DUPLICATE_CENTRE_RATIO  centres closer than this fraction of the
#                               smaller box's mean side are the same face;
#   * FACE_DUPLICATE_MAX_AREA_RATIO  the two boxes must be within this area
#                               ratio, so a legitimately SMALL face that merely
#                               sits near/inside a large one is never removed.
FACE_DUPLICATE_IOU = 0.5
FACE_DUPLICATE_MIN_IOU = 0.2
FACE_DUPLICATE_CENTRE_RATIO = 0.5
FACE_DUPLICATE_MAX_AREA_RATIO = 2.0

STATUS_KEEP = "KEEP"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_FILTERED = "FILTERED"


@dataclass
class DetectedFace:
    """A single detected face with its relevance metrics and (optional) embedding.

    ``embedding`` is ``None`` until :func:`embed_faces` runs; filtered faces are
    never embedded. Size fields are filled by :func:`filter_faces_by_size`,
    ``blur_score`` by :func:`filter_faces_by_quality`, and ``pitch``/``yaw``/
    ``roll`` by :func:`filter_faces_by_pose`.
    """

    bbox: list[float]
    det_score: float
    embedding: np.ndarray | None = field(default=None, repr=False)
    kps: np.ndarray | None = field(default=None, repr=False)
    width: float = 0.0
    height: float = 0.0
    area: float = 0.0
    relative_area: float = 0.0
    image_area: float = 0.0
    largest_face_area: float = 0.0
    ratio_to_largest_face: float = 0.0
    blur_score: float = 0.0
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    status: str = STATUS_KEEP
    reason: str = ""


# Lazily-created singleton so repeated calls do not reload the model.
_APP: FaceAnalysis | None = None
_APP_KEY: tuple | None = None


def normalize_det_size(det_size) -> tuple[int, int]:
    """Normalise a detector size (``None``, int edge, or ``(w, h)``) to a tuple.

    ``None`` means "use the default" so existing callers keep the 640 behaviour.
    """
    if det_size is None:
        det_size = DEFAULT_DET_SIZE
    if isinstance(det_size, int):
        return (det_size, det_size)
    values = tuple(int(v) for v in det_size)
    if len(values) == 1:
        return (values[0], values[0])
    return (values[0], values[1])


def load_model(
    name: str = MODEL_NAME,
    root: str = MODEL_ROOT,
    providers: list[str] | None = None,
    ctx_id: int = -1,
    det_thresh: float = DETECTION_THRESHOLD,
    det_size: int | tuple[int, int] | None = None,
) -> FaceAnalysis:
    """Load the pretrained model pack and return a ready ``FaceAnalysis``.

    On first call the model files are downloaded into ``root`` (~/.insightface
    by default), which is intentionally outside the repository. ``det_size``
    overrides the detector input resolution (``None`` = :data:`DEFAULT_DET_SIZE`).
    """
    providers = providers or DEFAULT_PROVIDERS
    app = FaceAnalysis(name=name, root=root, providers=providers)
    app.prepare(
        ctx_id=ctx_id,
        det_thresh=det_thresh,
        det_size=normalize_det_size(det_size),
    )
    return app


def get_model(**kwargs) -> FaceAnalysis:
    """Return a cached ``FaceAnalysis`` instance, loading it on first use.

    The cache key includes provider **and detector configuration** (det_size,
    det_thresh), so requesting a different ``det_size`` can never reuse an
    incompatible cached model instance.
    """
    global _APP, _APP_KEY
    providers = kwargs.get("providers") or DEFAULT_PROVIDERS
    det_size = normalize_det_size(kwargs.get("det_size"))
    det_thresh = kwargs.get("det_thresh", DETECTION_THRESHOLD)
    kwargs["det_size"] = det_size
    key = (
        kwargs.get("name", MODEL_NAME),
        kwargs.get("root", MODEL_ROOT),
        tuple(providers),
        det_size,
        det_thresh,
    )
    if _APP is None or _APP_KEY != key:
        _APP = load_model(**kwargs)
        _APP_KEY = key
    return _APP


def read_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk as a BGR ``numpy`` array (cv2 convention)."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


# ---------------------------------------------------------------------------
# Stage 1 — detection (no embeddings yet)
# ---------------------------------------------------------------------------


def detect_faces(image: np.ndarray, app: FaceAnalysis | None = None):
    """Detect faces in a BGR image; returns the raw InsightFace ``Face`` list."""
    app = app or get_model()
    return app.get(image)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes (``0``..``1``)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in box_b[:4])
    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return (intersection / union) if union > 0 else 0.0


def bbox_centre_distance(box_a: list[float], box_b: list[float]) -> float:
    """Euclidean distance (px) between the centres of two boxes."""
    acx, acy = (float(box_a[0]) + float(box_a[2])) / 2, (float(box_a[1]) + float(box_a[3])) / 2
    bcx, bcy = (float(box_b[0]) + float(box_b[2])) / 2, (float(box_b[1]) + float(box_b[3])) / 2
    return float(math.hypot(acx - bcx, acy - bcy))


def bbox_mean_side(box: list[float]) -> float:
    """Mean of a box's width and height (a resolution-independent size proxy)."""
    return (max(0.0, float(box[2]) - float(box[0]))
            + max(0.0, float(box[3]) - float(box[1]))) / 2


def duplicate_box_reason(
    box_a: list[float],
    box_b: list[float],
    iou_threshold: float = FACE_DUPLICATE_IOU,
    min_iou: float = FACE_DUPLICATE_MIN_IOU,
    centre_ratio: float = FACE_DUPLICATE_CENTRE_RATIO,
    max_area_ratio: float = FACE_DUPLICATE_MAX_AREA_RATIO,
) -> str | None:
    """Return why two boxes are the same physical face, or ``None``.

    Deliberately conservative, so two *different* nearby people are never merged:

    * ``"overlap"``        — a strong mutual overlap (IoU >= ``iou_threshold``)
                              between two boxes of comparable size. Two people
                              standing side by side cannot share most of their
                              box area, so this is one face detected twice.
    * ``"centre_distance"``— a weaker overlap plus almost the same centre
                              (distance <= ``centre_ratio`` of the smaller box's
                              mean side), again at a comparable size.

    The ``max_area_ratio`` guard means a legitimately smaller face (a distant or
    partially visible person) is never swallowed by a much larger neighbouring
    box, and the ``min_iou`` guard means two separate faces that merely sit near
    each other are never merged.
    """
    area_a = (max(0.0, float(box_a[2]) - float(box_a[0]))
              * max(0.0, float(box_a[3]) - float(box_a[1])))
    area_b = (max(0.0, float(box_b[2]) - float(box_b[0]))
              * max(0.0, float(box_b[3]) - float(box_b[1])))
    smaller, larger = min(area_a, area_b), max(area_a, area_b)
    if smaller <= 0.0 or larger <= 0.0:
        return None
    if (larger / smaller) > max_area_ratio:
        return None
    iou = bbox_iou(box_a, box_b)
    if iou >= iou_threshold:
        return "overlap"
    if iou < min_iou:
        return None
    distance = bbox_centre_distance(box_a, box_b)
    smaller_side = min(bbox_mean_side(box_a), bbox_mean_side(box_b))
    if distance <= centre_ratio * smaller_side:
        return "centre_distance"
    return None


def deduplicate_faces(
    faces: list[DetectedFace],
    iou_threshold: float = FACE_DUPLICATE_IOU,
    min_iou: float = FACE_DUPLICATE_MIN_IOU,
    centre_ratio: float = FACE_DUPLICATE_CENTRE_RATIO,
    max_area_ratio: float = FACE_DUPLICATE_MAX_AREA_RATIO,
) -> tuple[list[DetectedFace], list[dict]]:
    """Collapse boxes that describe the same physical face; keep one per face.

    The highest-confidence box of each duplicate group is kept and the survivors
    stay in the detector's own order, so on an image with no duplicates the
    output is exactly the input (and every face index is unchanged).

    Returns ``(kept_faces, removed)`` where each ``removed`` entry describes the
    dropped box, the box it duplicated and why — used by the reports/tests.
    """
    kept_indices: list[int] = []
    removed: list[dict] = []
    # Highest detection confidence wins (ties: earliest detection).
    for index in sorted(range(len(faces)), key=lambda i: (-faces[i].det_score, i)):
        face = faces[index]
        duplicate_of: int | None = None
        reason: str | None = None
        for kept_index in kept_indices:
            reason = duplicate_box_reason(
                face.bbox,
                faces[kept_index].bbox,
                iou_threshold=iou_threshold,
                min_iou=min_iou,
                centre_ratio=centre_ratio,
                max_area_ratio=max_area_ratio,
            )
            if reason is not None:
                duplicate_of = kept_index
                break
        if duplicate_of is None:
            kept_indices.append(index)
            continue
        reference = faces[duplicate_of]
        removed.append(
            {
                "bbox": [float(v) for v in face.bbox],
                "det_score": float(face.det_score),
                "duplicate_of_bbox": [float(v) for v in reference.bbox],
                "duplicate_of_score": float(reference.det_score),
                "iou": round(bbox_iou(face.bbox, reference.bbox), 6),
                "centre_distance": round(
                    bbox_centre_distance(face.bbox, reference.bbox), 3
                ),
                "reason": reason,
            }
        )
    kept = [faces[i] for i in sorted(kept_indices)]
    return kept, removed


def detect_face_regions(
    image: np.ndarray,
    app: FaceAnalysis | None = None,
    duplicates_out: list[dict] | None = None,
) -> list[DetectedFace]:
    """Detect faces only, *without* running the recognition model.

    Returns one :class:`DetectedFace` per physical face (bounding box, detection
    confidence and 5-point landmarks), after :func:`deduplicate_faces` has
    collapsed any boxes the detector returned twice for one face. Embeddings are
    intentionally left unset so filtering can drop irrelevant faces before the
    recognition model runs.

    When ``duplicates_out`` is given, it is extended with one entry per
    suppressed duplicate box (see :func:`deduplicate_faces`).
    """
    app = app or get_model()
    bboxes, kpss = app.det_model.detect(image, max_num=0, metric="default")
    faces: list[DetectedFace] = []
    for i in range(bboxes.shape[0]):
        kps = kpss[i] if kpss is not None else None
        faces.append(
            DetectedFace(
                bbox=[float(v) for v in bboxes[i, 0:4]],
                det_score=float(bboxes[i, 4]),
                kps=np.asarray(kps, dtype=np.float32) if kps is not None else None,
            )
        )
    faces, removed = deduplicate_faces(faces)
    if duplicates_out is not None:
        duplicates_out.extend(removed)
    return faces


# ---------------------------------------------------------------------------
# Stage 2 — face-size measurement and filtering
# ---------------------------------------------------------------------------


def face_size_metrics(
    bbox: list[float], image_shape: tuple[int, ...]
) -> tuple[float, float, float, float, float]:
    """Return ``(width, height, area, relative_area, image_area)`` for a bbox.

    ``relative_area`` is the face area divided by the whole image area, so it is
    independent of the image resolution.
    """
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    image_area = float(image_width * image_height)
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    relative_area = (area / image_area) if image_area > 0 else 0.0
    return width, height, area, relative_area, image_area


def filter_faces_by_size(
    faces: list[DetectedFace],
    image_shape: tuple[int, ...],
    min_relative_area: float = MIN_RELATIVE_FACE_AREA,
) -> tuple[list[DetectedFace], list[DetectedFace]]:
    """Annotate each face with size metrics and split into kept / filtered.

    A face is **kept** when its relative area is at least ``min_relative_area``,
    otherwise it is marked **FILTERED** (an obvious tiny / background candidate).

    Returns ``(kept, filtered)``; both lists together are exactly ``faces``.
    """
    kept: list[DetectedFace] = []
    filtered: list[DetectedFace] = []
    for face in faces:
        width, height, area, relative_area, image_area = face_size_metrics(
            face.bbox, image_shape
        )
        face.width = width
        face.height = height
        face.area = area
        face.relative_area = relative_area
        face.image_area = image_area
        if relative_area >= min_relative_area:
            face.status = STATUS_KEEP
            kept.append(face)
        else:
            face.status = STATUS_FILTERED
            face.reason = (
                f"size: relative area {relative_area:.5f} < {min_relative_area:.5f}"
            )
            filtered.append(face)
    return kept, filtered


# ---------------------------------------------------------------------------
# Stage 3 — face-quality filtering (new)
# ---------------------------------------------------------------------------


def compute_face_blur(
    image: np.ndarray, bbox: list[float], crop_size: int = BLUR_CROP_SIZE
) -> float:
    """Laplacian-variance blur score of a face crop (higher = sharper).

    The crop is taken from the bounding box (clipped to the image) and resized
    to ``crop_size`` before measuring, so the score does not simply track the
    crop's pixel dimensions.
    """
    height, width = image.shape[:2]
    x1 = int(max(0, min(width, round(bbox[0]))))
    y1 = int(max(0, min(height, round(bbox[1]))))
    x2 = int(max(0, min(width, round(bbox[2]))))
    y2 = int(max(0, min(height, round(bbox[3]))))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (crop_size, crop_size))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_face_quality(
    image: np.ndarray, faces: list[DetectedFace], blur_crop_size: int = BLUR_CROP_SIZE
) -> None:
    """Annotate ``blur_score`` on every face (in place).

    Cheap, so it runs for all detections to keep the report informative; the
    decision itself is applied afterwards by :func:`filter_faces_by_quality`.
    """
    for face in faces:
        face.blur_score = compute_face_blur(image, face.bbox, blur_crop_size)


def filter_faces_by_quality(
    faces: list[DetectedFace],
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
    min_blur_score: float = MIN_BLUR_SCORE,
) -> tuple[list[DetectedFace], list[DetectedFace]]:
    """Drop low-quality faces using the confidence and blur scores already set.

    A face is rejected when its detection confidence is below
    ``min_detection_confidence`` or its blur score is below ``min_blur_score``.
    Call :func:`measure_face_quality` first to populate ``blur_score``.
    Returns ``(passed, rejected)``.
    """
    passed: list[DetectedFace] = []
    rejected: list[DetectedFace] = []
    for face in faces:
        if face.det_score < min_detection_confidence:
            face.status = STATUS_FILTERED
            face.reason = (
                f"quality: detection confidence {face.det_score:.2f} "
                f"< {min_detection_confidence:.2f}"
            )
            rejected.append(face)
        elif face.blur_score < min_blur_score:
            face.status = STATUS_FILTERED
            face.reason = (
                f"quality: blur score {face.blur_score:.1f} < {min_blur_score:.1f}"
            )
            rejected.append(face)
        else:
            passed.append(face)
    return passed, rejected


# ---------------------------------------------------------------------------
# Stage 4 — head-orientation check (new, deliberately weak)
# ---------------------------------------------------------------------------


def compute_head_pose(
    image: np.ndarray, face: DetectedFace, app: FaceAnalysis | None = None
) -> tuple[float, float, float] | None:
    """Estimate ``(pitch, yaw, roll)`` in degrees, or ``None`` if unavailable.

    Uses the ``landmark_3d_68`` model that ships with ``buffalo_l``; its
    ``get()`` sets ``face['pose'] = [pitch, yaw, roll]``. The 5-point landmarks
    alone do not provide pitch, so this model is required for a real pose.
    Returns ``None`` when the model is not loaded.
    """
    app = app or get_model()
    model = app.models.get("landmark_3d_68")
    if model is None:
        return None
    raw = Face(bbox=face.bbox, kps=face.kps, det_score=face.det_score)
    model.get(image, raw)
    pose = raw.get("pose")  # Face subclasses dict; None when unset
    if pose is None:
        return None
    pitch, yaw, roll = (float(v) for v in np.asarray(pose, dtype=np.float32).ravel()[:3])
    return pitch, yaw, roll


def measure_head_pose(
    image: np.ndarray, faces: list[DetectedFace], app: FaceAnalysis | None = None
) -> None:
    """Annotate ``pitch``/``yaw``/``roll`` on every face (in place).

    Pose is computed for all detections so the report is complete; it is a
    lightweight landmark model, unlike the recognition (embedding) model, which
    still only runs on kept faces.
    """
    app = app or get_model()
    for face in faces:
        try:
            pose = compute_head_pose(image, face, app=app)
        except Exception:  # degenerate crop / alignment failure: skip pose only
            pose = None
        if pose is not None:
            face.pitch, face.yaw, face.roll = pose


def filter_faces_by_pose(
    faces: list[DetectedFace],
    max_yaw: float = MAX_YAW,
    max_pitch: float = MAX_PITCH,
    strong_relative_area: float = STRONG_RELATIVE_FACE_AREA,
) -> tuple[list[DetectedFace], list[DetectedFace], list[DetectedFace]]:
    """Apply the (weak) head-orientation guard.

    A face is only rejected for pose when its yaw or pitch exceeds the generous
    ``max_yaw`` / ``max_pitch`` limits **and** it is not a clearly large face.
    A large face with an extreme angle is marked **UNCERTAIN** instead of being
    dropped, so legitimate people are not lost.

    Call :func:`measure_head_pose` first to populate the angles.
    Returns ``(kept, uncertain, filtered)``.
    """
    kept: list[DetectedFace] = []
    uncertain: list[DetectedFace] = []
    filtered: list[DetectedFace] = []
    for face in faces:
        pose = None if face.yaw is None else (face.pitch, face.yaw, face.roll)

        extreme = pose is not None and (
            abs(face.yaw) > max_yaw or abs(face.pitch) > max_pitch
        )
        if not extreme:
            face.status = STATUS_KEEP
            kept.append(face)
        elif face.relative_area >= strong_relative_area:
            face.status = STATUS_UNCERTAIN
            face.reason = (
                f"pose: extreme angle (yaw {face.yaw:.0f}, pitch {face.pitch:.0f}) "
                "on a large face"
            )
            uncertain.append(face)
        else:
            face.status = STATUS_FILTERED
            face.reason = (
                f"pose: extreme angle (yaw {face.yaw:.0f}, pitch {face.pitch:.0f}) "
                "on a non-large face"
            )
            filtered.append(face)
    return kept, uncertain, filtered


# ---------------------------------------------------------------------------
# Stage 3b — perspective / relative-scale check (new, ADDITIONAL signal)
# ---------------------------------------------------------------------------


def measure_relative_scale(faces: list[DetectedFace]) -> None:
    """Annotate ``largest_face_area`` and ``ratio_to_largest_face`` (in place).

    ``largest_face_area`` is the maximum face area among the detected faces of
    the same image; ``ratio_to_largest_face`` is this face's area divided by it.
    Requires :func:`filter_faces_by_size` to have set ``area`` on every face.
    A single-face image therefore gets ``ratio = 1.0`` and is never penalised.
    """
    largest = max((face.area for face in faces), default=0.0)
    for face in faces:
        face.largest_face_area = largest
        face.ratio_to_largest_face = (face.area / largest) if largest > 0 else 0.0


def filter_faces_by_relative_scale(
    faces: list[DetectedFace],
    min_ratio_to_largest: float = MIN_RATIO_TO_LARGEST_FACE,
    scale_keep_blur: float = SCALE_KEEP_BLUR,
    scale_keep_confidence: float = SCALE_KEEP_CONFIDENCE,
    scale_max_yaw: float = SCALE_MAX_YAW,
    scale_max_pitch: float = SCALE_MAX_PITCH,
) -> tuple[list[DetectedFace], list[DetectedFace], list[DetectedFace]]:
    """Additional perspective-aware gate combining several existing signals.

    Call :func:`measure_relative_scale` first. A face at/above
    ``min_ratio_to_largest`` of the dominant face passes untouched. A face below
    it (clearly much smaller than the dominant face) is only **filtered** when it
    also lacks quality/pose evidence; if it is sharp, confident and reasonably
    front-facing it becomes **UNCERTAIN** (kept + flagged) rather than removed.

    Returns ``(kept, uncertain, filtered)``.
    """
    kept: list[DetectedFace] = []
    uncertain: list[DetectedFace] = []
    filtered: list[DetectedFace] = []
    for face in faces:
        ratio = face.ratio_to_largest_face
        if ratio >= min_ratio_to_largest:
            kept.append(face)
            continue
        high_quality = (
            face.blur_score >= scale_keep_blur
            and face.det_score >= scale_keep_confidence
        )
        extreme_pose = (
            (face.yaw is not None and abs(face.yaw) > scale_max_yaw)
            or (face.pitch is not None and abs(face.pitch) > scale_max_pitch)
        )
        if high_quality and not extreme_pose:
            face.status = STATUS_UNCERTAIN
            face.reason = (
                f"scale: only {ratio:.2f} of the largest face but high quality "
                f"(blur {face.blur_score:.0f}, conf {face.det_score:.2f}) — kept as uncertain"
            )
            uncertain.append(face)
        else:
            face.status = STATUS_FILTERED
            face.reason = (
                f"scale: only {ratio:.2f} of the largest face (blur "
                f"{face.blur_score:.0f}, conf {face.det_score:.2f}"
                + (
                    f", yaw {face.yaw:.0f}"
                    if face.yaw is not None
                    else ""
                )
                + ")"
            )
            filtered.append(face)
    return kept, uncertain, filtered


# ---------------------------------------------------------------------------
# Combined relevance decision
# ---------------------------------------------------------------------------


def evaluate_faces(
    image: np.ndarray,
    faces: list[DetectedFace],
    app: FaceAnalysis | None = None,
    min_relative_area: float = MIN_RELATIVE_FACE_AREA,
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
    min_blur_score: float = MIN_BLUR_SCORE,
    max_yaw: float = MAX_YAW,
    max_pitch: float = MAX_PITCH,
    strong_relative_area: float = STRONG_RELATIVE_FACE_AREA,
    blur_crop_size: int = BLUR_CROP_SIZE,
    min_ratio_to_largest: float = MIN_RATIO_TO_LARGEST_FACE,
    scale_keep_blur: float = SCALE_KEEP_BLUR,
    scale_keep_confidence: float = SCALE_KEEP_CONFIDENCE,
    scale_max_yaw: float = SCALE_MAX_YAW,
    scale_max_pitch: float = SCALE_MAX_PITCH,
) -> tuple[list[DetectedFace], list[DetectedFace], list[DetectedFace]]:
    """Run the full relevance pipeline: size -> quality -> pose -> scale.

    Returns ``(kept, uncertain, filtered)``. ``kept`` and ``uncertain`` are the
    faces that should be embedded (``uncertain`` optionally, see
    ``EMBED_UNCERTAIN``); ``filtered`` are the rejected background candidates.
    """
    app = app or get_model()
    # Measure quality and pose for every detection (cheap) so the report is
    # complete; the gates below decide which faces are actually kept.
    measure_face_quality(image, faces, blur_crop_size)
    measure_head_pose(image, faces, app=app)

    size_kept, size_filtered = filter_faces_by_size(faces, image.shape, min_relative_area)
    # The perspective signal is measured against the LARGEST detected face in the
    # image, so it must be computed over all detections (not just the size-kept).
    measure_relative_scale(faces)
    quality_kept, quality_filtered = filter_faces_by_quality(
        size_kept, min_detection_confidence, min_blur_score
    )
    kept, uncertain, pose_filtered = filter_faces_by_pose(
        quality_kept, max_yaw=max_yaw, max_pitch=max_pitch,
        strong_relative_area=strong_relative_area,
    )
    kept, scale_uncertain, scale_filtered = filter_faces_by_relative_scale(
        kept,
        min_ratio_to_largest=min_ratio_to_largest,
        scale_keep_blur=scale_keep_blur,
        scale_keep_confidence=scale_keep_confidence,
        scale_max_yaw=scale_max_yaw,
        scale_max_pitch=scale_max_pitch,
    )
    uncertain = uncertain + scale_uncertain
    return kept, uncertain, size_filtered + quality_filtered + pose_filtered + scale_filtered


# ---------------------------------------------------------------------------
# Stage 5 — embeddings (kept faces only)
# ---------------------------------------------------------------------------


def embed_faces(image: np.ndarray, faces: list[DetectedFace], app: FaceAnalysis | None = None) -> None:
    """Compute recognition embeddings for the given faces, in place.

    Only the faces passed in are embedded, so callers should pass the faces that
    survived filtering.
    """
    app = app or get_model()
    recognition = app.models.get("recognition")
    if recognition is None:
        raise RuntimeError(
            "Recognition model is not loaded; cannot generate embeddings."
        )
    for face in faces:
        raw = Face(bbox=face.bbox, kps=face.kps, det_score=face.det_score)
        recognition.get(image, raw)
        embedding = getattr(raw, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(raw, "embedding", None)
        face.embedding = np.asarray(embedding, dtype=np.float32)


def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Cosine similarity between two embeddings (``-1``..``1``, higher = closer)."""
    a = np.asarray(embedding1, dtype=np.float32).ravel()
    b = np.asarray(embedding2, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def same_person(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    threshold: float = FACE_SIMILARITY_THRESHOLD,
) -> bool:
    """Decide whether two embeddings belong to the same person."""
    return compare_embeddings(embedding1, embedding2) >= threshold


# ---------------------------------------------------------------------------
# Convenience: the full per-image pipeline
# ---------------------------------------------------------------------------


@dataclass
class ImageAnalysis:
    """Outcome of running detection -> filtering -> embedding on one image."""

    image_path: Path
    image_shape: tuple[int, ...]
    all_faces: list[DetectedFace]
    kept: list[DetectedFace]
    uncertain: list[DetectedFace]
    filtered: list[DetectedFace]
    thresholds: dict
    detection_seconds: float = 0.0
    filter_seconds: float = 0.0
    embedding_seconds: float = 0.0
    # Duplicate boxes suppressed at detection time (one entry per removed box).
    duplicates_removed: list[dict] = field(default_factory=list)
    # Raw detector output before duplicate suppression (``all_faces`` is after).
    raw_detections: int = 0

    @property
    def total_faces(self) -> int:
        return len(self.all_faces)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates_removed)

    @property
    def embedded(self) -> list[DetectedFace]:
        return [f for f in self.all_faces if f.embedding is not None]

    @property
    def total_seconds(self) -> float:
        return self.detection_seconds + self.filter_seconds + self.embedding_seconds


def analyze_image(
    image: np.ndarray,
    image_path: str | Path = "",
    min_relative_area: float = MIN_RELATIVE_FACE_AREA,
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
    min_blur_score: float = MIN_BLUR_SCORE,
    max_yaw: float = MAX_YAW,
    max_pitch: float = MAX_PITCH,
    strong_relative_area: float = STRONG_RELATIVE_FACE_AREA,
    min_ratio_to_largest: float = MIN_RATIO_TO_LARGEST_FACE,
    scale_keep_blur: float = SCALE_KEEP_BLUR,
    scale_keep_confidence: float = SCALE_KEEP_CONFIDENCE,
    scale_max_yaw: float = SCALE_MAX_YAW,
    scale_max_pitch: float = SCALE_MAX_PITCH,
    embed_uncertain: bool = EMBED_UNCERTAIN,
    app: FaceAnalysis | None = None,
) -> ImageAnalysis:
    """Run the full relevance pipeline on one already-loaded image."""
    app = app or get_model()

    t_det = time.perf_counter()
    duplicates: list[dict] = []
    detected = detect_face_regions(image, app=app, duplicates_out=duplicates)
    detection_seconds = time.perf_counter() - t_det

    t_filt = time.perf_counter()
    kept, uncertain, filtered = evaluate_faces(
        image,
        detected,
        app=app,
        min_relative_area=min_relative_area,
        min_detection_confidence=min_detection_confidence,
        min_blur_score=min_blur_score,
        max_yaw=max_yaw,
        max_pitch=max_pitch,
        strong_relative_area=strong_relative_area,
        min_ratio_to_largest=min_ratio_to_largest,
        scale_keep_blur=scale_keep_blur,
        scale_keep_confidence=scale_keep_confidence,
        scale_max_yaw=scale_max_yaw,
        scale_max_pitch=scale_max_pitch,
    )
    filter_seconds = time.perf_counter() - t_filt

    to_embed = kept + (uncertain if embed_uncertain else [])
    t_emb = time.perf_counter()
    embed_faces(image, to_embed, app=app)
    embedding_seconds = time.perf_counter() - t_emb

    return ImageAnalysis(
        image_path=Path(image_path),
        image_shape=image.shape,
        all_faces=detected,
        kept=kept,
        uncertain=uncertain,
        filtered=filtered,
        thresholds={
            "min_relative_area": min_relative_area,
            "min_detection_confidence": min_detection_confidence,
            "min_blur_score": min_blur_score,
            "max_yaw": max_yaw,
            "max_pitch": max_pitch,
            "strong_relative_area": strong_relative_area,
            "min_ratio_to_largest": min_ratio_to_largest,
            "scale_keep_blur": scale_keep_blur,
            "scale_keep_confidence": scale_keep_confidence,
            "scale_max_yaw": scale_max_yaw,
            "scale_max_pitch": scale_max_pitch,
        },
        detection_seconds=detection_seconds,
        filter_seconds=filter_seconds,
        embedding_seconds=embedding_seconds,
        duplicates_removed=duplicates,
        raw_detections=len(detected) + len(duplicates),
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

DEBUG_VALUES = 5


def _fmt_bbox(bbox: list[float]) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in bbox) + "]"


def _fmt_angle(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f} deg"


def _describe_face(index: int, face: DetectedFace) -> None:
    print(f"\n  Face {index}")
    print(f"    Bounding box: {_fmt_bbox(face.bbox)}")
    print(f"    Width: {face.width:.1f}  Height: {face.height:.1f} px")
    print(f"    Relative area: {face.relative_area:.5f} "
          f"({face.relative_area * 100:.3f}%)")
    print(f"    Ratio to largest face: {face.ratio_to_largest_face:.3f}"
          f"  (largest face area: {face.largest_face_area:.1f} px)")
    print(f"    Detection confidence: {face.det_score:.3f}")
    print(f"    Blur score: {face.blur_score:.1f}")
    print(f"    Yaw: {_fmt_angle(face.yaw)}   Pitch: {_fmt_angle(face.pitch)}   "
          f"Roll: {_fmt_angle(face.roll)}")
    print(f"    Decision: {face.status}")
    if face.reason:
        print(f"    Reason: {face.reason}")
    if face.embedding is not None:
        print(f"    Embedding shape: {face.embedding.shape}")
        head = ", ".join(f"{v:.4f}" for v in face.embedding[:DEBUG_VALUES])
        print(f"    Embedding preview (first {DEBUG_VALUES}): [{head}, ...]")
    else:
        print("    Embedding: skipped (not embedded)")


def _similarity_report(
    faces: list[DetectedFace],
    labels: list[str],
    threshold: float,
) -> None:
    """Print pairwise cosine similarities between the given (kept) faces."""
    if len(faces) < 2:
        return
    print("\nPairwise cosine similarity (threshold "
          f"{threshold:.2f} => same person):")
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            score = compare_embeddings(faces[i].embedding, faces[j].embedding)
            verdict = "SAME" if score >= threshold else "different"
            print(f"  {labels[i]} <-> {labels[j]}: {score:+.4f}  [{verdict}]")


def _report_analysis(analysis: ImageAnalysis, threshold: float) -> None:
    name = analysis.image_path.name
    height, width = int(analysis.image_shape[0]), int(analysis.image_shape[1])
    print("\n" + "=" * 60)
    print(f"Image: {name}")
    print(f"Image dimensions: {width} x {height}  "
          f"(image area: {width * height:,} px)")
    print(f"Original faces detected: {analysis.total_faces}")
    if analysis.duplicates_removed:
        print(f"Duplicate boxes suppressed: {analysis.duplicate_count} "
              f"(raw detector boxes: {analysis.raw_detections})")
        for duplicate in analysis.duplicates_removed:
            print(f"    dropped {_fmt_bbox(duplicate['bbox'])} "
                  f"(conf {duplicate['det_score']:.3f}) — {duplicate['reason']} "
                  f"with {_fmt_bbox(duplicate['duplicate_of_bbox'])}, "
                  f"IoU {duplicate['iou']:.3f}, centre distance "
                  f"{duplicate['centre_distance']:.1f} px")

    for i, face in enumerate(analysis.all_faces, start=1):
        _describe_face(i, face)

    embedded = analysis.embedded
    print(f"\n  Final relevant faces (KEEP): {len(analysis.kept)}")
    print(f"  Uncertain faces: {len(analysis.uncertain)}")
    print(f"  Filtered faces: {len(analysis.filtered)}")
    print(f"  Embeddings generated: {len(embedded)}")
    if embedded:
        print(f"  Embedding shape: {embedded[0].embedding.shape}")
    print(f"  Detection time: {analysis.detection_seconds * 1000:.0f} ms")
    print(f"  Filtering time: {analysis.filter_seconds * 1000:.0f} ms")
    print(f"  Embedding time: {analysis.embedding_seconds * 1000:.0f} ms")
    print(f"  Processing time: {analysis.total_seconds * 1000:.0f} ms")

    # Similarity between kept faces within this image.
    _similarity_report(
        analysis.kept,
        [f"{name}#{i + 1}" for i in range(len(analysis.kept))],
        threshold,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Path(s) to test image(s)")
    parser.add_argument(
        "--provider",
        default=None,
        help="ONNX Runtime provider, e.g. CPUExecutionProvider or "
             "CoreMLExecutionProvider (default: CoreML if available, else CPU).",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=DEFAULT_DET_SIZE,
        help="Detector input resolution (edge length). 640 matches the InsightFace "
             "default; larger values (e.g. 1280) detect smaller faces at higher "
             "cost (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FACE_SIMILARITY_THRESHOLD,
        help="Cosine-similarity threshold for 'same person' (default: %(default)s).",
    )
    parser.add_argument(
        "--min-relative-area",
        type=float,
        default=MIN_RELATIVE_FACE_AREA,
        help="Minimum face area as a fraction of the image (default: %(default)s).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=MIN_DETECTION_CONFIDENCE,
        help="Minimum detection confidence (default: %(default)s).",
    )
    parser.add_argument(
        "--min-blur-score",
        type=float,
        default=MIN_BLUR_SCORE,
        help="Minimum face-crop blur score (Laplacian variance) (default: %(default)s).",
    )
    parser.add_argument(
        "--max-yaw",
        type=float,
        default=MAX_YAW,
        help="Maximum absolute yaw in degrees before a face is considered "
             "strongly side-facing (default: %(default)s).",
    )
    parser.add_argument(
        "--max-pitch",
        type=float,
        default=MAX_PITCH,
        help="Maximum absolute pitch in degrees (default: %(default)s).",
    )
    parser.add_argument(
        "--min-ratio-to-largest-face",
        type=float,
        default=MIN_RATIO_TO_LARGEST_FACE,
        help="Minimum face area as a fraction of the largest detected face before "
             "the perspective/background gate applies (default: %(default)s).",
    )
    parser.add_argument(
        "--scale-keep-blur",
        type=float,
        default=SCALE_KEEP_BLUR,
        help="Blur score at/above which a much-smaller face is kept as uncertain "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--scale-keep-confidence",
        type=float,
        default=SCALE_KEEP_CONFIDENCE,
        help="Detection confidence at/above which a much-smaller face is kept as "
             "uncertain (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    providers = resolve_providers(args.provider) if args.provider else None

    t0 = time.perf_counter()
    app = get_model(providers=providers, det_size=args.det_size)
    print("Model loaded successfully.")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Providers: {providers or DEFAULT_PROVIDERS}")
    print(f"  Detection size: {args.det_size}x{args.det_size}")
    print(f"  Model load time: {time.perf_counter() - t0:.2f}s")
    print("  Filter thresholds [EXPERIMENTAL - tune on real event photographs]:")
    print(f"    MIN_RELATIVE_FACE_AREA    = {args.min_relative_area:.5f} "
          f"({args.min_relative_area * 100:.3f}% of image)")
    print(f"    MIN_DETECTION_CONFIDENCE  = {args.min_detection_confidence:.2f}")
    print(f"    MIN_BLUR_SCORE            = {args.min_blur_score:.1f}")
    print(f"    MAX_YAW                   = {args.max_yaw:.1f} deg")
    print(f"    MAX_PITCH                 = {args.max_pitch:.1f} deg")
    print(f"    MIN_RATIO_TO_LARGEST_FACE = {args.min_ratio_to_largest_face:.2f} "
          f"(perspective gate; SCALE_KEEP_BLUR={args.scale_keep_blur:.0f}, "
          f"SCALE_KEEP_CONFIDENCE={args.scale_keep_confidence:.2f})")

    # Collect the first kept face of each image for cross-image comparison.
    first_faces: list[DetectedFace] = []
    first_labels: list[str] = []

    for image_path in args.images:
        image = read_image(image_path)
        analysis = analyze_image(
            image,
            image_path=image_path,
            min_relative_area=args.min_relative_area,
            min_detection_confidence=args.min_detection_confidence,
            min_blur_score=args.min_blur_score,
            max_yaw=args.max_yaw,
            max_pitch=args.max_pitch,
            min_ratio_to_largest=args.min_ratio_to_largest_face,
            scale_keep_blur=args.scale_keep_blur,
            scale_keep_confidence=args.scale_keep_confidence,
            app=app,
        )
        _report_analysis(analysis, args.threshold)

        if analysis.kept:
            first_faces.append(analysis.kept[0])
            first_labels.append(f"{analysis.image_path.name}#1")

    if len(first_faces) >= 2:
        print("\n" + "=" * 60)
        print("Cross-image comparison (primary kept face of each image):")
        _similarity_report(first_faces, first_labels, args.threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
