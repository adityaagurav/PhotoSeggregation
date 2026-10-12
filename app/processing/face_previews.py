"""Representative per-person face preview crops for the Results UI.

This is a DISPLAY-ONLY post-processing step. It runs on the face data the
EXISTING pipeline already produced: every face carries its bounding box on the
in-memory ``FaceRecord`` object, and a ``PersonIdentity.cluster.members`` exposes
exactly the faces that belong to that person. Nothing about detection, filtering,
embeddings, matching, clustering, Person IDs, uncertain recovery, Person-Photo
records or the Merge Sort is touched, and **detection is never re-run**.

The original uploaded photos are opened READ-ONLY and are never replaced,
resized, cropped in place, annotated or recompressed; only a small new square
JPEG crop is written into the job's output directory::

    data/output/<job_id>/previews/<Person_slug>.jpg
    data/output/<job_id>/face_previews.json      (manifest: person -> crop)

Coordinate space
----------------
The source image is decoded with ``cv2.imread`` — the SAME call the pipeline uses
(``ml.inference.predict.read_image``). This matters: OpenCV applies the EXIF
orientation to JPEGs while Pillow does not, so a Pillow crop of a rotated photo
would land on the wrong pixels. Decoding identically guarantees the stored
bounding boxes and the crop always share one coordinate space.

The manifest lets the API report preview availability per person, so a person
with no usable crop simply falls back to the folder icon in the UI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2

PREVIEW_DIRNAME = "previews"
MANIFEST_FILENAME = "face_previews.json"

# Padding added around the detected face box, as a fraction of the box's longer
# side, so the crop does not cut off the top of the head. A display choice only.
PADDING_RATIO = 0.4

# Display thumbnail size (square, px) and JPEG quality.
PREVIEW_SIZE = 200
JPEG_QUALITY = 85

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


# ---------------------------------------------------------------------------
# Representative-face selection
# ---------------------------------------------------------------------------


def person_preview_filename(person_id: Any) -> str:
    """Deterministic, filesystem-safe crop filename for a Person ID.

    ``"Person #1"`` -> ``"Person_1.jpg"`` (the manifest keeps the real ID).
    """
    slug = _SLUG_RE.sub("_", str(person_id)).strip("_") or "person"
    return f"{slug}.jpg"


def _bbox(record: Any) -> list[float]:
    bbox = getattr(record, "bbox", None) or []
    if len(bbox) < 4:
        return []
    return [float(value) for value in bbox[:4]]


def face_pixel_area(record: Any) -> float:
    """Detected face area in pixels (a good proxy for crop detail)."""
    bbox = _bbox(record)
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def face_frontality(record: Any) -> float:
    """0..1 factor; 1.0 is a straight-on face. Unmeasured pose counts as frontal."""
    yaw = getattr(record, "yaw", None)
    pitch = getattr(record, "pitch", None)
    yaw = abs(float(yaw)) if yaw is not None else 0.0
    pitch = abs(float(pitch)) if pitch is not None else 0.0
    return max(0.0, 1.0 - yaw / 90.0) * max(0.0, 1.0 - pitch / 90.0)


def representative_face_score(record: Any) -> tuple[float, float, float, float, int]:
    """Ranking key for picking a person's representative face.

    Prefers a large, front-facing, confident, sharp face. Everything is derived
    from data the existing pipeline already measured — nothing is hardcoded.
    """
    pixels = face_pixel_area(record)
    det_score = float(getattr(record, "det_score", 0.0) or 0.0)
    blur_score = float(getattr(record, "blur_score", 0.0) or 0.0)
    face_index = int(getattr(record, "face_index", 0) or 0)
    return (pixels * face_frontality(record), pixels, det_score, blur_score, -face_index)


def select_representative_face(members: Iterable[Any]) -> Any | None:
    """Return the best face to preview for a person, or ``None`` if unusable."""
    usable = [member for member in members if face_pixel_area(member) > 0]
    if not usable:
        return None
    return max(usable, key=representative_face_score)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def square_crop_box(
    bbox: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
    padding_ratio: float = PADDING_RATIO,
) -> tuple[int, int, int, int] | None:
    """A square, padded, in-bounds pixel box centered on the face.

    The box is grown by ``padding_ratio`` on every side (so the head is not cut
    off) and then shifted — never shrunk more than necessary — to stay inside the
    image, so no invented pixels are added. Returns ``None`` for a degenerate box.
    """
    if len(bbox) < 4 or image_width <= 0 or image_height <= 0:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    side = max(width, height)
    half = side / 2.0 + padding_ratio * side

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left, top = center_x - half, center_y - half
    right, bottom = center_x + half, center_y + half

    # Slide the square back inside the image before clamping, so it stays as
    # large as possible (a face near an edge is not clipped to a thin sliver).
    if left < 0:
        right -= left
        left = 0.0
    if top < 0:
        bottom -= top
        top = 0.0
    if right > image_width:
        left -= right - image_width
        right = float(image_width)
    if bottom > image_height:
        top -= bottom - image_height
        bottom = float(image_height)

    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(image_width), right)
    bottom = min(float(image_height), bottom)
    if right - left < 2 or bottom - top < 2:
        return None
    return (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))


def _resize_square(image: Any, size: int) -> Any:
    """Resize a BGR crop to ``size`` x ``size`` (area filter when downscaling)."""
    height, width = image.shape[:2]
    interpolation = (
        cv2.INTER_AREA if height >= size and width >= size else cv2.INTER_CUBIC
    )
    return cv2.resize(image, (size, size), interpolation=interpolation)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    path = output_dir / MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_face_previews(output_dir: str | Path) -> dict[str, Any]:
    """Load the preview manifest written for a job (``{}`` when absent/invalid)."""
    path = Path(output_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def generate_person_previews(
    persons: Iterable[Any],
    upload_dir: str | Path,
    output_dir: str | Path,
    *,
    padding_ratio: float = PADDING_RATIO,
    size: int = PREVIEW_SIZE,
) -> dict[str, Any]:
    """Write one representative face crop per person; return the manifest.

    Never raises for a single unreadable image or unusable face: that person
    simply has no preview and the UI falls back to the folder icon. Each distinct
    original photo is decoded once, even when several people share it.
    """
    upload_dir = Path(upload_dir)
    output_dir = Path(output_dir)
    preview_dir = output_dir / PREVIEW_DIRNAME
    manifest: dict[str, Any] = {}

    # Pick the representative face for every person up front.
    chosen: list[tuple[str, Any]] = []
    for person in persons:
        person_id = getattr(person, "person_id", None)
        cluster = getattr(person, "cluster", None)
        members = list(getattr(cluster, "members", []) or [])
        record = select_representative_face(members)
        if person_id is None or record is None:
            continue
        chosen.append((str(person_id), record))

    preview_dir.mkdir(parents=True, exist_ok=True)
    if not chosen:
        _write_manifest(output_dir, manifest)
        return manifest

    by_photo: dict[str, list[tuple[str, Any]]] = {}
    for person_id, record in chosen:
        photo_name = Path(str(getattr(record, "image_filename", ""))).name
        if photo_name:
            by_photo.setdefault(photo_name, []).append((person_id, record))

    for photo_name, items in by_photo.items():
        source = upload_dir / photo_name
        if not source.is_file():
            continue
        # Decode exactly like the pipeline (ml.inference.predict.read_image) so the
        # stored bounding boxes and the crop share one coordinate space.
        image = cv2.imread(str(source))
        if image is None:
            continue
        height, width = image.shape[:2]
        for person_id, record in items:
            box = square_crop_box(_bbox(record), width, height, padding_ratio)
            if box is None:
                continue
            left, top, right, bottom = box
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue
            filename = person_preview_filename(person_id)
            saved = cv2.imwrite(
                str(preview_dir / filename),
                _resize_square(crop, size),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if not saved:
                continue
            manifest[person_id] = {
                "file": filename,
                "photo": photo_name,
                "face_id": getattr(record, "face_index", None),
                "bbox": _bbox(record),
                "width": size,
                "height": size,
            }

    _write_manifest(output_dir, manifest)
    return manifest
