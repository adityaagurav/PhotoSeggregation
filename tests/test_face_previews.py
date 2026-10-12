"""Unit tests for the display-only per-person face preview generator.

These use synthetic images and lightweight stand-ins for the pipeline's
``FaceRecord`` / ``PersonIdentity`` objects, so no model or real photos are
needed. The original image must never be modified.

Run with::

    .venv/bin/python -m unittest tests.test_face_previews -v
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.processing import face_previews as fp


# ---------------------------------------------------------------------------
# Test doubles mirroring the real pipeline objects
# ---------------------------------------------------------------------------


@dataclass
class FakeFace:
    image_filename: str
    face_index: int
    bbox: list[float]
    det_score: float = 0.9
    blur_score: float = 100.0
    yaw: float | None = None
    pitch: float | None = None


@dataclass
class FakeCluster:
    members: list = field(default_factory=list)


@dataclass
class FakePerson:
    person_id: str
    cluster: FakeCluster


def make_person(person_id: str, *faces: FakeFace) -> FakePerson:
    return FakePerson(person_id, FakeCluster(list(faces)))


def make_image(path: Path, size=(400, 300), box=None) -> None:
    image = Image.new("RGB", size, (245, 245, 245))
    if box is not None:
        ImageDraw.Draw(image).rectangle(box, fill=(30, 60, 120))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=90)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------


class NamingTests(unittest.TestCase):
    def test_person_preview_filename_is_safe_and_stable(self) -> None:
        self.assertEqual(fp.person_preview_filename("Person #1"), "Person_1.jpg")
        self.assertEqual(fp.person_preview_filename("Person #12"), "Person_12.jpg")
        self.assertEqual(fp.person_preview_filename("../../etc/passwd"), "etc_passwd.jpg")


class RepresentativeFaceTests(unittest.TestCase):
    def test_picks_large_frontal_face(self) -> None:
        small = FakeFace("p.jpg", 0, [10, 10, 30, 30], det_score=0.95, yaw=0.0)
        large = FakeFace("p.jpg", 1, [10, 10, 150, 150], det_score=0.9, yaw=0.0)
        angled = FakeFace("q.jpg", 2, [10, 10, 150, 150], det_score=0.99, yaw=85.0)
        chosen = fp.select_representative_face([small, large, angled])
        self.assertIs(chosen, large)

    def test_prefers_frontal_over_angled_when_similar_size(self) -> None:
        frontal = FakeFace("p.jpg", 0, [10, 10, 100, 100], yaw=2.0, pitch=-3.0)
        turned = FakeFace("p.jpg", 1, [10, 10, 105, 105], yaw=80.0, pitch=30.0)
        self.assertIs(fp.select_representative_face([turned, frontal]), frontal)

    def test_returns_none_without_usable_faces(self) -> None:
        self.assertIsNone(fp.select_representative_face([]))
        self.assertIsNone(fp.select_representative_face([FakeFace("p.jpg", 0, [])]))
        self.assertIsNone(
            fp.select_representative_face([FakeFace("p.jpg", 0, [5, 5, 5, 5])])
        )


class CropBoxTests(unittest.TestCase):
    def test_padding_is_symmetric_and_square(self) -> None:
        box = fp.square_crop_box([100, 100, 200, 200], 1000, 1000)
        self.assertEqual(box, (60, 60, 240, 240))
        left, top, right, bottom = box
        self.assertEqual(right - left, bottom - top)

    def test_box_is_slid_inside_the_image_not_clipped(self) -> None:
        # Face at the top-left corner: the square must slide in, not shrink.
        box = fp.square_crop_box([0, 0, 100, 100], 1000, 1000)
        self.assertEqual(box, (0, 0, 180, 180))

    def test_box_is_clamped_to_small_images(self) -> None:
        box = fp.square_crop_box([0, 0, 100, 100], 60, 60)
        assert box is not None
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[2], 60)
        self.assertLessEqual(box[3], 60)

    def test_degenerate_inputs_return_none(self) -> None:
        self.assertIsNone(fp.square_crop_box([1, 2], 100, 100))
        self.assertIsNone(fp.square_crop_box([0, 0, 10, 10], 0, 100))


class GeneratePreviewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.upload_dir = self.root / "uploads"
        self.output_dir = self.root / "output"
        make_image(self.upload_dir / "a.jpg", box=(140, 90, 260, 230))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_writes_square_crop_and_manifest(self) -> None:
        person = make_person(
            "Person #1",
            FakeFace("a.jpg", 0, [140, 90, 260, 230], yaw=1.0, pitch=-2.0),
        )
        manifest = fp.generate_person_previews(
            [person], self.upload_dir, self.output_dir
        )

        self.assertIn("Person #1", manifest)
        entry = manifest["Person #1"]
        self.assertEqual(entry["file"], "Person_1.jpg")
        self.assertEqual(entry["photo"], "a.jpg")
        self.assertEqual(entry["face_id"], 0)

        crop_path = self.output_dir / "previews" / entry["file"]
        self.assertTrue(crop_path.is_file())
        with Image.open(crop_path) as crop:
            self.assertEqual(crop.size, (fp.PREVIEW_SIZE, fp.PREVIEW_SIZE))
            self.assertEqual(crop.format, "JPEG")

        on_disk = fp.read_face_previews(self.output_dir)
        self.assertEqual(on_disk, manifest)

    def test_original_photo_is_byte_identical(self) -> None:
        original = self.upload_dir / "a.jpg"
        before = sha256(original)
        fp.generate_person_previews(
            [make_person("Person #1", FakeFace("a.jpg", 0, [140, 90, 260, 230]))],
            self.upload_dir,
            self.output_dir,
        )
        self.assertEqual(sha256(original), before)

    def test_crop_is_centered_on_the_face(self) -> None:
        # A dark face rectangle on a light image: the crop's center must be dark.
        person = make_person("Person #1", FakeFace("a.jpg", 0, [140, 90, 260, 230]))
        manifest = fp.generate_person_previews(
            [person], self.upload_dir, self.output_dir
        )
        crop_path = self.output_dir / "previews" / manifest["Person #1"]["file"]
        with Image.open(crop_path) as crop:
            r, g, b = crop.convert("RGB").getpixel((crop.width // 2, crop.height // 2))
        self.assertLess(r + g + b, 400)  # dark face fill, not the light background

    def test_exif_rotated_source_uses_the_pipeline_coordinate_space(self) -> None:
        # cv2 (the pipeline's decoder) applies the EXIF orientation while Pillow
        # does not. The crop must land on the face as cv2 located it, so the
        # stored bounding boxes and the crop share one coordinate space.
        raw = Image.new("RGB", (400, 200), (245, 245, 245))
        ImageDraw.Draw(raw).rectangle((300, 50, 380, 130), fill=(20, 50, 110))
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees
        raw.save(
            self.upload_dir / "rot.jpg", format="JPEG", quality=95, exif=exif
        )

        decoded = cv2.imread(str(self.upload_dir / "rot.jpg"))
        self.assertIsNotNone(decoded)
        dark = np.argwhere(decoded.sum(axis=2) < 260)
        ys, xs = dark[:, 0], dark[:, 1]
        bbox = [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ]

        manifest = fp.generate_person_previews(
            [make_person("Person #1", FakeFace("rot.jpg", 0, bbox))],
            self.upload_dir,
            self.output_dir,
        )
        crop_path = self.output_dir / "previews" / manifest["Person #1"]["file"]
        with Image.open(crop_path) as crop:
            centre = crop.convert("RGB").getpixel((crop.width // 2, crop.height // 2))
        self.assertLess(sum(centre), 400)

    def test_missing_source_photo_is_skipped_gracefully(self) -> None:
        person = make_person("Person #1", FakeFace("ghost.jpg", 0, [0, 0, 50, 50]))
        manifest = fp.generate_person_previews(
            [person], self.upload_dir, self.output_dir
        )
        self.assertEqual(manifest, {})
        # The manifest file is still written so the API can report "no previews".
        self.assertTrue((self.output_dir / fp.MANIFEST_FILENAME).is_file())

    def test_person_without_usable_face_gets_no_preview(self) -> None:
        people = [
            make_person("Person #1", FakeFace("a.jpg", 0, [140, 90, 260, 230])),
            make_person("Person #2"),  # no members
        ]
        manifest = fp.generate_person_previews(
            people, self.upload_dir, self.output_dir
        )
        self.assertIn("Person #1", manifest)
        self.assertNotIn("Person #2", manifest)

    def test_shared_photo_is_decoded_once_but_cropped_per_person(self) -> None:
        people = [
            make_person("Person #1", FakeFace("a.jpg", 0, [140, 90, 260, 230])),
            make_person("Person #2", FakeFace("a.jpg", 1, [150, 100, 250, 220])),
        ]
        manifest = fp.generate_person_previews(
            people, self.upload_dir, self.output_dir
        )
        self.assertEqual(sorted(manifest), ["Person #1", "Person #2"])
        for person_id in ("Person #1", "Person #2"):
            self.assertTrue(
                (self.output_dir / "previews" / manifest[person_id]["file"]).is_file()
            )

    def test_read_manifest_returns_empty_when_missing_or_invalid(self) -> None:
        self.assertEqual(fp.read_face_previews(self.output_dir), {})
        bad = self.output_dir / fp.MANIFEST_FILENAME
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{not json")
        self.assertEqual(fp.read_face_previews(self.output_dir), {})
        bad.write_text("[]")
        self.assertEqual(fp.read_face_previews(self.output_dir), {})
        bad.write_text(json.dumps({"Person #1": {"file": "Person_1.jpg"}}))
        self.assertEqual(
            fp.read_face_previews(self.output_dir),
            {"Person #1": {"file": "Person_1.jpg"}},
        )


if __name__ == "__main__":
    unittest.main()
