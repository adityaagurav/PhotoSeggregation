"""Tests for duplicate-detection suppression in the detection stage.

The bug being guarded here: the detector could return two bounding boxes for one
physical face (an offset / re-scaled response of the same face that survives the
model's own fixed-IoU NMS). Every box was carried through as its own "face", so
one person could become two people. ``predict.detect_face_regions`` now removes
such boxes before any filtering, embedding or clustering happens.

The model is never loaded here: a fake detector / recognition model feeds
synthetic bounding boxes, exactly like the other unit tests use synthetic faces.

Run with::

    .venv/bin/python -m unittest tests.test_face_deduplication -v
"""

from __future__ import annotations

import unittest

import numpy as np

from app.ai.face_cluster import cluster_faces, select_usable_faces
from app.algorithms.merge_sort import merge_sort, person_id_sort_key
from app.processing.person_manager import PersonManager
from app.processing.record_generator import generate_records
from ml.inference import predict as P


# ---------------------------------------------------------------------------
# Fakes (no model loading)
# ---------------------------------------------------------------------------


class FakeDetModel:
    """Stand-in for the InsightFace detector; returns canned boxes."""

    def __init__(self, boxes: list[tuple[float, float, float, float, float]]) -> None:
        self.boxes = boxes

    def detect(self, image, max_num=0, metric="default"):
        if not self.boxes:
            return np.empty((0, 5), dtype=np.float32), None
        return np.asarray(self.boxes, dtype=np.float32), None


class FakeRecognition:
    """Minimal recognition model: gives every face the same embedding."""

    DIM = 8

    def get(self, image, face) -> None:
        face.embedding = np.ones(self.DIM, dtype=np.float32)


class FakeApp:
    """Only what ``detect_face_regions`` / ``analyze_image`` touch."""

    def __init__(self, boxes) -> None:
        self.det_model = FakeDetModel(boxes)
        self.models = {"recognition": FakeRecognition(), "landmark_3d_68": None}


def detection(bbox, score=0.9) -> P.DetectedFace:
    return P.DetectedFace(bbox=[float(v) for v in bbox], det_score=float(score))


def noiseless_image(height: int = 400, width: int = 600) -> np.ndarray:
    """A textured BGR image so face crops have a high blur score (KEEP)."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Two highly overlapping boxes around the same face are deduplicated
# ---------------------------------------------------------------------------


class OverlappingBoxTests(unittest.TestCase):
    def test_identical_boxes_collapse_to_one(self) -> None:
        faces = [detection([100, 100, 300, 300], 0.91), detection([100, 100, 300, 300], 0.74)]
        kept, removed = P.deduplicate_faces(faces)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["reason"], "overlap")
        self.assertAlmostEqual(removed[0]["iou"], 1.0, places=6)

    def test_the_highest_confidence_box_is_the_one_kept(self) -> None:
        low = detection([100, 100, 300, 300], 0.55)
        high = detection([104, 103, 302, 298], 0.93)
        kept, removed = P.deduplicate_faces([low, high])
        self.assertEqual(kept, [high])
        self.assertEqual(removed[0]["duplicate_of_score"], 0.93)

    def test_removed_entry_carries_bounding_box_details(self) -> None:
        faces = [detection([100, 100, 300, 300], 0.9), detection([110, 108, 308, 296], 0.6)]
        _kept, removed = P.deduplicate_faces(faces)
        entry = removed[0]
        self.assertEqual(entry["bbox"], [110.0, 108.0, 308.0, 296.0])
        self.assertEqual(entry["duplicate_of_bbox"], [100.0, 100.0, 300.0, 300.0])
        self.assertGreater(entry["iou"], P.FACE_DUPLICATE_IOU)
        self.assertGreater(entry["centre_distance"], 0.0)

    def test_near_identical_centre_of_a_rescaled_response_is_deduplicated(self) -> None:
        # Overlap below FACE_DUPLICATE_IOU, but the centres nearly coincide and
        # the sizes are comparable -> the same face detected at another scale.
        faces = [detection([0, 0, 200, 200], 0.9), detection([50, 50, 250, 250], 0.8)]
        iou = P.bbox_iou(faces[0].bbox, faces[1].bbox)
        self.assertLess(iou, P.FACE_DUPLICATE_IOU)
        self.assertGreaterEqual(iou, P.FACE_DUPLICATE_MIN_IOU)
        kept, removed = P.deduplicate_faces(faces)
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed[0]["reason"], "centre_distance")

    def test_clean_detection_list_is_returned_unchanged_in_order(self) -> None:
        faces = [
            detection([10, 10, 110, 110], 0.95),
            detection([300, 10, 400, 110], 0.92),
            detection([10, 200, 110, 300], 0.90),
        ]
        kept, removed = P.deduplicate_faces(faces)
        self.assertEqual(kept, faces)  # same objects, same order -> same face slots
        self.assertEqual(removed, [])


# ---------------------------------------------------------------------------
# 2. Two nearby but distinct faces are not incorrectly merged
# ---------------------------------------------------------------------------


class DistinctFaceTests(unittest.TestCase):
    def test_side_by_side_faces_are_not_merged(self) -> None:
        left = detection([100, 100, 300, 300], 0.9)
        right = detection([330, 100, 530, 300], 0.88)
        kept, removed = P.deduplicate_faces([left, right])
        self.assertEqual(kept, [left, right])
        self.assertEqual(removed, [])

    def test_partially_overlapping_neighbouring_faces_are_not_merged(self) -> None:
        # Two people standing close: boxes overlap ~18% and the centres are
        # ~0.45 box-sides apart -> not the same face, so both must survive.
        a = detection([0, 0, 200, 200], 0.9)
        b = detection([90, 90, 290, 290], 0.85)
        iou = P.bbox_iou(a.bbox, b.bbox)
        self.assertLess(iou, P.FACE_DUPLICATE_MIN_IOU)
        kept, removed = P.deduplicate_faces([a, b])
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, [])

    def test_face_suppression_never_merges_a_small_face_into_a_large_one(self) -> None:
        # A distant face whose box sits inside a much larger face's box must not
        # be swallowed: the area ratio guard rejects it as a "duplicate".
        large = detection([0, 0, 400, 400], 0.9)
        small = detection([20, 20, 120, 120], 0.8)  # area ratio 13x
        kept, removed = P.deduplicate_faces([large, small])
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, [])

    def test_almost_identical_boxes_are_one_region_not_two_people(self) -> None:
        # Two boxes describing the same pixels cannot be two people, so exactly
        # one survives even when the scores differ a lot.
        a = detection([10, 10, 200, 200], 0.9)
        b = detection([10.5, 10.5, 200.5, 200.5], 0.7)
        kept, removed = P.deduplicate_faces([a, b])
        self.assertEqual(kept, [a])
        self.assertEqual(removed[0]["det_score"], 0.7)


# ---------------------------------------------------------------------------
# 3. A legitimate small face is not discarded just to lower the count
# ---------------------------------------------------------------------------


class SmallFacePreservationTests(unittest.TestCase):
    def test_isolated_small_face_is_kept(self) -> None:
        big = detection([0, 0, 500, 500], 0.95)
        small = detection([900, 100, 960, 180], 0.71)   # distant person
        kept, removed = P.deduplicate_faces([big, small])
        self.assertIn(small, kept)
        self.assertEqual(removed, [])

    def test_small_face_touching_a_large_box_is_kept(self) -> None:
        big = detection([0, 0, 400, 400], 0.95)
        small = detection([380, 380, 440, 450], 0.68)
        kept, _removed = P.deduplicate_faces([big, small])
        self.assertEqual(len(kept), 2)

    def test_dedup_only_ever_removes_real_duplicates(self) -> None:
        faces = [
            detection([0, 0, 200, 200], 0.9),
            detection([5, 5, 205, 205], 0.6),     # duplicate of the first
            detection([500, 0, 520, 25], 0.7),    # tiny but distinct
            detection([600, 300, 800, 500], 0.85),
        ]
        kept, removed = P.deduplicate_faces(faces)
        self.assertEqual(len(kept), 3)
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(kept) + len(removed), len(faces))


# ---------------------------------------------------------------------------
# 4. Duplicate processing does not count a face twice
# ---------------------------------------------------------------------------


class DuplicateProcessingTests(unittest.TestCase):
    # 6 real faces, plus 2 boxes that are the same face returned a second time.
    BOXES_6_PLUS_2_DUPLICATES = [
        (10, 10, 110, 110, 0.95),
        (130, 10, 230, 110, 0.93),
        (250, 10, 350, 110, 0.91),
        (10, 150, 110, 250, 0.90),
        (130, 150, 230, 250, 0.89),
        (250, 150, 350, 250, 0.88),
        (13, 12, 113, 112, 0.72),    # duplicate of face 0
        (253, 152, 353, 252, 0.70),  # duplicate of face 5
    ]

    def test_eight_boxes_for_six_faces_count_as_six(self) -> None:
        app = FakeApp(self.BOXES_6_PLUS_2_DUPLICATES)
        removed: list[dict] = []
        faces = P.detect_face_regions(noiseless_image(), app=app, duplicates_out=removed)
        self.assertEqual(len(faces), 6)
        self.assertEqual(len(removed), 2)
        self.assertEqual({entry["reason"] for entry in removed}, {"overlap"})

    def test_deduplication_is_idempotent(self) -> None:
        app = FakeApp(self.BOXES_6_PLUS_2_DUPLICATES)
        once = P.detect_face_regions(noiseless_image(), app=app)
        twice = P.detect_face_regions(noiseless_image(), app=app)
        self.assertEqual(len(once), 6)
        self.assertEqual([f.bbox for f in once], [f.bbox for f in twice])
        again, removed_again = P.deduplicate_faces(list(once))
        self.assertEqual(again, once)
        self.assertEqual(removed_again, [])

    def test_processing_the_same_photo_twice_does_not_accumulate_faces(self) -> None:
        app = FakeApp(self.BOXES_6_PLUS_2_DUPLICATES)
        image = noiseless_image()
        first = P.analyze_image(image, image_path="event.jpg", app=app)
        second = P.analyze_image(image, image_path="event.jpg", app=app)
        self.assertEqual(first.total_faces, 6)
        self.assertEqual(second.total_faces, 6)
        self.assertEqual(first.raw_detections, 8)
        self.assertEqual(first.duplicate_count, 2)
        self.assertEqual(len(first.kept), len(second.kept))

    def test_duplicates_removed_are_reported_for_scoring(self) -> None:
        app = FakeApp(self.BOXES_6_PLUS_2_DUPLICATES)
        analysis = P.analyze_image(noiseless_image(), image_path="event.jpg", app=app)
        self.assertEqual(len(analysis.duplicates_removed), 2)
        for entry in analysis.duplicates_removed:
            self.assertIn("bbox", entry)
            self.assertIn("duplicate_of_bbox", entry)
            self.assertGreater(entry["iou"], 0.0)


# ---------------------------------------------------------------------------
# 5. Person-Photo records stay correct and Merge Sort still passes
# ---------------------------------------------------------------------------


class OneHotFace:
    """Minimal face for the clustering/records stage (one photo, distinct people)."""

    def __init__(self, index: int, dim: int = 6) -> None:
        self.image_filename = "event.jpg"
        self.face_index = index
        self.embedding = np.eye(dim, dtype=np.float32)[index]
        self.status = "KEEP"
        self.final_status = "KEEP"


class RecordsAndMergeSortTests(unittest.TestCase):
    def test_one_record_per_physical_face_after_deduplication(self) -> None:
        app = FakeApp(DuplicateProcessingTests.BOXES_6_PLUS_2_DUPLICATES)
        faces = P.detect_face_regions(noiseless_image(), app=app)
        self.assertEqual(len(faces), 6)

        # One distinct person per surviving face, all in the same photo.
        synthetic = [OneHotFace(i, dim=len(faces)) for i in range(len(faces))]
        clustering = cluster_faces(select_usable_faces(synthetic), threshold=0.60)
        self.assertEqual(len(clustering.clusters), len(faces))
        persons = PersonManager().assign(clustering.clusters)
        records = merge_sort(generate_records(persons, clustering.assignments))

        self.assertEqual(len(records), 6)
        keys = [person_id_sort_key(r.person_id) for r in records]
        self.assertEqual(keys, sorted(keys))  # Merge Sort PASS

    def test_merge_sort_still_passes_on_an_undeduplicated_count(self) -> None:
        # Control: without dedup the same eight boxes would be eight records /
        # eight persons, which is exactly the reported symptom.
        synthetic = [OneHotFace(i, dim=8) for i in range(8)]
        clustering = cluster_faces(select_usable_faces(synthetic), threshold=0.60)
        persons = PersonManager().assign(clustering.clusters)
        records = merge_sort(generate_records(persons, clustering.assignments))
        self.assertEqual(len(records), 8)
        keys = [person_id_sort_key(r.person_id) for r in records]
        self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
