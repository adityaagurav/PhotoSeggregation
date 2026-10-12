"""End-to-end tests for the final clustering -> records -> Merge Sort stage.

Uses synthetic faces (no photos / model needed) to verify the data structures,
in particular that one photo can belong to multiple people.

Run with::

    .venv/bin/python -m unittest tests.test_pipeline -v
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from app.ai.face_cluster import cluster_faces, select_usable_faces
from app.algorithms.merge_sort import merge_sort, person_id_sort_key
from app.processing.person_manager import PersonManager
from app.processing.record_generator import (
    build_photo_index,
    build_photo_people_index,
    generate_records,
)


def unit(angle_degrees: float) -> np.ndarray:
    radians = math.radians(angle_degrees)
    return np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)


@dataclass
class FakeFace:
    image_filename: str
    face_index: int
    embedding: np.ndarray
    status: str = "KEEP"
    final_status: str = "KEEP"

    @property
    def photo(self) -> str:
        return self.image_filename


def run_stage(faces: list[FakeFace], threshold: float = 0.60):
    usable = select_usable_faces(faces)
    clustering = cluster_faces(usable, threshold)
    persons = PersonManager().assign(clustering.clusters)
    records = generate_records(persons, clustering.assignments)
    sorted_records = merge_sort(records)
    return persons, sorted_records, clustering


class PipelineTests(unittest.TestCase):
    def test_records_sorted_by_person_id(self) -> None:
        # Two photos; photo1 has person X and person Y (orthogonal), photo2 has X.
        faces = [
            FakeFace("photo1.jpg", 0, unit(0)),    # X
            FakeFace("photo1.jpg", 1, unit(90)),   # Y (different person, same photo)
            FakeFace("photo2.jpg", 0, unit(0)),    # X again
        ]
        _persons, sorted_records, _clustering = run_stage(faces)
        keys = [person_id_sort_key(r.person_id) for r in sorted_records]
        self.assertEqual(keys, sorted(keys))

    def test_multi_person_photo_is_supported(self) -> None:
        faces = [
            FakeFace("event05.jpeg", 0, unit(0)),    # person A
            FakeFace("event05.jpeg", 1, unit(90)),   # person B (same photo)
            FakeFace("event01.jpeg", 0, unit(0)),    # person A again
        ]
        persons, sorted_records, _clustering = run_stage(faces)
        photo_people = build_photo_people_index(sorted_records)

        # The first photo must list more than one person.
        self.assertIn("event05.jpeg", photo_people)
        self.assertEqual(len(photo_people["event05.jpeg"]), 2)
        self.assertEqual(sorted(photo_people["event05.jpeg"]), ["Person #1", "Person #2"])

        photo_index = build_photo_index(sorted_records)
        # Person #1 appears in both photos (in face-processing order, which the
        # stable Merge Sort preserves within a person); Person #2 only in event05.
        self.assertEqual(photo_index["Person #1"], ["event05.jpeg", "event01.jpeg"])
        self.assertEqual(photo_index["Person #2"], ["event05.jpeg"])

    def test_records_keep_face_and_similarity(self) -> None:
        faces = [
            FakeFace("a.jpg", 0, unit(0)),
            FakeFace("b.jpg", 0, unit(0)),
        ]
        _persons, sorted_records, _clustering = run_stage(faces)
        self.assertTrue(all(r.face_id >= 0 for r in sorted_records))
        # The second face joined an existing cluster -> it has a similarity value.
        similarities = [r.similarity for r in sorted_records]
        self.assertTrue(any(s is not None for s in similarities))

    def test_uncertain_faces_do_not_create_persons(self) -> None:
        faces = [
            FakeFace("a.jpg", 0, unit(0), status="KEEP", final_status="KEEP"),
            FakeFace("b.jpg", 0, unit(5), status="UNCERTAIN", final_status="UNCERTAIN"),
        ]
        persons, sorted_records, _clustering = run_stage(faces)
        self.assertEqual(len(persons), 1)
        self.assertEqual(len(sorted_records), 1)

    def test_photo_appears_once_per_person_not_exclusively(self) -> None:
        faces = [
            FakeFace("shared.jpg", 0, unit(0)),
            FakeFace("shared.jpg", 1, unit(90)),
            FakeFace("shared.jpg", 2, unit(180)),
        ]
        _persons, sorted_records, _clustering = run_stage(faces)
        photo_people = build_photo_people_index(sorted_records)
        self.assertEqual(len(photo_people["shared.jpg"]), 3)
        self.assertEqual(len(sorted_records), 3)


if __name__ == "__main__":
    unittest.main()
