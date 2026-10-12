"""Unit tests for the representative/centroid final clustering.

Run with::

    .venv/bin/python -m unittest tests.test_face_clustering -v
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from app.ai.face_cluster import (
    PersonCluster,
    cluster_faces,
    select_usable_faces,
)


def unit(angle_degrees: float) -> np.ndarray:
    """Unit 2-D vector at ``angle_degrees``; cosine(a, b) = cos(angle_a - angle_b)."""
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


class CentroidUnitTests(unittest.TestCase):
    def test_representative_is_unit_normalised_mean(self) -> None:
        cluster = PersonCluster()
        cluster.members.extend(
            [FakeFace("a.jpg", 0, unit(0)), FakeFace("b.jpg", 0, unit(0))]
        )
        representative = cluster.representative()
        self.assertAlmostEqual(float(np.linalg.norm(representative)), 1.0, places=5)
        # Two identical unit vectors -> the mean is the same direction.
        self.assertAlmostEqual(float(representative[0]), 1.0, places=5)

    def test_representative_excludes_a_photo(self) -> None:
        cluster = PersonCluster()
        cluster.members.extend(
            [FakeFace("a.jpg", 0, unit(0)), FakeFace("b.jpg", 0, unit(90))]
        )
        only_a = cluster.representative(exclude_photo="b.jpg")
        self.assertAlmostEqual(float(only_a[0]), 1.0, places=5)
        # Excluding every member returns None (no evidence left).
        single = PersonCluster(members=[FakeFace("a.jpg", 0, unit(0))])
        self.assertIsNone(single.representative(exclude_photo="a.jpg"))


class SamePhotoRuleTests(unittest.TestCase):
    def test_same_photo_faces_never_merge(self) -> None:
        faces = [
            FakeFace("event05.jpeg", 0, unit(0)),
            FakeFace("event05.jpeg", 1, unit(0)),  # identical embedding, same photo
        ]
        result = cluster_faces(faces, threshold=0.60)
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(result.same_photo_comparisons, 0)

    def test_cross_photo_identical_faces_merge(self) -> None:
        faces = [
            FakeFace("event01.jpeg", 0, unit(0)),
            FakeFace("event02.jpeg", 0, unit(0)),
        ]
        result = cluster_faces(faces, threshold=0.60)
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.same_photo_comparisons, 0)

    def test_same_photo_comparisons_counter_stays_zero(self) -> None:
        faces = [
            FakeFace("p1.jpg", 0, unit(0)),
            FakeFace("p1.jpg", 1, unit(45)),
            FakeFace("p2.jpg", 0, unit(5)),
            FakeFace("p2.jpg", 1, unit(50)),
        ]
        result = cluster_faces(faces, threshold=0.60)
        self.assertEqual(result.same_photo_comparisons, 0)
        # Every assignment used cross-photo evidence (or opened a new cluster).
        self.assertTrue(result.comparisons >= 1)


class ChainingResistanceTests(unittest.TestCase):
    def test_centroid_does_not_chain_like_single_linkage(self) -> None:
        # A(0), B(30), C(75 degrees):
        #   A-B = cos(30)  = 0.866  (>= 0.65)
        #   B-C = cos(45)  = 0.707  (>= 0.65)
        #   A-C = cos(75)  = 0.259
        # Single linkage / connected components would merge A, B and C through B.
        # The centroid of {A, B} sits at 15 degrees, and cos(75 - 15) = 0.5 < 0.65,
        # so C must stay in its own cluster.
        a = FakeFace("p1.jpg", 0, unit(0))
        b = FakeFace("p2.jpg", 0, unit(30))
        c = FakeFace("p3.jpg", 0, unit(75))
        result = cluster_faces([a, b, c], threshold=0.65)

        self.assertEqual(len(result.clusters), 2)
        cluster_sizes = [cluster.size for cluster in result.clusters]
        self.assertIn(2, cluster_sizes)  # A and B together
        self.assertIn(1, cluster_sizes)  # C alone
        # C is not in the same cluster as A.
        cluster_of = {id(assignment.face): assignment.cluster for assignment in result.assignments}
        self.assertIsNot(cluster_of[id(a)], cluster_of[id(c)])

    def test_distinct_people_stay_separate(self) -> None:
        faces = [
            FakeFace("p1.jpg", 0, unit(0)),
            FakeFace("p2.jpg", 0, unit(90)),
            FakeFace("p3.jpg", 0, unit(180)),
        ]
        result = cluster_faces(faces, threshold=0.60)
        self.assertEqual(len(result.clusters), 3)


class UsableFaceSelectionTests(unittest.TestCase):
    def test_only_keep_and_confirmed_participate(self) -> None:
        faces = [
            FakeFace("p1.jpg", 0, unit(0), status="KEEP", final_status="KEEP"),
            FakeFace("p2.jpg", 0, unit(10), status="UNCERTAIN",
                     final_status="CONFIRMED_FROM_MATCH"),
            FakeFace("p3.jpg", 0, unit(20), status="UNCERTAIN", final_status="UNCERTAIN"),
        ]
        usable = select_usable_faces(faces)
        statuses = [face.final_status for face in usable]
        self.assertEqual(statuses, ["KEEP", "CONFIRMED_FROM_MATCH"])

    def test_unresolved_uncertain_does_not_form_a_person(self) -> None:
        faces = [
            FakeFace("p1.jpg", 0, unit(0)),
            FakeFace("p2.jpg", 0, unit(5), status="UNCERTAIN", final_status="UNCERTAIN"),
        ]
        usable = select_usable_faces(faces)
        self.assertEqual(len(usable), 1)
        result = cluster_faces(usable, threshold=0.60)
        self.assertEqual(len(result.clusters), 1)


if __name__ == "__main__":
    unittest.main()
