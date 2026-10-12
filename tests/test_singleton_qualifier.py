"""Unit tests for strict singleton-person qualification.

Run with::

    .venv/bin/python -m unittest tests.test_singleton_qualifier -v
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from app.ai.face_cluster import PersonCluster
from app.ai.singleton_qualifier import (
    REASON_LOW_CONFIDENCE,
    REASON_MULTIPLE,
    REASON_TOO_SMALL,
    REASON_YAW,
    STATUS_ACCEPTED_SINGLETON,
    STATUS_SINGLETON_REJECTED,
    DEFAULT_SINGLETON_THRESHOLDS,
    SingletonThresholds,
    cluster_photos,
    evaluate_singleton_face,
    is_singleton_cluster,
    qualify_singletons,
)


@dataclass
class FakeFace:
    image_filename: str
    face_index: int
    relative_area: float = 0.05
    det_score: float = 0.90
    blur_score: float = 200.0
    yaw: float | None = 5.0
    pitch: float | None = 5.0
    status: str = "KEEP"
    final_status: str = "KEEP"
    embedding: np.ndarray = None  # unused by the qualifier

    @property
    def photo(self) -> str:
        return self.image_filename


def cluster(*faces: FakeFace) -> PersonCluster:
    return PersonCluster(members=list(faces))


class ClusterShapeTests(unittest.TestCase):
    def test_singleton_detection(self) -> None:
        self.assertTrue(is_singleton_cluster(cluster(FakeFace("a.jpg", 0))))
        self.assertFalse(
            is_singleton_cluster(cluster(FakeFace("a.jpg", 0), FakeFace("b.jpg", 0)))
        )

    def test_cluster_photos_distinct(self) -> None:
        self.assertEqual(
            cluster_photos(cluster(FakeFace("a.jpg", 0), FakeFace("a.jpg", 1), FakeFace("b.jpg", 0))),
            ["a.jpg", "b.jpg"],
        )


class SingletonGateTests(unittest.TestCase):
    def test_good_face_is_accepted(self) -> None:
        decision = evaluate_singleton_face(FakeFace("a.jpg", 0))
        self.assertEqual(decision.status, STATUS_ACCEPTED_SINGLETON)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reasons, [])

    def test_small_face_rejected(self) -> None:
        decision = evaluate_singleton_face(FakeFace("a.jpg", 0, relative_area=0.001))
        self.assertEqual(decision.status, STATUS_SINGLETON_REJECTED)
        self.assertIn(REASON_TOO_SMALL, decision.reasons)

    def test_multiple_failures_recorded_with_aggregate(self) -> None:
        decision = evaluate_singleton_face(
            FakeFace("a.jpg", 0, relative_area=0.001, det_score=0.5, yaw=80.0)
        )
        self.assertEqual(decision.status, STATUS_SINGLETON_REJECTED)
        self.assertIn(REASON_TOO_SMALL, decision.reasons)
        self.assertIn(REASON_LOW_CONFIDENCE, decision.reasons)
        self.assertIn(REASON_YAW, decision.reasons)
        self.assertIn(REASON_MULTIPLE, decision.reasons)

    def test_unmeasured_pose_does_not_reject(self) -> None:
        decision = evaluate_singleton_face(FakeFace("a.jpg", 0, yaw=None, pitch=None))
        self.assertEqual(decision.status, STATUS_ACCEPTED_SINGLETON)

    def test_slightly_turned_face_is_accepted(self) -> None:
        # 40 deg yaw is within the 45 deg limit -> acceptable, not required frontal.
        decision = evaluate_singleton_face(FakeFace("a.jpg", 0, yaw=40.0, pitch=-20.0))
        self.assertEqual(decision.status, STATUS_ACCEPTED_SINGLETON)

    def test_thresholds_are_configurable(self) -> None:
        strict = SingletonThresholds(min_relative_face_area=0.5)
        decision = evaluate_singleton_face(FakeFace("a.jpg", 0, relative_area=0.05), strict)
        self.assertEqual(decision.status, STATUS_SINGLETON_REJECTED)


class QualificationFlowTests(unittest.TestCase):
    def test_multi_photo_cluster_is_never_gated(self) -> None:
        # A multi-photo cluster with an individually-weak face must be retained.
        weak = FakeFace("b.jpg", 0, relative_area=0.0001, det_score=0.4, yaw=88.0)
        good = FakeFace("a.jpg", 0)
        multi = cluster(good, weak)
        result = qualify_singletons([multi])
        self.assertEqual(result.examined, 0)
        self.assertEqual(result.retained_clusters, [multi])
        self.assertEqual(result.multi_photo_clusters, [multi])

    def test_singleton_rejected_is_not_retained(self) -> None:
        bad = cluster(FakeFace("a.jpg", 0, relative_area=0.0001))
        result = qualify_singletons([bad])
        self.assertEqual(result.examined, 1)
        self.assertEqual(result.retained_clusters, [])
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.rejected[0].status, STATUS_SINGLETON_REJECTED)

    def test_singleton_accepted_is_retained(self) -> None:
        good = cluster(FakeFace("a.jpg", 0))
        result = qualify_singletons([good])
        self.assertEqual(result.examined, 1)
        self.assertEqual(result.retained_clusters, [good])
        self.assertEqual(len(result.accepted), 1)

    def test_retained_order_is_preserved(self) -> None:
        multi = cluster(FakeFace("a.jpg", 0), FakeFace("b.jpg", 0))
        good_single = cluster(FakeFace("c.jpg", 0))
        bad_single = cluster(FakeFace("d.jpg", 0, relative_area=0.0001))
        result = qualify_singletons([multi, bad_single, good_single])
        self.assertEqual(result.retained_clusters, [multi, good_single])


if __name__ == "__main__":
    unittest.main()
