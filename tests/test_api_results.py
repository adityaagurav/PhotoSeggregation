"""Tests for the read-only results endpoint and the original-photo endpoint.

These exercise the real functions in :mod:`app.api` directly (no HTTP server and
no extra test dependency required): the results payload is built from the actual
``person_clusters.json`` / ``person_photo_records.csv`` files, and the photo
route's path-traversal / membership guards are checked against a temp job dir.

Run with::

    .venv/bin/python -m unittest tests.test_api_results -v
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from fastapi.responses import FileResponse

from app import api

VALID_JOB = "0123456789ab"


def write_clusters(path: Path, persons, unresolved=None, metadata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "persons": persons,
        "unresolved_uncertain": unresolved or [],
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload))


def write_records(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["person_id", "photo", "face_id", "similarity", "status"])
        for row in rows:
            writer.writerow(row)


def write_manifest(output_dir: Path, manifest: dict) -> None:
    (output_dir / api.PREVIEW_DIRNAME).mkdir(parents=True, exist_ok=True)
    (output_dir / "face_previews.json").write_text(json.dumps(manifest))


class BuildResultsTests(unittest.TestCase):
    """``_build_job_results`` must read the real on-disk pipeline output."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.upload_dir = self.root / "uploads" / VALID_JOB
        self.output_dir = self.root / "output" / VALID_JOB
        self.upload_dir.mkdir(parents=True)
        (self.upload_dir / "a.jpg").write_bytes(b"x")
        (self.upload_dir / "b.jpg").write_bytes(b"y")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_returns_none_without_output_files(self) -> None:
        self.assertIsNone(
            api._build_job_results(
                VALID_JOB, self.upload_dir, self.output_dir, summary=None
            )
        )

    def test_summary_none_path_derives_upload_count(self) -> None:
        # Regression: with no in-memory summary the function used to raise
        # UnboundLocalError on ``photos_uploaded``.
        write_clusters(
            self.output_dir / "person_clusters.json",
            [{"person_id": "Person #1", "photos": ["a.jpg"]}],
            unresolved=[{"photo": "a.jpg", "face_id": 0}],
            metadata={"photos": 2},
        )
        write_records(
            self.output_dir / "person_photo_records.csv",
            [["Person #1", "a.jpg", 0, "", "KEEP"]],
        )
        payload = api._build_job_results(
            VALID_JOB, self.upload_dir, self.output_dir, summary=None
        )
        assert payload is not None
        self.assertEqual(payload["photos_uploaded"], 2)
        self.assertEqual(payload["photos_processed"], 2)
        self.assertEqual(payload["people_identified"], 1)
        self.assertEqual(payload["unresolved_uncertain_faces"], 1)
        self.assertEqual(payload["person_photo_records"], 1)
        self.assertTrue(payload["merge_sort"]["ordered"])
        self.assertTrue(payload["merge_sort"]["verified_from_records"])
        # Faces detected/retained are not persisted to disk -> reported as None.
        self.assertIsNone(payload["faces_detected"])
        self.assertIsNone(payload["faces_retained"])

    def test_persons_come_from_clusters_and_match_records(self) -> None:
        write_clusters(
            self.output_dir / "person_clusters.json",
            [
                {"person_id": "Person #1", "photos": ["a.jpg", "b.jpg"]},
                {"person_id": "Person #2", "photos": ["b.jpg"]},
            ],
        )
        write_records(
            self.output_dir / "person_photo_records.csv",
            [
                ["Person #1", "a.jpg", 0, "", "KEEP"],
                ["Person #1", "b.jpg", 0, "0.9", "KEEP"],
                ["Person #2", "b.jpg", 1, "", "KEEP"],
            ],
        )
        payload = api._build_job_results(
            VALID_JOB, self.upload_dir, self.output_dir, summary=None
        )
        assert payload is not None
        by_person = {p["person_id"]: p["photos"] for p in payload["persons"]}
        self.assertEqual(by_person["Person #1"], ["a.jpg", "b.jpg"])
        self.assertEqual(by_person["Person #2"], ["b.jpg"])
        # The shared photo b.jpg appears under both people.
        self.assertIn("b.jpg", by_person["Person #1"])
        self.assertIn("b.jpg", by_person["Person #2"])
        # Every uploaded original is listed, not only photos with people.
        self.assertEqual(payload["photos"], ["a.jpg", "b.jpg"])
        # No preview manifest -> every person reports "no preview available".
        for person in payload["persons"]:
            self.assertEqual(
                person["preview"], {"available": False, "file": None}
            )

    def test_preview_info_is_attached_when_a_crop_exists(self) -> None:
        write_clusters(
            self.output_dir / "person_clusters.json",
            [
                {"person_id": "Person #1", "photos": ["a.jpg"]},
                {"person_id": "Person #2", "photos": ["b.jpg"]},
            ],
        )
        write_records(
            self.output_dir / "person_photo_records.csv",
            [
                ["Person #1", "a.jpg", 0, "", "KEEP"],
                ["Person #2", "b.jpg", 0, "", "KEEP"],
            ],
        )
        write_manifest(
            self.output_dir,
            {
                "Person #1": {"file": "Person_1.jpg", "photo": "a.jpg"},
                # Person #2 has a manifest entry but no crop file on disk.
                "Person #2": {"file": "Person_2.jpg", "photo": "b.jpg"},
            },
        )
        crop = self.output_dir / api.PREVIEW_DIRNAME / "Person_1.jpg"
        crop.write_bytes(b"\xff\xd8jpegcrop")

        payload = api._build_job_results(
            VALID_JOB, self.upload_dir, self.output_dir, summary=None
        )
        assert payload is not None
        by_person = {p["person_id"]: p["preview"] for p in payload["persons"]}
        self.assertEqual(
            by_person["Person #1"], {"available": True, "file": "Person_1.jpg"}
        )
        # A manifest entry whose crop file is missing must fall back.
        self.assertEqual(
            by_person["Person #2"], {"available": False, "file": None}
        )

    def test_records_rebuilt_when_clusters_json_missing(self) -> None:
        write_records(
            self.output_dir / "person_photo_records.csv",
            [
                ["Person #1", "a.jpg", 0, "", "KEEP"],
                ["Person #2", "b.jpg", 0, "", "KEEP"],
                ["Person #1", "b.jpg", 0, "0.8", "KEEP"],
            ],
        )
        payload = api._build_job_results(
            VALID_JOB, self.upload_dir, self.output_dir, summary=None
        )
        assert payload is not None
        by_person = {p["person_id"]: p["photos"] for p in payload["persons"]}
        self.assertEqual(by_person["Person #1"], ["a.jpg", "b.jpg"])
        self.assertEqual(by_person["Person #2"], ["b.jpg"])

    def test_scalar_counts_come_from_summary_when_present(self) -> None:
        write_clusters(
            self.output_dir / "person_clusters.json",
            [{"person_id": "Person #1", "photos": ["a.jpg"]}],
        )
        write_records(
            self.output_dir / "person_photo_records.csv",
            [["Person #1", "a.jpg", 0, "", "KEEP"]],
        )
        summary = {
            "photos_uploaded": 5,
            "photos_processed": 4,
            "faces_detected": 9,
            "faces_retained": 7,
            "unresolved_uncertain_faces": 2,
            "person_photo_records": 1,
            "merge_sort": {"records": 1, "ordered": True, "time_ms": 0.42},
        }
        payload = api._build_job_results(
            VALID_JOB, self.upload_dir, self.output_dir, summary=summary
        )
        assert payload is not None
        self.assertEqual(payload["photos_uploaded"], 5)
        self.assertEqual(payload["photos_processed"], 4)
        self.assertEqual(payload["faces_detected"], 9)
        self.assertEqual(payload["faces_retained"], 7)
        self.assertEqual(payload["unresolved_uncertain_faces"], 2)
        self.assertEqual(payload["merge_sort"]["time_ms"], 0.42)


class SortVerificationTests(unittest.TestCase):
    def test_records_sorted_numeric_aware(self) -> None:
        ordered = [
            {"person_id": "Person #1"},
            {"person_id": "Person #2"},
            {"person_id": "Person #10"},
        ]
        self.assertTrue(api._records_are_sorted(ordered))
        unordered = [
            {"person_id": "Person #10"},
            {"person_id": "Person #2"},
        ]
        self.assertFalse(api._records_are_sorted(unordered))

    def test_person_photos_are_basenames_and_deduplicated(self) -> None:
        clusters = {
            "persons": [
                {"person_id": "Person #1", "photos": ["a.jpg", "a.jpg", "dir/b.jpg"]},
            ]
        }
        persons = api._persons_from_clusters(clusters)
        self.assertEqual(persons[0]["photos"], ["a.jpg", "b.jpg"])
        self.assertEqual(persons[0]["photo_count"], 2)


class PhotoEndpointGuardTests(unittest.TestCase):
    """The photo route must only serve files that belong to the job."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.upload_root = self.root / "uploads"
        self.job_dir = self.upload_root / VALID_JOB
        self.job_dir.mkdir(parents=True)
        (self.job_dir / "event01.jpeg").write_bytes(b"jpeg-bytes")
        self._patch = mock.patch.object(api, "UPLOAD_ROOT", self.upload_root)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def test_serves_original_byte_for_byte(self) -> None:
        response = api.photo(VALID_JOB, "event01.jpeg")
        self.assertIsInstance(response, FileResponse)
        self.assertEqual(Path(response.path).name, "event01.jpeg")

    def test_rejects_path_separators(self) -> None:
        for name in ("../run.py", "..%2frun.py", "sub/event01.jpeg", "..\\run.py"):
            with self.subTest(name=name):
                with self.assertRaises(HTTPException) as ctx:
                    api.photo(VALID_JOB, name)
                self.assertIn(ctx.exception.status_code, (400, 404))

    def test_rejects_file_not_uploaded(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            api.photo(VALID_JOB, "not-uploaded.jpg")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_rejects_unknown_job(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            api.photo("ffffffffffff", "event01.jpeg")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_rejects_malformed_job_id(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            api.photo("..", "event01.jpeg")
        self.assertEqual(ctx.exception.status_code, 404)


class ResultsEndpointStateTests(unittest.TestCase):
    """Unknown / processing / error / done are distinct, appropriate responses."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.upload_root = self.root / "uploads"
        self.output_root = self.root / "output"
        self.upload_root.mkdir()
        self.output_root.mkdir()
        self._patches = [
            mock.patch.object(api, "UPLOAD_ROOT", self.upload_root),
            mock.patch.object(api, "OUTPUT_ROOT", self.output_root),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()
        self._tmp.cleanup()

    def _store_with(self, status: str):
        store = api.JobStore()
        job = store.create()
        job.status = status
        return store, job

    def test_unknown_job_is_404(self) -> None:
        with mock.patch.object(api, "JOBS", api.JobStore()):
            with self.assertRaises(HTTPException) as ctx:
                api.results(VALID_JOB)
            self.assertEqual(ctx.exception.status_code, 404)

    def test_processing_job_is_202(self) -> None:
        store, job = self._store_with("processing")
        with mock.patch.object(api, "JOBS", store):
            with self.assertRaises(HTTPException) as ctx:
                api.results(job.job_id)
            self.assertEqual(ctx.exception.status_code, 202)

    def test_failed_job_is_500(self) -> None:
        store, job = self._store_with("error")
        job.error = "FileNotFoundError: boom"
        with mock.patch.object(api, "JOBS", store):
            with self.assertRaises(HTTPException) as ctx:
                api.results(job.job_id)
            self.assertEqual(ctx.exception.status_code, 500)

    def test_done_job_reads_files(self) -> None:
        store, job = self._store_with("done")
        job.filenames = ["a.jpg"]
        job.upload_dir.mkdir(parents=True)
        (job.upload_dir / "a.jpg").write_bytes(b"x")
        job.output_dir.mkdir(parents=True)
        write_clusters(
            job.output_dir / "person_clusters.json",
            [{"person_id": "Person #1", "photos": ["a.jpg"]}],
        )
        write_records(
            job.output_dir / "person_photo_records.csv",
            [["Person #1", "a.jpg", 0, "", "KEEP"]],
        )
        with mock.patch.object(api, "JOBS", store):
            payload = api.results(job.job_id)
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["people_identified"], 1)

    def test_done_job_without_output_files_is_500(self) -> None:
        store, job = self._store_with("done")
        job.upload_dir.mkdir(parents=True)
        job.output_dir.mkdir(parents=True)
        with mock.patch.object(api, "JOBS", store):
            with self.assertRaises(HTTPException) as ctx:
                api.results(job.job_id)
            self.assertEqual(ctx.exception.status_code, 500)


class FacePreviewEndpointGuardTests(unittest.TestCase):
    """The /previews route must only serve crops inside the job's previews dir."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.output_root = self.root / "output"
        self.preview_dir = self.output_root / VALID_JOB / api.PREVIEW_DIRNAME
        self.preview_dir.mkdir(parents=True)
        (self.preview_dir / "Person_1.jpg").write_bytes(b"\xff\xd8jpegcrop")
        self._patch = mock.patch.object(api, "OUTPUT_ROOT", self.output_root)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def test_serves_the_crop(self) -> None:
        response = api.face_preview(VALID_JOB, "Person_1.jpg")
        self.assertIsInstance(response, FileResponse)
        self.assertEqual(Path(response.path).name, "Person_1.jpg")
        self.assertEqual(response.media_type, "image/jpeg")

    def test_rejects_path_separators(self) -> None:
        for name in ("../run.py", "sub/Person_1.jpg", "..\\run.py"):
            with self.subTest(name=name):
                with self.assertRaises(HTTPException) as ctx:
                    api.face_preview(VALID_JOB, name)
                self.assertIn(ctx.exception.status_code, (400, 404))

    def test_rejects_file_not_generated_for_job(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            api.face_preview(VALID_JOB, "Person_9.jpg")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_rejects_unknown_and_malformed_job_id(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            api.face_preview("ffffffffffff", "Person_1.jpg")
        self.assertEqual(ctx.exception.status_code, 404)
        with self.assertRaises(HTTPException) as ctx:
            api.face_preview("..", "Person_1.jpg")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
