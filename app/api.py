"""Local FastAPI backend for AI Photo Segregation.

The API is a thin, local-only wrapper around the EXISTING pipeline. It does not
re-implement any AI/DAA stage: it writes uploaded originals into a job-specific
directory and calls :func:`app.processing.photo_processor.run_pipeline` with
``--dir`` and ``--out-dir`` set to that job's directories.

Endpoints
---------
* ``GET  /health``            - liveness + environment info
* ``POST /upload``            - multipart upload; validates images, creates a job
* ``POST /process/{job_id}``  - starts the existing pipeline in a background thread
* ``GET  /status/{job_id}``   - real status + results for the job
* ``GET  /results/{job_id}``  - the real, file-backed results of a finished job
* ``GET  /photos/{job_id}/{filename}`` - the pristine original uploaded photo

Everything lives on the local machine: no cloud, auth, or database. Jobs are
isolated under ``data/uploads/<job_id>/`` and ``data/output/<job_id>/``.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow ``python run_api.py`` / ``python -m app.api`` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from PIL import Image  # noqa: E402

from app.algorithms.merge_sort import person_id_sort_key  # noqa: E402
from app.processing.face_previews import (  # noqa: E402
    PREVIEW_DIRNAME,
    generate_person_previews,
    read_face_previews,
)
from app.processing.photo_processor import build_parser, run_pipeline  # noqa: E402
from ml.inference import predict as P  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

MAX_FILES = 120
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB per photo

# Output files written by the EXISTING pipeline into ``data/output/<job_id>/``.
PERSON_CLUSTERS_JSON = "person_clusters.json"
PERSON_PHOTO_RECORDS_CSV = "person_photo_records.csv"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Job ids are ``uuid4().hex[:12]``; validating this shape keeps a request from
# ever being turned into a filesystem path it should not touch.
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# Image media types by extension, so served originals keep their real type.
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


# ---------------------------------------------------------------------------
# Job registry (in-memory, local only)
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One upload/processing session, isolated on disk."""

    job_id: str
    created_at: float
    upload_dir: Path
    output_dir: Path
    status: str = "uploaded"  # uploaded | processing | done | error
    filenames: list[str] = field(default_factory=list)
    total_bytes: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    preview_error: str | None = None

    def public(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "photos_uploaded": len(self.filenames),
            "files": self.filenames,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        if self.preview_error is not None:
            payload["preview_error"] = self.preview_error
        return payload


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            created_at=time.time(),
            upload_dir=UPLOAD_ROOT / job_id,
            output_dir=OUTPUT_ROOT / job_id,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)


JOBS = JobStore()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    """Reduce an uploaded name to a safe basename (no path separators)."""
    base = Path(name or "").name
    base = _SAFE_NAME_RE.sub("_", base).strip("._-")
    if not base:
        base = "image"
    return base[:120]


def _validate_image(data: bytes) -> tuple[bool, str]:
    """Return ``(ok, reason)`` by actually opening the bytes as an image."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            image.verify()  # integrity check (no full decode)
    except Exception as exc:  # unreadable / truncated / not an image
        return False, f"not a readable image ({type(exc).__name__})"
    if image_format not in ALLOWED_PIL_FORMATS:
        return False, f"unsupported image format {image_format}"
    return True, ""


def _unique_path(directory: Path, filename: str) -> Path:
    """A non-colliding path inside ``directory`` (never overwrites)."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    index = 1
    while True:
        candidate = directory / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


# ---------------------------------------------------------------------------
# Pipeline runner (background thread)
# ---------------------------------------------------------------------------


def _merge_sort_ordered(records: list[Any]) -> bool:
    return all(
        person_id_sort_key(records[i].person_id)
        <= person_id_sort_key(records[i + 1].person_id)
        for i in range(len(records) - 1)
    )


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _summarize(job: Job, result: Any) -> dict[str, Any]:
    """Build the job-specific result payload from the real pipeline output."""
    persons = [
        {
            "person_id": person.person_id,
            "photo_count": len(result.photo_index.get(person.person_id, [])),
        }
        for person in result.persons
    ]
    return {
        "job_id": job.job_id,
        "photos_uploaded": len(job.filenames),
        "photos_processed": result.photos,
        "faces_detected": result.detected,
        "faces_deduplicated": result.duplicates_removed,
        "faces_retained": result.retained,
        "embeddings": result.retained,
        "keep": result.kept,
        "confirmed_from_match": result.confirmed,
        "people_identified": len(result.persons),
        "multi_photo_persons": result.multi_photo_persons,
        "unresolved_uncertain_faces": len(result.unresolved),
        "uncertain_promoted": len(result.promotions),
        "person_photo_records": len(result.records_sorted),
        "merge_sort": {
            "records": len(result.records_sorted),
            "ordered": _merge_sort_ordered(result.records_sorted),
            "time_ms": round(result.sort_seconds * 1000, 3),
        },
        "persons": persons,
        "cluster_threshold": result.cluster_threshold,
        "det_size": result.det_size,
        "timings": {
            "model_load_s": round(result.load_seconds, 3),
            "clustering_s": round(result.cluster_seconds, 3),
            "merge_sort_s": round(result.sort_seconds, 3),
        },
        "outputs": {
            "person_clusters_json": _rel(result.json_path),
            "person_photo_records_csv": _rel(result.csv_path),
            "confirmed_uncertain_matches_csv": _rel(result.confirmed_csv_path),
            "singleton_validation_csv": _rel(result.singleton_csv_path),
        },
    }


# ---------------------------------------------------------------------------
# Reading real pipeline output (source of truth: person_clusters.json + CSV)
# ---------------------------------------------------------------------------


def _list_job_photos(upload_dir: Path) -> list[str]:
    """Names of the real image files stored for a job (never a path)."""
    if not upload_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in upload_dir.iterdir()
        if entry.is_file() and entry.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _read_person_clusters(path: Path) -> dict[str, Any] | None:
    """Load ``person_clusters.json`` if it exists and is valid JSON."""
    if not path.is_file():
        return None
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_records_csv(path: Path) -> list[dict[str, str]] | None:
    """Load the Person-Photo record rows from ``person_photo_records.csv``."""
    if not path.is_file():
        return None
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None


def _records_are_sorted(rows: list[dict[str, str]]) -> bool:
    """Verify the real records are ordered by Person ID (numeric-aware)."""
    ids = [row.get("person_id", "") for row in rows]
    return all(
        person_id_sort_key(ids[i]) <= person_id_sort_key(ids[i + 1])
        for i in range(len(ids) - 1)
    )


def _persons_from_clusters(clusters: dict[str, Any]) -> list[dict[str, Any]]:
    """Person ID -> associated original photo filenames (ordered, de-duplicated)."""
    persons: list[dict[str, Any]] = []
    for person in clusters.get("persons", []):
        if not isinstance(person, dict):
            continue
        photos: list[str] = []
        for photo in person.get("photos", []):
            name = Path(str(photo)).name
            if name and name not in photos:
                photos.append(name)
        persons.append(
            {
                "person_id": person.get("person_id"),
                "photos": photos,
                "photo_count": len(photos),
            }
        )
    return persons


def _persons_from_records(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Fallback: rebuild person -> photos from the CSV in first-seen order."""
    order: list[str] = []
    photos_by_person: dict[str, list[str]] = {}
    for row in rows:
        person_id = row.get("person_id") or ""
        name = Path(row.get("photo") or "").name
        if not person_id or not name:
            continue
        if person_id not in photos_by_person:
            photos_by_person[person_id] = []
            order.append(person_id)
        if name not in photos_by_person[person_id]:
            photos_by_person[person_id].append(name)
    return [
        {
            "person_id": person_id,
            "photos": photos_by_person[person_id],
            "photo_count": len(photos_by_person[person_id]),
        }
        for person_id in order
    ]


def _build_job_results(
    job_id: str,
    upload_dir: Path,
    output_dir: Path,
    *,
    summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Assemble the results payload from the real, on-disk pipeline output.

    ``person_clusters.json`` is the source of truth for the person -> photos
    mapping and ``person_photo_records.csv`` for the record count + sorted-order
    verification. ``summary`` (the pipeline result kept in memory) only supplies
    the scalar counts that are not persisted to disk. Returns ``None`` when the
    job has no usable output files at all.
    """
    clusters = _read_person_clusters(output_dir / PERSON_CLUSTERS_JSON)
    rows = _read_records_csv(output_dir / PERSON_PHOTO_RECORDS_CSV)
    if clusters is None and rows is None:
        return None

    if clusters is not None:
        persons = _persons_from_clusters(clusters)
    else:
        persons = _persons_from_records(rows or [])

    # Every original photo uploaded for the job (real filenames from the job's
    # upload directory), so the UI can browse the full event, not just the
    # photos that happen to contain an identified person.
    all_photos = _list_job_photos(upload_dir)

    # Attach the representative face-preview info the UI needs. "available" is
    # only true when the crop file the manifest points at actually exists, so a
    # person without a usable crop simply falls back to the folder icon.
    previews = read_face_previews(output_dir)
    preview_dir = output_dir / PREVIEW_DIRNAME
    for person in persons:
        info = previews.get(person.get("person_id"))
        filename = info.get("file") if isinstance(info, dict) else None
        available = bool(filename) and (preview_dir / str(filename)).is_file()
        person["preview"] = {
            "available": available,
            "file": str(filename) if available else None,
        }

    metadata = clusters.get("metadata", {}) if clusters else {}
    sorted_ok = _records_are_sorted(rows) if rows is not None else None

    if summary is not None:
        photos_uploaded = summary.get("photos_uploaded")
        photos_processed = summary.get("photos_processed")
        faces_detected = summary.get("faces_detected")
        faces_deduplicated = summary.get("faces_deduplicated")
        faces_retained = summary.get("faces_retained")
        unresolved = summary.get("unresolved_uncertain_faces")
        sort_time_ms = summary.get("merge_sort", {}).get("time_ms")
        cluster_threshold = summary.get("cluster_threshold")
        det_size = summary.get("det_size")
        outputs = summary.get("outputs")
        timings = summary.get("timings")
    else:
        photos_uploaded = None  # derived from the job's upload directory below
        photos_processed = metadata.get("photos")
        faces_detected = None
        faces_deduplicated = metadata.get("duplicate_detections_removed")
        faces_retained = None
        unresolved = (
            len(clusters.get("unresolved_uncertain", [])) if clusters else None
        )
        sort_time_ms = None
        cluster_threshold = metadata.get("cluster_threshold")
        det_size = metadata.get("det_size")
        outputs = None
        timings = None

    if photos_uploaded is None:
        photos_uploaded = len(_list_job_photos(upload_dir))

    record_count = len(rows) if rows is not None else None
    if record_count is None and summary is not None:
        record_count = summary.get("person_photo_records")

    return {
        "job_id": job_id,
        "status": "done",
        "photos_uploaded": photos_uploaded,
        "photos_processed": photos_processed,
        "faces_detected": faces_detected,
        "faces_deduplicated": faces_deduplicated,
        "faces_retained": faces_retained,
        "people_identified": len(persons),
        "unresolved_uncertain_faces": unresolved,
        "person_photo_records": record_count,
        "unresolved_uncertain": clusters.get("unresolved_uncertain", [])
        if clusters
        else [],
        "merge_sort": {
            # ``ordered`` is a real verification of the on-disk records, never a
            # hardcoded PASS. It is ``None`` when no records could be verified.
            "records": record_count,
            "ordered": sorted_ok,
            "verified_from_records": rows is not None,
            "time_ms": sort_time_ms,
        },
        "persons": persons,
        "photos": all_photos,
        "cluster_threshold": cluster_threshold,
        "det_size": det_size,
        "timings": timings,
        "outputs": outputs,
    }


def _run_job(job: Job) -> None:
    """Run the existing pipeline for ``job`` and record the real outcome."""
    job.status = "processing"
    job.started_at = time.time()
    job.error = None
    try:
        args = build_parser().parse_args(
            ["--dir", str(job.upload_dir), "--out-dir", str(job.output_dir)]
        )
        result = run_pipeline(args)
        job.result = _summarize(job, result)
        # Display-only face crops for the Results UI: built from the bounding
        # boxes the pipeline already computed (no detection is re-run). This is
        # best-effort - a failure must never fail the job or the results page.
        try:
            generate_person_previews(result.persons, job.upload_dir, job.output_dir)
        except Exception as exc:  # noqa: BLE001 - previews are best-effort
            job.preview_error = f"{type(exc).__name__}: {exc}"
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - surface the real failure to the client
        job.error = f"{type(exc).__name__}: {exc}"
        job.status = "error"
    finally:
        job.finished_at = time.time()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Photo Segregation API", version="0.1.0")

# The Vite dev server proxies /api -> this server, but allow direct local calls too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": P.MODEL_NAME,
        "providers": P.DEFAULT_PROVIDERS,
        "jobs": JOBS.count(),
        "upload_root": _rel(UPLOAD_ROOT),
        "output_root": _rel(OUTPUT_ROOT),
        "max_files": MAX_FILES,
        "max_file_bytes": MAX_FILE_BYTES,
    }


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Validate and store uploaded photos for a new job."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400, detail=f"Too many files (max {MAX_FILES})."
        )

    job = JOBS.create()

    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    valid: list[tuple[str, bytes]] = []

    for upload_file in files:
        original_name = upload_file.filename or "image"
        extension = Path(original_name).suffix.lower()
        data = await upload_file.read()

        if extension not in ALLOWED_EXTENSIONS:
            rejected.append(
                {"name": original_name, "reason": f"unsupported file type '{extension or 'none'}'"}
            )
            continue
        if len(data) == 0:
            rejected.append({"name": original_name, "reason": "empty file"})
            continue
        if len(data) > MAX_FILE_BYTES:
            rejected.append(
                {"name": original_name, "reason": f"exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit"}
            )
            continue

        ok, reason = _validate_image(data)
        if not ok:
            rejected.append({"name": original_name, "reason": reason})
            continue

        valid.append((_safe_filename(original_name), data))

    if not valid:
        # Nothing usable: drop the job so no empty job directory is left behind.
        JOBS.remove(job.job_id)
        raise HTTPException(
            status_code=400,
            detail={"message": "No valid images were uploaded.", "rejected": rejected},
        )

    # Only create the job directory once there is something to store.
    job.upload_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in valid:
        target = _unique_path(job.upload_dir, filename)
        target.write_bytes(data)  # pristine copy; never modified afterwards
        accepted.append(target.name)
        job.total_bytes += len(data)

    job.filenames = accepted

    return {
        "job_id": job.job_id,
        "photos_uploaded": len(accepted),
        "files": accepted,
        "rejected": rejected,
    }


@app.post("/process/{job_id}")
def process(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    if not job.filenames:
        raise HTTPException(status_code=400, detail="Job has no uploaded photos.")
    if job.status == "processing":
        return {"job_id": job_id, "status": "processing", "message": "Already running."}
    if job.status == "done":
        return {"job_id": job_id, "status": "done", "message": "Already finished."}

    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job.public()


@app.get("/results/{job_id}")
def results(job_id: str) -> dict[str, Any]:
    """Return the real results of a finished job (never reruns the pipeline).

    * ``404`` - unknown job id (not in memory and no job directories on disk)
    * ``202`` - the job has not finished yet (uploaded / processing)
    * ``500`` - the pipeline failed, or its output files are missing
    """
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="Unknown job id.")

    job = JOBS.get(job_id)
    if job is not None:
        if job.status in ("uploaded", "processing"):
            raise HTTPException(
                status_code=202,
                detail={
                    "job_id": job_id,
                    "status": job.status,
                    "message": "Job has not finished processing yet.",
                },
            )
        if job.status == "error":
            raise HTTPException(
                status_code=500,
                detail={
                    "job_id": job_id,
                    "status": "error",
                    "message": "The pipeline reported an error for this job.",
                    "error": job.error,
                },
            )
        payload = _build_job_results(
            job_id, job.upload_dir, job.output_dir, summary=job.result
        )
    else:
        # Not in memory (e.g. after a server restart): rebuild from disk if the
        # job directories still exist. No pipeline rerun, no fabricated data.
        upload_dir = UPLOAD_ROOT / job_id
        output_dir = OUTPUT_ROOT / job_id
        if not upload_dir.is_dir() and not output_dir.is_dir():
            raise HTTPException(status_code=404, detail="Unknown job id.")
        payload = _build_job_results(
            job_id, upload_dir, output_dir, summary=None
        )

    if payload is None:
        raise HTTPException(
            status_code=500,
            detail={
                "job_id": job_id,
                "status": "error",
                "message": "No pipeline output files were found for this job.",
            },
        )
    return payload


@app.get("/photos/{job_id}/{filename}")
def photo(job_id: str, filename: str) -> FileResponse:
    """Serve one ORIGINAL uploaded photo for a job, unmodified.

    The file is streamed straight from ``data/uploads/<job_id>/`` with no
    resizing, cropping, annotation or recompression. Only files that actually
    belong to the job can be requested: the job id is validated, the filename
    must be a bare basename, and the resolved path must stay inside that job's
    upload directory.
    """
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="Unknown job id.")

    # Reject anything that is not a plain filename (path traversal / separators).
    if filename != Path(filename).name or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    upload_dir = UPLOAD_ROOT / job_id
    if not upload_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unknown job id.")

    job = JOBS.get(job_id)
    if job is not None and filename not in job.filenames:
        raise HTTPException(status_code=404, detail="Photo not found for this job.")

    resolved_dir = upload_dir.resolve()
    candidate = (resolved_dir / filename).resolve()
    if candidate.parent != resolved_dir or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Photo not found for this job.")

    media_type = _MEDIA_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
    return FileResponse(candidate, media_type=media_type)


@app.get("/previews/{job_id}/{filename}")
def face_preview(job_id: str, filename: str) -> FileResponse:
    """Serve one generated face-preview crop for a job.

    These are the small display-only crops written under
    ``data/output/<job_id>/previews/``. Same job-isolation and traversal guards
    as ``/photos``: the id is validated, the filename must be a bare basename,
    and the resolved path must stay inside that job's previews directory.
    """
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="Unknown job id.")
    if filename != Path(filename).name or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    preview_dir = (OUTPUT_ROOT / job_id / PREVIEW_DIRNAME).resolve()
    if not preview_dir.is_dir():
        raise HTTPException(status_code=404, detail="No face preview for this job.")

    candidate = (preview_dir / filename).resolve()
    if candidate.parent != preview_dir or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Face preview not found.")

    return FileResponse(candidate, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
