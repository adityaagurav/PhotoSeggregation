"""Converts clustering output into Person-Photo records and machine-readable files.

A **Person-Photo record** is one (person, photo, face) link::

    PersonPhotoRecord(person_id="Person #1", photo="event05.jpeg",
                      face_id=3, similarity=0.84, status="KEEP")

Because a single photo can contain several people (e.g. ``event05.jpeg`` may hold
Person #1, Person #3 and Person #5), the *same photo appears once per person*.
A photo is therefore **never** assigned to only one person — the data structure
is a flat list of records, and grouping by person happens later (after the custom
Merge Sort) by traversal, not by making photos exclusive.

Records are produced in face-processing order (NOT sorted by Person ID); the
custom Merge Sort is what orders them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from app.ai.face_cluster import face_face_id, face_photo, face_status
from app.processing.person_manager import PersonIdentity


@dataclass
class PersonPhotoRecord:
    """One (person, photo, face) link produced by final clustering."""

    person_id: str
    photo: str
    face_id: int
    similarity: float | None
    status: str


def generate_records(
    persons: list[PersonIdentity], assignments: Iterable[Any]
) -> list[PersonPhotoRecord]:
    """Build Person-Photo records in face-processing order.

    ``assignments`` is the ordered list of
    :class:`app.ai.face_cluster.FaceAssignment` from clustering. Each face is
    mapped to its person via the cluster it was assigned to. The returned order
    is deliberately NOT sorted by Person ID, so the Merge Sort step is meaningful.
    """
    person_by_cluster = {id(person.cluster): person.person_id for person in persons}
    records: list[PersonPhotoRecord] = []
    for assignment in assignments:
        records.append(
            PersonPhotoRecord(
                person_id=person_by_cluster[id(assignment.cluster)],
                photo=face_photo(assignment.face),
                face_id=face_face_id(assignment.face),
                similarity=assignment.similarity,
                status=face_status(assignment.face),
            )
        )
    return records


def build_photo_index(
    sorted_records: list[PersonPhotoRecord],
) -> dict[str, list[str]]:
    """Map ``person_id -> [photos]`` by traversing the (Merge-Sorted) records.

    Photos are de-duplicated per person while preserving the order in which they
    appear, so a photo with several people contributes its name to each person.
    """
    index: dict[str, list[str]] = {}
    for record in sorted_records:
        photos = index.setdefault(record.person_id, [])
        if record.photo not in photos:
            photos.append(record.photo)
    return index


def build_photo_people_index(
    sorted_records: list[PersonPhotoRecord],
) -> dict[str, list[str]]:
    """Map ``photo -> [person_ids]`` (a photo can list several people)."""
    index: dict[str, list[str]] = {}
    for record in sorted_records:
        people = index.setdefault(record.photo, [])
        if record.person_id not in people:
            people.append(record.person_id)
    return index


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_person_photo_records_csv(
    records: list[PersonPhotoRecord], csv_path: str | Path
) -> Path:
    """Write the Person-Photo records (already Merge-Sorted) to CSV."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["person_id", "photo", "face_id", "similarity", "status"])
        for record in records:
            writer.writerow(
                [
                    record.person_id,
                    record.photo,
                    record.face_id,
                    "" if record.similarity is None else f"{record.similarity:.6f}",
                    record.status,
                ]
            )
    return path


def write_person_clusters_json(
    persons: list[PersonIdentity],
    photo_index: dict[str, list[str]],
    json_path: str | Path,
    unresolved_uncertain: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the final machine-readable person clusters to JSON.

    Shape (conceptually)::

        {
          "persons": [
            {"person_id": "Person #1", "photos": ["event01.jpeg", ...]},
            ...
          ],
          "unresolved_uncertain": [...],
          "metadata": {...}
        }
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "persons": [
            {"person_id": person.person_id, "photos": photo_index.get(person.person_id, [])}
            for person in persons
        ],
        "unresolved_uncertain": unresolved_uncertain or [],
        "metadata": metadata or {},
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def records_to_dicts(records: list[PersonPhotoRecord]) -> list[dict[str, Any]]:
    """Serialize records to plain dicts (for JSON or debugging)."""
    return [asdict(record) for record in records]


def write_singleton_validation_csv(
    decisions: list[Any], csv_path: str | Path
) -> Path:
    """Write the singleton qualification outcome (one row per singleton face).

    Columns: ``photo, face_id, relative_face_area, detection_confidence,
    blur_score, yaw, pitch, singleton_status, rejection_reason``.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "photo",
                "face_id",
                "relative_face_area",
                "detection_confidence",
                "blur_score",
                "yaw",
                "pitch",
                "singleton_status",
                "rejection_reason",
            ]
        )
        for decision in decisions:
            writer.writerow(
                [
                    decision.photo,
                    decision.face_id,
                    f"{decision.relative_area:.6f}",
                    f"{decision.det_score:.6f}",
                    f"{decision.blur_score:.4f}",
                    "" if decision.yaw is None else f"{decision.yaw:.4f}",
                    "" if decision.pitch is None else f"{decision.pitch:.4f}",
                    decision.status,
                    decision.rejection_reason,
                ]
            )
    return path
