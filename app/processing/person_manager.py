"""Manages temporary (anonymous) Person IDs and their assignments.

Person IDs are created **dynamically** at runtime from the clustering output:
``Person #1``, ``Person #2``, ``Person #3`` ... The system only ever knows
anonymous clusters — no names, no hardcoded identities, no filename-as-identity,
and no assumption that ``Person #1`` is any particular real person. A human can
later map an anonymous ID to a real name through the (not-yet-built) web
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The only place a Person ID string is formatted. IDs are positional labels over
# anonymous clusters, never identities.
PERSON_ID_TEMPLATE = "Person #{index}"


def make_person_id(index: int) -> str:
    """Return the anonymous ID for a 1-based person index (``Person #1``)."""
    return PERSON_ID_TEMPLATE.format(index=index)


@dataclass
class PersonIdentity:
    """An anonymous person and the cluster of faces assigned to it."""

    person_id: str
    cluster: Any

    @property
    def size(self) -> int:
        return len(self.cluster.members)


class PersonManager:
    """Assigns anonymous Person IDs to clusters, in dynamic first-appearance order.

    IDs are generated from the order clusters were created while processing
    faces, so they are stable for a given run and contain no identity information.
    """

    def __init__(self, id_prefix: str = "Person #") -> None:
        self.id_prefix = id_prefix
        self.persons: list[PersonIdentity] = []

    def assign(self, clusters: list[Any]) -> list[PersonIdentity]:
        """Assign ``Person #1..N`` to ``clusters`` and remember them.

        The clusters are numbered in the order they are given (cluster creation
        order), which is deterministic for a given face ordering.
        """
        self.persons = [
            PersonIdentity(person_id=make_person_id(index), cluster=cluster)
            for index, cluster in enumerate(clusters, start=1)
        ]
        return self.persons

    def __len__(self) -> int:
        return len(self.persons)
