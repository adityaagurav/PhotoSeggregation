"""Custom Merge Sort implementation (DAA component).

Sorts Person-Photo records like ``(Person_ID, Photo_ID, ...)`` primarily by
Person ID before they are traversed to build person-wise photo galleries.

This is the DAA ("Design and Analysis of Algorithms") component of the project:
it orders the records that the AI pipeline produced. It does **not** perform any
face recognition and must never be used to decide which faces belong to the same
person.

Python's built-in ``sorted`` / ``list.sort`` must NOT be used to replace this
implementation in the pipeline.

Complexity (n = number of Person-Photo records)
-----------------------------------------------
* Time:  O(n log n)   — the list is divided in half (log n levels) and each level
  merges n records in linear time.
* Auxiliary space: O(n) — each merge allocates a new combined list; the total
  extra memory across the recursion is linear in n.

The sort is stable: records with the same Person ID keep their original relative
order (so photos stay in the order the pipeline produced them).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar("T")

# Anonymous Person IDs look like ``Person #1``; the leading integer is used for
# the ordering so ``Person #2`` sorts before ``Person #10`` (a plain string sort
# would not do that).
_PERSON_ID_RE = re.compile(r"(\d+)")


def person_id_sort_key(person_id: Any) -> tuple[int, str]:
    """Return a numeric-aware sort key for an anonymous Person ID.

    ``"Person #2"`` -> ``(2, "Person #2")``. IDs without a number fall back to
    ``(0, <text>)`` so they still have a total, deterministic order.
    """
    text = "" if person_id is None else str(person_id)
    match = _PERSON_ID_RE.search(text)
    if match:
        return (int(match.group(1)), text)
    return (0, text)


def _default_key(record: T) -> tuple[int, str]:
    """Default key: sort by the record's ``person_id`` attribute (or tuple slot)."""
    person_id = getattr(record, "person_id", None)
    if person_id is None and isinstance(record, (tuple, list)) and len(record) > 0:
        person_id = record[0]
    return person_id_sort_key(person_id)


def merge_sort(records: Sequence[T], key: Callable[[T], Any] | None = None) -> list[T]:
    """Return a NEW list of ``records`` sorted with our own recursive Merge Sort.

    Parameters
    ----------
    records:
        The records to sort (any sequence). The input is never mutated.
    key:
        Optional key function mapping a record to a sortable value. Defaults to
        the record's ``person_id`` (numeric-aware, see
        :func:`person_id_sort_key`).

    Returns
    -------
    list
        A new list sorted in ascending order, stable for equal keys.
    """
    items = list(records)  # copy only; no sorting built-in is used
    key_fn = key or _default_key
    return _merge_sort(items, key_fn)


def _merge_sort(items: list[T], key_fn: Callable[[T], Any]) -> list[T]:
    """Recursive divide step: split, sort each half, then merge."""
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    left = _merge_sort(items[:mid], key_fn)
    right = _merge_sort(items[mid:], key_fn)
    return _merge(left, right, key_fn)


def _merge(left: list[T], right: list[T], key_fn: Callable[[T], Any]) -> list[T]:
    """Merge two already-sorted lists into one sorted list (stable)."""
    merged: list[T] = []
    i = 0
    j = 0
    n_left = len(left)
    n_right = len(right)
    while i < n_left and j < n_right:
        # ``<=`` keeps the sort stable (ties take the left/earlier element first).
        if key_fn(left[i]) <= key_fn(right[j]):
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    # Append whatever is left; these slices are already in order.
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
