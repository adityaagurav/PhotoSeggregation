"""Unit tests for the custom Merge Sort (DAA component).

Run with::

    .venv/bin/python -m unittest tests.test_merge_sort -v
"""

from __future__ import annotations

import unittest

from app.algorithms.merge_sort import merge_sort, person_id_sort_key


class _Rec:
    """Minimal stand-in for a Person-Photo record."""

    def __init__(self, person_id: str, photo: str) -> None:
        self.person_id = person_id
        self.photo = photo

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Rec({self.person_id!r}, {self.photo!r})"


class PersonIdKeyTests(unittest.TestCase):
    def test_numeric_aware_key(self) -> None:
        # "Person #2" must sort before "Person #10" (string order would not).
        self.assertLess(person_id_sort_key("Person #2"), person_id_sort_key("Person #10"))
        self.assertLess(person_id_sort_key("Person #1"), person_id_sort_key("Person #2"))

    def test_non_numeric_falls_back(self) -> None:
        self.assertEqual(person_id_sort_key("anonymous"), (0, "anonymous"))


class MergeSortTests(unittest.TestCase):
    def test_empty_and_single(self) -> None:
        self.assertEqual(merge_sort([]), [])
        only = [_Rec("Person #1", "a.jpg")]
        self.assertEqual(merge_sort(only), only)

    def test_sorts_by_person_id(self) -> None:
        records = [
            _Rec("Person #3", "event05.jpeg"),
            _Rec("Person #1", "event01.jpeg"),
            _Rec("Person #2", "event04.jpeg"),
            _Rec("Person #1", "event07.jpeg"),
            _Rec("Person #3", "event08.jpeg"),
        ]
        result = merge_sort(records)
        self.assertEqual(
            [(r.person_id, r.photo) for r in result],
            [
                ("Person #1", "event01.jpeg"),
                ("Person #1", "event07.jpeg"),
                ("Person #2", "event04.jpeg"),
                ("Person #3", "event05.jpeg"),
                ("Person #3", "event08.jpeg"),
            ],
        )

    def test_numeric_ordering_beyond_nine(self) -> None:
        records = [_Rec(f"Person #{i}", f"{i}.jpg") for i in (10, 2, 1, 12, 3)]
        result = merge_sort(records)
        self.assertEqual(
            [r.person_id for r in result],
            ["Person #1", "Person #2", "Person #3", "Person #10", "Person #12"],
        )

    def test_stability(self) -> None:
        # Equal Person IDs must keep their original relative order.
        records = [
            _Rec("Person #1", "first.jpg"),
            _Rec("Person #2", "b.jpg"),
            _Rec("Person #1", "second.jpg"),
        ]
        result = merge_sort(records)
        p1 = [r.photo for r in result if r.person_id == "Person #1"]
        self.assertEqual(p1, ["first.jpg", "second.jpg"])

    def test_does_not_mutate_input(self) -> None:
        records = [_Rec("Person #2", "b.jpg"), _Rec("Person #1", "a.jpg")]
        snapshot = list(records)
        merge_sort(records)
        self.assertEqual(records, snapshot)

    def test_matches_reference_ordering(self) -> None:
        # Compare against an independent oracle (built-in sort in the TEST only).
        records = [
            _Rec("Person #5", "e.jpg"),
            _Rec("Person #1", "a.jpg"),
            _Rec("Person #3", "c.jpg"),
            _Rec("Person #1", "b.jpg"),
            _Rec("Person #2", "d.jpg"),
        ]
        expected = sorted(records, key=lambda r: person_id_sort_key(r.person_id))
        result = merge_sort(records)
        self.assertEqual(
            [(r.person_id, r.photo) for r in result],
            [(r.person_id, r.photo) for r in expected],
        )

    def test_custom_key_and_tuples(self) -> None:
        tuples = [("Person #3", "c"), ("Person #1", "a"), ("Person #2", "b")]
        result = merge_sort(tuples)
        self.assertEqual([t[0] for t in result], ["Person #1", "Person #2", "Person #3"])


if __name__ == "__main__":
    unittest.main()
