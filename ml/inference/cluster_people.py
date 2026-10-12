"""Final person clustering + anonymous Person IDs + custom Merge Sort (ML runner).

Thin convenience wrapper around the shared application orchestrator
:mod:`app.processing.photo_processor`, which is also what ``run.py`` /
``app/main.py`` execute. Keeping a single implementation means the application
entry point and this ML-side runner can never diverge.

It REUSES the already-verified pipeline (``ml.inference.match_faces``:
detection -> filtering -> embeddings -> uncertain recovery) unchanged and adds:

    usable faces -> centroid-based person clustering -> anonymous Person IDs
                 -> Person-Photo records -> custom Merge Sort
                 -> person-wise photo galleries

Usage
-----
    python ml/inference/cluster_people.py --dir ml/datasets/test --glob "event*" --det-size 640
    python ml/inference/cluster_people.py --cluster-threshold 0.55
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (``python ml/inference/cluster_people.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.processing.photo_processor import run_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline over a directory of images; return an exit code."""
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
