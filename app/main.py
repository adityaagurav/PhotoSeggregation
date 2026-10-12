"""Main application entry point.

This module wires the AI processing pipeline together and is what ``run.py``
launches. The current stage runs the **final batch pipeline**:

    detection -> filtering -> embeddings -> uncertain recovery
              -> person clustering -> anonymous Person IDs
              -> Person-Photo records -> custom Merge Sort -> galleries

The interactive web interface (Streamlit upload / naming UI) is intentionally
NOT implemented yet; ``main`` therefore runs the pipeline over a directory of
photographs and prints the results. The entry point is fully implemented and
runs end-to-end.

Usage::

    python run.py
    python run.py --dir ml/datasets/test --glob "event*" --det-size 640
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (``python app/main.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.processing.photo_processor import main as run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the final photo pipeline; return a process exit code."""
    return run_pipeline(argv)


if __name__ == "__main__":
    raise SystemExit(main())
