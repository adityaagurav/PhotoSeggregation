"""Command-line experiment for the pretrained face model (Buffalo_L).

This is a thin wrapper kept for backwards compatibility: the actual CLI lives in
``ml/inference/predict.py`` (importable as ``ml.inference.predict``), so both of
these are equivalent:

    python ml/inference/predict.py ml/datasets/test/single_face.png
    python ml/inference/run_model.py ml/datasets/test/single_face.png

It runs detection + embedding extraction on one or more images and reports
bounding boxes, detection confidence, embedding dimensions, timing and — when
two or more faces are available — cosine similarity between them.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (``python ml/inference/run_model.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.inference.predict import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
