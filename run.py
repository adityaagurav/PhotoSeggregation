"""Convenient application launcher.

Runs the final photo pipeline (detection -> filtering -> embeddings -> uncertain
recovery -> person clustering -> anonymous Person IDs -> Merge Sort -> galleries)
through :func:`app.main.main`. The Streamlit web UI is not implemented yet, so
this launcher executes the batch pipeline and prints the results.

Usage::

    python run.py
    python run.py --dir ml/datasets/test --glob "event*" --det-size 640
"""
from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
