"""Streamlit entry point for the AI Photo Segregation web app.

Step 1 implements ONLY the landing page. Upload, processing, person galleries and
all backend logic are intentionally not implemented yet.

Start the web application with::

    .venv/bin/python -m streamlit run app/web/streamlit_app.py

Then open the local URL Streamlit prints (default http://localhost:8501).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit adds this file's directory to sys.path, not the project root, so add
# the root explicitly to allow ``app.web.*`` imports when run via `streamlit run`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.web.landing import render  # noqa: E402

st.set_page_config(
    page_title="AI Photo Segregation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render()
