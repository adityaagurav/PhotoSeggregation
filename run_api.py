"""Local API launcher for AI Photo Segregation.

Starts the FastAPI backend (``app/api.py``) on the local machine with uvicorn.
This is separate from ``run.py`` (the batch pipeline CLI), which is unchanged.

Usage::

    python run_api.py
    API_PORT=8000 python run_api.py

Then the frontend (Vite dev server) proxies ``/api/*`` to this server.
"""

from __future__ import annotations

import os

import uvicorn

from app.api import app

if __name__ == "__main__":
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
