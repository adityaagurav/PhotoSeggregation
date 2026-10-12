"""Face similarity and matching utilities.

Thin application-level wrappers around the verified similarity function in
:mod:`ml.inference.predict`. The cosine-similarity maths is **not** re-implemented
here — the pipeline already has one authoritative implementation
(``compare_embeddings``), and everything (grouping, uncertain recovery, final
clustering) reuses it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.inference.predict import compare_embeddings


def cosine_similarity(embedding_a: Any, embedding_b: Any) -> float:
    """Cosine similarity between two embeddings (``-1``..``1``, higher = closer).

    Delegates to :func:`ml.inference.predict.compare_embeddings` so the exact
    same maths is used everywhere.
    """
    return compare_embeddings(
        np.asarray(embedding_a, dtype=np.float32),
        np.asarray(embedding_b, dtype=np.float32),
    )


def is_same_person(embedding_a: Any, embedding_b: Any, threshold: float) -> bool:
    """Return ``True`` when two embeddings are at least ``threshold`` similar.

    ``threshold`` is configurable; there is no single "magic" value that means
    "same person" (see the final clustering documentation).
    """
    return cosine_similarity(embedding_a, embedding_b) >= threshold
