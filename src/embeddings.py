"""Shared, lazily-loaded models.

Umbra always re-embeds queries and retrieved chunks with its own model rather
than trusting the RAG system's scores, so coverage scores are comparable
across systems (spec §2: RAG-agnostic). all-MiniLM-L6-v2 produces 384-dim
vectors, matching the vector(384) columns in the storage schema.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Sequence

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# torch models (on MPS in particular) are not safe to call from several threads
# at once: concurrent encode() calls abort the process. FastAPI runs sync
# endpoints in a threadpool, so all inference goes through this lock.
MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def get_embedder(model_name: str = EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def embed(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = get_embedder()
    with MODEL_LOCK:
        vectors = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
