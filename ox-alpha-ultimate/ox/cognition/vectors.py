"""Dependency-free text embeddings for memory recall.

A deterministic hashed n-gram bag-of-features model. No model download, no
network, no training step: text -> fixed-width L2-normalised float32 vector,
compared by cosine similarity. Good enough to rank semantic recall candidates
and cheap enough to run on every episodic write.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

DIMENSIONS = 512
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(str(text).lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def embed(text: str, dimensions: int = DIMENSIONS) -> np.ndarray:
    """Hashed TF vector, sublinear-term weighted and L2 normalised."""
    vector = np.zeros(dimensions, dtype=np.float32)
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "little") % dimensions
        sign = 1.0 if digest[0] & 1 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
