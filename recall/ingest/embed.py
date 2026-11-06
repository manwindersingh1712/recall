"""Embedders: real (fastembed bge-small) and a deterministic hash test double."""

import hashlib
import math
import re
from typing import Protocol

from recall.config import Settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed passages; each vector is L2-normalized."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query; L2-normalized."""
        ...


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return v if norm == 0.0 else [x / norm for x in v]


class FastEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dim = 384
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._get_model().passage_embed(texts, batch_size=self.batch_size)
        return [_normalize([float(x) for x in v]) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        v = next(iter(self._get_model().query_embed(text)))
        return _normalize([float(x) for x in v])


class HashEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in re.split(r"\W+", text.lower()):
            if token:
                v[int(hashlib.sha1(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _normalize(v)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def get_embedder(settings: Settings, fake: bool = False) -> Embedder:
    if fake:
        return HashEmbedder(settings.embed_dim)
    e = FastEmbedder(settings.embed_model)
    e.dim = settings.embed_dim
    return e
