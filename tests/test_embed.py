import math

import pytest

from recall.config import get_settings
from recall.ingest.embed import FastEmbedder, HashEmbedder, get_embedder


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_hash_embedder_shape_and_norm():
    v = HashEmbedder().embed(["postgres backend role"])[0]
    assert len(v) == 384 and math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-6)


def test_hash_embedder_deterministic_and_similarity():
    e = HashEmbedder()
    assert e.embed(["a b"]) == e.embed(["a b"])
    q = e.embed_query("postgres backend")
    assert cos(q, e.embed(["postgres backend job"])[0]) > cos(
        q, e.embed(["banana bread recipe"])[0]
    )


def test_hash_embedder_empty_text_is_zero_vector():
    assert HashEmbedder().embed([""])[0] == [0.0] * 384
    assert HashEmbedder().embed_query("  !!! ") == [0.0] * 384


def test_embedder_fixture_and_get_embedder_fake(embedder):
    assert isinstance(embedder, HashEmbedder)
    e = get_embedder(get_settings(), fake=True)
    assert isinstance(e, HashEmbedder) and e.dim == get_settings().embed_dim


def test_get_embedder_real_is_lazy():
    e = get_embedder(get_settings())
    assert isinstance(e, FastEmbedder) and e.dim == 384 and e._model is None


@pytest.mark.slow
def test_fastembedder_semantics():
    e = FastEmbedder()
    q = e.embed_query("database engineer job")
    near, far = e.embed(["Senior PostgreSQL backend engineer role", "Recipe for banana bread"])
    assert len(q) == 384 and cos(q, near) > cos(q, far)
    assert math.isclose(cos(q, q), 1.0, rel_tol=1e-4)
    assert isinstance(q, list) and isinstance(q[0], float)
