from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from recall.api.app import create_app
from recall.config import Settings
from recall.ingest.embed import HashEmbedder


@pytest.fixture
def client(db_url):
    app = create_app(Settings(database_url=db_url), embedder=HashEmbedder())
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "db": True}


def test_search_returns_hits(client, seeded, user_a):
    r = client.post(
        "/search", json={"query": "pgvector", "user_id": str(user_a), "mode": "keyword"}
    )
    assert (
        r.status_code == 200
        and "Littlebird" in r.json()["hits"][0]["text"]
        and r.json()["took_ms"] >= 0
    )


def test_search_no_data_returns_empty(client):
    r = client.post("/search", json={"query": "anything", "user_id": str(uuid4())})
    assert r.status_code == 200 and r.json()["hits"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "x", "k": 0},
        {"query": "x", "k": 51},
        {"query": "x", "mode": "graph"},
    ],
)
def test_search_validation(client, user_a, body):
    assert client.post("/search", json={"user_id": str(user_a), **body}).status_code == 422
