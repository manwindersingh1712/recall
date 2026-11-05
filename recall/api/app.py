"""FastAPI app exposing /health and /search."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field

from recall.config import Settings, get_settings
from recall.db import make_pool
from recall.ingest.embed import Embedder, get_embedder
from recall.models import Mode, SearchHit
from recall.retrieval.search import search

log = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_id: UUID
    mode: Mode = "hybrid"
    k: int = Field(default=10, ge=1, le=50)


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    took_ms: float


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = make_pool(settings.database_url)
        app.state.pool = pool
        try:
            yield
        finally:
            pool.close()

    app = FastAPI(title="Recall", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request, response: Response) -> dict[str, object]:
        try:
            with request.app.state.pool.connection() as conn:
                conn.execute("SELECT 1")
        except Exception:
            log.exception("health check failed")
            response.status_code = 503
            return {"status": "error", "db": False}
        return {"status": "ok", "db": True}

    @app.post("/search")
    def search_endpoint(body: SearchRequest, request: Request) -> SearchResponse:
        start = time.perf_counter()
        with request.app.state.pool.connection() as conn:
            hits = search(
                conn,
                user_id=body.user_id,
                query=body.query,
                embedder=embedder,
                mode=body.mode,
                k=body.k,
            )
        return SearchResponse(hits=hits, took_ms=(time.perf_counter() - start) * 1000)

    return app
