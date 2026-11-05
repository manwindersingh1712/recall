.PHONY: up down test test-all lint migrate ingest search api

up:
	docker compose up -d --wait

down:
	docker compose down

test:
	uv run pytest -m "not slow"

test-all:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

migrate:
	uv run recall migrate

ingest:
	uv run recall ingest $(or $(path),tests/fixtures/docs)

search:
	uv run recall search "$(q)"

api:
	uv run uvicorn recall.api.app:create_app --factory --reload --port 8000
