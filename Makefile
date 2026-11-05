.PHONY: up down test test-all lint

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
