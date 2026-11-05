"""Eval runner: scores each retrieval mode on a labelled question set."""

import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import UUID

import psycopg
from pydantic import BaseModel, Field, ValidationError

from recall.evaluation.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from recall.ingest.embed import Embedder
from recall.models import Mode, SearchHit
from recall.retrieval.search import search

EVAL_USER_ID = UUID("00000000-0000-0000-0000-0000000000e1")

QuestionType = Literal["single_fact", "filter_semantic", "multi_hop"]
_TYPES: list[QuestionType] = ["single_fact", "filter_semantic", "multi_hop"]
_MODE_LABELS: dict[str, str] = {
    "vector": "Vector only",
    "keyword": "Keyword only",
    "hybrid": "Hybrid (RRF)",
}
_BLANK_ROWS = ["Hybrid + re-rank", "Hybrid + re-rank + graph", "Full (+ MMR, decay)"]


class Gold(BaseModel):
    doc: str
    contains: str


class Question(BaseModel):
    id: str
    type: QuestionType
    question: str
    gold: list[Gold] = Field(min_length=1)
    answer: str


class ModeResult(BaseModel):
    mode: Mode
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    p95_ms: float
    cost_per_query_usd: float
    by_type: dict[QuestionType, dict[str, float]]


def load_questions(path: Path) -> list[Question]:
    questions = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            questions.append(Question.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"{path}: line {n}: {e}") from e
    return questions


def _key(g: Gold) -> str:
    return f"{g.doc}::{g.contains}"


def _body(h: SearchHit) -> str:
    """Chunk text without the contextual "title — company" prefix paragraph."""
    if h.title and h.text.startswith(h.title) and "\n\n" in h.text:
        return h.text.split("\n\n", 1)[1]
    return h.text


def label_hits(hits: list[SearchHit], gold: list[Gold]) -> list[set[str]]:
    """For each hit, the set of gold keys it satisfies (matched against the chunk body)."""
    labels = []
    for h in hits:
        body = _body(h).lower()
        labels.append(
            {_key(g) for g in gold if h.source_ref == g.doc and g.contains.lower() in body}
        )
    return labels


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_eval(
    conn: psycopg.Connection,
    *,
    user_id: UUID,
    embedder: Embedder,
    questions: list[Question],
    modes: list[Mode],
    k: int = 10,
) -> list[ModeResult]:
    results = []
    for mode in modes:
        scores: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        latencies: list[float] = []
        if questions:  # untimed warm-up so connection/plan caches don't skew p95
            search(
                conn,
                user_id=user_id,
                query=questions[0].question,
                embedder=embedder,
                mode=mode,
                k=k,
            )
        for q in questions:
            t0 = time.perf_counter()
            hits = search(
                conn, user_id=user_id, query=q.question, embedder=embedder, mode=mode, k=k
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            retrieved = label_hits(hits, q.gold)
            relevant = {_key(g) for g in q.gold}
            row = (
                recall_at_k(retrieved, relevant, k),
                reciprocal_rank(retrieved, relevant),
                ndcg_at_k(retrieved, relevant, k),
            )
            scores["all"].append(row)
            scores[q.type].append(row)
        by_type: dict[QuestionType, dict[str, float]] = {
            t: {
                "recall_at_10": _mean([r[0] for r in scores[t]]),
                "mrr": _mean([r[1] for r in scores[t]]),
                "n": float(len(scores[t])),
            }
            for t in _TYPES
            if scores[t]
        }
        allrows = scores["all"]
        results.append(
            ModeResult(
                mode=mode,
                recall_at_10=_mean([r[0] for r in allrows]),
                mrr=_mean([r[1] for r in allrows]),
                ndcg_at_10=_mean([r[2] for r in allrows]),
                p95_ms=percentile(latencies, 95) if latencies else 0.0,
                cost_per_query_usd=0.0,
                by_type=by_type,
            )
        )
    return results


def render_results_md(
    results: list[ModeResult], *, corpus_docs: int, n_questions: int, embed_model: str
) -> str:
    by_mode = {r.mode: r for r in results}
    lines = [
        "# Recall eval results\n",
        (
            f"Date: {date.today().isoformat()} | Corpus: {corpus_docs} docs | "
            f"Questions: {n_questions} | Embedding model: {embed_model}\n"
        ),
        "| Configuration | Recall@10 | MRR | p95 latency (ms) | Cost per query |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode, label in _MODE_LABELS.items():
        r = by_mode.get(mode)  # type: ignore[call-overload]
        if r is None:
            lines.append(f"| {label} |  |  |  |  |")
        else:
            lines.append(
                f"| {label} | {r.recall_at_10:.3f} | {r.mrr:.3f} | {r.p95_ms:.1f} | "
                f"${r.cost_per_query_usd:.4f} |"
            )
    for label in _BLANK_ROWS:
        lines.append(f"| {label} |  |  |  |  |")
    lines += [
        "",
        "## nDCG@10",
        "",
        "| Configuration | nDCG@10 |",
        "| --- | --- |",
    ]
    for r in results:
        lines.append(f"| {_MODE_LABELS[r.mode]} | {r.ndcg_at_10:.3f} |")
    lines += [
        "",
        "## By question type",
        "",
        "| Configuration | Type | n | Recall@10 | MRR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        for t, m in r.by_type.items():
            lines.append(
                f"| {_MODE_LABELS[r.mode]} | {t} | {int(m['n'])} | "
                f"{m['recall_at_10']:.3f} | {m['mrr']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def write_results_md(
    results: list[ModeResult],
    path: Path,
    *,
    corpus_docs: int,
    n_questions: int,
    embed_model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_results_md(
            results, corpus_docs=corpus_docs, n_questions=n_questions, embed_model=embed_model
        )
    )
