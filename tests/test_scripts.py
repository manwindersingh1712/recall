"""Offline tests for the corpus generator and question drafter (no network, no API key)."""

import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from recall.evaluation.run import load_questions
from recall.ingest.loader import load_file
from scripts import draft_questions, generate_corpus
from scripts.draft_questions import draft
from scripts.generate_corpus import DocSpec, build_jobs, render, write_doc

WORLD = yaml.safe_load((Path(__file__).parent.parent / "eval" / "world.yaml").read_text())


def _message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
    )


def _tagged(prompt: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", prompt, re.DOTALL)
    assert m, f"prompt has no <{tag}> block"
    return m.group(1)


class FakeClient:
    """Mimics only `client.messages.create(...)`; returns a fixed body, or echoes the facts."""

    def __init__(self, body: str | None = None, stop_reason: str = "end_turn"):
        self.body = body
        self.stop_reason = stop_reason
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.body is not None:
            return _message(self.body, self.stop_reason)
        facts = json.loads(_tagged(kwargs["messages"][0]["content"], "facts"))
        lines = [f"Notes about {facts['company']}."]
        if facts.get("role_title"):
            lines.append(f"We are hiring a {facts['role_title']} at {facts['company']}.")
            lines.append(f"Compensation is {facts['salary']}.")
        lines.append(facts["remote_policy"])
        for key, label in (("recruiter", "Recruiter"), ("interviewer", "Interviewer")):
            if facts.get(key):
                lines.append(f"{label} for this process is {facts[key]}.")
        return _message("\n\n".join(lines), self.stop_reason)


class FakeQuestionClient:
    """Returns the document line containing the requested must_include text, as JSON."""

    def __init__(self, bad: bool = False):
        self.bad = bad
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        body, needle = _tagged(prompt, "document"), _tagged(prompt, "must_include")
        if self.bad:
            return _message(json.dumps({"contains": f"{needle} but not in the document"}))
        line = next((ln for ln in body.splitlines() if needle.lower() in ln.lower()), "")
        return _message(json.dumps({"contains": line.strip()}))


def _build_fixture_corpus() -> Path:
    root = Path(tempfile.mkdtemp(prefix="recall-fixture-corpus-"))
    client = FakeClient()
    for spec in build_jobs(WORLD, 300):
        write_doc(spec, client, root)
    return root


FIXTURE_CORPUS = _build_fixture_corpus()


def test_build_jobs_mix_and_determinism():
    world = yaml.safe_load(Path("eval/world.yaml").read_text())
    jobs = build_jobs(world, target_docs=200)
    assert len(jobs) == 200 and build_jobs(world, 200) == jobs
    share = Counter(j.source for j in jobs)
    assert 0.40 <= share["job_post"] / 200 <= 0.50


def test_build_jobs_refs_unique_and_reposts():
    jobs = build_jobs(WORLD, 400)
    refs = [j.source_ref for j in jobs]
    assert len(set(refs)) == len(refs)
    reposts = [j for j in jobs if j.facts.get("repost_of")]
    assert 0.03 <= len(reposts) / 400 <= 0.07
    originals = {j.source_ref: j for j in jobs}
    for r in reposts:
        orig = originals[r.facts["repost_of"]]
        assert (orig.company, orig.title) == (r.company, r.title)
    assert {j.source for j in jobs} == {"job_post", "email", "note", "interview"}


def test_world_supports_multi_hop():
    assert 35 <= len(WORLD["companies"]) <= 45 and len(WORLD["roles"]) == 60
    assert len(WORLD["people"]) >= 25
    assert {r["status"] for r in WORLD["roles"]} == {
        "saved",
        "applied",
        "interviewing",
        "rejected",
        "offer",
    }
    assert {c["remote_from_india"] for c in WORLD["companies"]} == {True, False}
    companies = {c["name"] for c in WORLD["companies"]}
    assert all(r["company"] in companies for r in WORLD["roles"])
    assert all(p["company"] in companies for p in WORLD["people"])
    assert any(p["role"] == "recruiter" for p in WORLD["people"])


def test_render_writes_loadable_front_matter(tmp_path):
    path = write_doc(build_jobs(WORLD, 1)[0], FakeClient("## About\nWe build things."), tmp_path)
    d = load_file(path)
    assert d.source in {"job_post", "email", "note", "interview"} and d.source_ref and d.body


def test_write_doc_layout_and_resumable(tmp_path):
    spec = build_jobs(WORLD, 20)[0]
    client = FakeClient("Some prose here.")
    path = write_doc(spec, client, tmp_path)
    assert path == tmp_path / spec.source / f"{spec.source_ref}.md"
    assert len(client.calls) == 1
    assert write_doc(spec, client, tmp_path) == path
    assert len(client.calls) == 1  # existing file: no second API call
    d = load_file(path)
    assert (d.source, d.source_ref, d.title, d.company) == (
        spec.source,
        spec.source_ref,
        spec.title,
        spec.company,
    )


def test_prompt_states_facts_as_data_and_body_only():
    spec = build_jobs(WORLD, 20)[0]
    client = FakeClient("Body.")
    render(spec, client)
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert "temperature" not in call
    prompt = call["messages"][0]["content"]
    assert json.loads(_tagged(prompt, "facts"))["company"] == spec.company
    system = call["system"].lower()
    assert "front matter" in system and "data" in system


def test_render_strips_model_front_matter_and_fences():
    spec = build_jobs(WORLD, 20)[0]
    evil = "---\nsource: note\ncompany: Evil\n---\n\n## Real body\nHello."
    assert render(spec, FakeClient(evil)) == "## Real body\nHello."
    assert render(spec, FakeClient("```markdown\nHello there.\n```")) == "Hello there."


def test_generated_text_cannot_override_front_matter(tmp_path):
    spec = build_jobs(WORLD, 20)[0]
    path = write_doc(
        spec, FakeClient("---\ncompany: Evil\nsource: email\n---\nBody text."), tmp_path
    )
    d = load_file(path)
    assert d.company == spec.company and d.source == spec.source and d.body == "Body text."


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_render_rejects_unusable_responses(stop_reason):
    with pytest.raises(RuntimeError, match=stop_reason):
        render(build_jobs(WORLD, 20)[0], FakeClient("partial", stop_reason))


def test_render_rejects_empty_body():
    with pytest.raises(ValueError, match="empty"):
        render(build_jobs(WORLD, 20)[0], FakeClient("   "))


def test_write_doc_rejects_unsafe_source_ref(tmp_path):
    bad = DocSpec("note", "../../etc/passwd", "t", "c", {})
    with pytest.raises(ValueError, match="source_ref"):
        write_doc(bad, FakeClient("x"), tmp_path)


def test_main_dry_run_makes_no_calls_and_no_files(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "corpus"
    generate_corpus.main(["--target", "50", "--out", str(out), "--dry-run"])
    text = capsys.readouterr().out
    assert "50" in text and "job_post" in text
    assert not out.exists()


def test_main_without_api_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        generate_corpus.main(["--target", "5", "--out", str(tmp_path)])
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_main_limit_writes_only_new_files(tmp_path, monkeypatch):
    client = FakeClient("Prose.")
    monkeypatch.setattr(generate_corpus, "_make_client", lambda: client)
    generate_corpus.main(["--target", "30", "--limit", "4", "--out", str(tmp_path)])
    assert len(list(tmp_path.rglob("*.md"))) == 4
    generate_corpus.main(["--target", "30", "--limit", "4", "--out", str(tmp_path)])
    assert len(list(tmp_path.rglob("*.md"))) == 8  # resumes with the next four
    assert len(client.calls) == 8


def test_draft_questions_schema(tmp_path):
    out = draft(
        WORLD, corpus_dir=FIXTURE_CORPUS, client=FakeQuestionClient(), out=tmp_path / "q.jsonl"
    )
    qs = load_questions(out)
    assert Counter(q.type for q in qs) == {
        "single_fact": 20,
        "filter_semantic": 15,
        "multi_hop": 15,
    }


def test_drafted_gold_phrases_are_distinctive_body_phrases(tmp_path):
    out = draft(
        WORLD, corpus_dir=FIXTURE_CORPUS, client=FakeQuestionClient(), out=tmp_path / "q.jsonl"
    )
    qs = load_questions(out)
    assert len({q.id for q in qs}) == 50
    bodies = {
        d.source_ref: d
        for d in (load_file(p, FIXTURE_CORPUS) for p in FIXTURE_CORPUS.rglob("*.md"))
    }
    for q in qs:
        assert q.question.strip() and q.answer.strip()
        if q.type == "multi_hop":
            assert len(q.gold) >= 2
        for g in q.gold:
            doc = bodies[g.doc]
            assert g.contains.lower() in doc.body.lower()
            assert len(g.contains) >= 8
            assert g.contains.lower() not in {doc.title.lower(), (doc.company or "").lower()}


def test_draft_rejects_phrases_not_in_body(tmp_path):
    with pytest.raises(ValueError, match="candidates"):
        draft(
            WORLD,
            corpus_dir=FIXTURE_CORPUS,
            client=FakeQuestionClient(bad=True),
            out=tmp_path / "q.jsonl",
        )


def test_draft_main_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        draft_questions.main(["--out", str(tmp_path / "q.jsonl")])
    assert "ANTHROPIC_API_KEY" in str(exc.value)
