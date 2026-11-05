"""Offline tests for the corpus generator and question drafter (no network, no API key)."""

import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
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
        prompt = kwargs["messages"][0]["content"]
        facts = json.loads(_tagged(prompt, "facts"))
        doc_type = _tagged(prompt, "document_type").replace("_", " ")
        # a per-call reference makes each doc's lines unique, like real varied prose
        ref = f"Ref {len(self.calls)} {doc_type}: "
        lines = [f"{ref}Notes about {facts['company']}."]
        if facts.get("role_title"):
            lines.append(f"{ref}We are hiring a {facts['role_title']} at {facts['company']}.")
            lines.append(f"{ref}Compensation is {facts['salary']}.")
        if facts.get("skills"):
            lines.append(f"{ref}Requirements include {', '.join(facts['skills'])}.")
        lines.append(f"{ref}{facts['remote_policy']}")
        for key, label in (("recruiter", "Recruiter"), ("interviewer", "Interviewer")):
            if facts.get(key):
                lines.append(f"{ref}{label} for this process is {facts[key]}.")
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


def _spec(source: str) -> DocSpec:
    return next(j for j in build_jobs(WORLD, 60) if j.source == source)


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
    spec = _spec("note")
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
    spec = _spec("job_post")
    client = FakeClient()
    render(spec, client)
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert "temperature" not in call
    prompt = call["messages"][0]["content"]
    assert json.loads(_tagged(prompt, "facts"))["company"] == spec.company
    system = call["system"].lower()
    assert "front matter" in system and "data" in system


def test_render_strips_model_front_matter_and_fences():
    spec = _spec("note")
    evil = "---\nsource: note\ncompany: Evil\n---\n\n## Real body\nHello."
    assert render(spec, FakeClient(evil)) == "## Real body\nHello."
    assert render(spec, FakeClient("```markdown\nHello there.\n```")) == "Hello there."


def test_generated_text_cannot_override_front_matter(tmp_path):
    spec = _spec("note")
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
    client = FakeClient()
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
            assert sum(g.contains.lower() in d.body.lower() for d in bodies.values()) == 1
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


# --- fix round 1 -------------------------------------------------------------------------


class ScriptedClient:
    """Returns queued bodies (or raises queued exceptions), one per call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _message(reply)


def _write_corpus_doc(root: Path, source: str, ref: str, body: str, **meta) -> None:
    front = yaml.safe_dump(
        {"source": source, "title": f"Title {ref}", "company": "Acme", "source_ref": ref, **meta}
    )
    path = root / source / f"{ref}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{front}---\n\n{body}\n", encoding="utf-8")


def _need(corpus: draft_questions._Corpus, ref: str, must: str) -> draft_questions.Need:
    return draft_questions.Need(next(d for d in corpus.docs if d.source_ref == ref), must)


def _phrase_reply(phrase: str) -> str:
    return json.dumps({"contains": phrase})


def test_phrase_in_two_sibling_docs_is_rejected(tmp_path):
    shared = "Compensation for this role is 30-40 LPA."
    _write_corpus_doc(tmp_path, "job_post", "jp-a", f"Intro A.\n\n{shared}")
    _write_corpus_doc(tmp_path, "job_post", "jp-b", f"Intro B.\n\n{shared}")
    corpus = draft_questions._Corpus(tmp_path)
    need = _need(corpus, "jp-a", "30-40 LPA")
    assert (
        draft_questions._phrase(ScriptedClient(*[_phrase_reply(shared)] * 2), need, corpus) is None
    )


def test_rejected_phrase_gets_one_more_attempt(tmp_path):
    shared = "Compensation for this role is 30-40 LPA."
    _write_corpus_doc(tmp_path, "job_post", "jp-a", f"Intro A, remote friendly.\n\n{shared}")
    _write_corpus_doc(tmp_path, "job_post", "jp-b", f"Intro B.\n\n{shared}")
    corpus = draft_questions._Corpus(tmp_path)
    client = ScriptedClient(_phrase_reply(shared), _phrase_reply("Intro A, remote friendly."))
    need = _need(corpus, "jp-a", "friendly")
    assert draft_questions._phrase(client, need, corpus) == "Intro A, remote friendly."
    assert "<rejected>" in client.calls[1]["messages"][0]["content"]


def test_phrase_must_match_indexed_text_not_raw_body(tmp_path):
    _write_corpus_doc(tmp_path, "job_post", "jp-a", "Pay is 30-40\nLPA for this role.")
    _write_corpus_doc(tmp_path, "job_post", "jp-b", "Pay is  50-60  LPA, double spaced.")
    corpus = draft_questions._Corpus(tmp_path)
    # newline survives normalize, so the one-line phrase is not in any chunk
    client = ScriptedClient(*[_phrase_reply("Pay is 30-40 LPA for this role.")] * 2)
    assert draft_questions._phrase(client, _need(corpus, "jp-a", "30-40"), corpus) is None
    # normalize collapses runs of spaces, so this phrase IS what the index holds
    ok = _phrase_reply("Pay is 50-60 LPA, double spaced.")
    got = draft_questions._phrase(ScriptedClient(ok), _need(corpus, "jp-b", "50-60"), corpus)
    assert got == "Pay is 50-60 LPA, double spaced."


@pytest.mark.parametrize(
    "body",
    [
        "Intro line.\n> Pay is 30-40 LPA for this role.\nMore text.",
        "Intro line.\n--\nPay is 30-40 LPA for this role.",
    ],
)
def test_phrase_dropped_by_normalize_is_rejected(tmp_path, body):
    _write_corpus_doc(tmp_path, "email", "em-a", body)
    corpus = draft_questions._Corpus(tmp_path)
    client = ScriptedClient(*[_phrase_reply("Pay is 30-40 LPA for this role.")] * 2)
    assert draft_questions._phrase(client, _need(corpus, "em-a", "30-40"), corpus) is None


def test_phrase_in_other_docs_quoted_text_does_not_count_as_duplicate(tmp_path):
    _write_corpus_doc(tmp_path, "email", "em-a", "Pay is 30-40 LPA for this role.")
    _write_corpus_doc(tmp_path, "email", "em-b", "Hello.\n> Pay is 30-40 LPA for this role.")
    corpus = draft_questions._Corpus(tmp_path)
    reply = _phrase_reply("Pay is 30-40 LPA for this role.")
    assert draft_questions._phrase(ScriptedClient(reply), _need(corpus, "em-a", "30-40"), corpus)


def test_phrase_unique_and_raw_matching_accepted(tmp_path):
    _write_corpus_doc(tmp_path, "job_post", "jp-a", "Pay is 30-40 LPA for this role.")
    _write_corpus_doc(tmp_path, "job_post", "jp-b", "Something else entirely.")
    corpus = draft_questions._Corpus(tmp_path)
    reply = _phrase_reply("pay is 30-40 LPA for this role.")  # case differs: eval is case-blind
    got = draft_questions._phrase(ScriptedClient(reply), _need(corpus, "jp-a", "30-40"), corpus)
    assert got == "pay is 30-40 LPA for this role."


def test_phrase_swallows_api_errors(tmp_path):
    _write_corpus_doc(tmp_path, "job_post", "jp-a", "Pay is 30-40 LPA for this role.")
    corpus = draft_questions._Corpus(tmp_path)
    boom = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    client = ScriptedClient(boom)
    assert draft_questions._phrase(client, _need(corpus, "jp-a", "30-40"), corpus) is None


def _mini_world() -> dict:
    roles = [
        {
            "id": f"r{i}",
            "company": "Acme",
            "title": f"Dev {i}",
            "skills": ["rust"],
            "status": "saved",
            "salary": f"{i}0-{i}5 LPA",
        }
        for i in (1, 2, 3)
    ]
    company = {
        "name": "Acme",
        "location": "Pune",
        "remote_from_india": True,
        "remote_policy": "Acme is remote friendly for India.",
        "industry": "x",
    }
    return {"companies": [company], "roles": roles, "people": [], "skills": ["rust"]}


def test_filter_semantic_skips_question_when_a_hit_role_has_no_post(tmp_path):
    world = _mini_world()
    for i in (1, 2):  # r3 has no job post in the corpus
        _write_corpus_doc(tmp_path, "job_post", f"jp-r{i}-0", f"Rust role {i}.", role_id=f"r{i}")
    corpus = draft_questions._Corpus(tmp_path)
    assert draft_questions._filter_semantic(world, corpus) == []
    _write_corpus_doc(tmp_path, "job_post", "jp-r3-0", "Rust role 3.", role_id="r3")
    corpus = draft_questions._Corpus(tmp_path)
    questions = [q for q, _, _ in draft_questions._filter_semantic(world, corpus)]
    assert "Which 'saved' roles require rust?" in questions


def test_filter_semantic_skips_question_when_a_hit_role_has_a_repost(tmp_path):
    world = _mini_world()
    for i in (1, 2, 3):
        _write_corpus_doc(tmp_path, "job_post", f"jp-r{i}-0", f"Rust role {i}.", role_id=f"r{i}")
    _write_corpus_doc(
        tmp_path,
        "job_post",
        "jp-r3-0-repost",
        "Rust role 3 again.",
        role_id="r3",
        repost_of="jp-r3-0",
    )
    corpus = draft_questions._Corpus(tmp_path)
    assert draft_questions._filter_semantic(world, corpus) == []


def test_render_retries_once_when_required_fact_missing():
    spec = _spec("job_post")
    good = f"Pay: {spec.facts['salary']}. {spec.facts['remote_policy']}"
    client = ScriptedClient(spec.facts["remote_policy"], good)
    assert render(spec, client) == good
    assert spec.facts["salary"] in client.calls[1]["messages"][0]["content"]
    assert len(client.calls) == 2


def test_render_raises_and_writes_nothing_when_fact_still_missing(tmp_path):
    spec = _spec("job_post")
    no_salary = spec.facts["remote_policy"]
    client = ScriptedClient(no_salary, no_salary)
    with pytest.raises(ValueError, match="missing required facts"):
        write_doc(spec, client, tmp_path)
    assert not list(tmp_path.rglob("*.md"))
    assert len(client.calls) == 2


def test_render_requires_person_names_case_insensitively():
    spec = _spec("email")
    facts = spec.facts
    body = f"{facts['salary']} {facts['remote_policy']} sincerely {facts['recruiter'].upper()}"
    assert render(spec, ScriptedClient(body)) == body
    with pytest.raises(ValueError, match="missing required facts"):
        render(spec, ScriptedClient("no facts here", "still none"))


@pytest.mark.parametrize("junk", ["> ", "--\n"])
def test_render_requires_facts_in_normalized_body(junk):
    spec = _spec("job_post")
    facts = spec.facts
    hidden = f"{junk}Pay: {facts['salary']}. {facts['remote_policy']}"
    good = f"Pay: {facts['salary']}. {facts['remote_policy']}"
    client = ScriptedClient(f"Intro.\n{hidden}", good)
    assert render(spec, client) == good
    assert len(client.calls) == 2
