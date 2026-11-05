import pytest

from recall.ingest.loader import discover, load_file


def test_load_front_matter(tmp_path):
    p = tmp_path / "jd.md"
    p.write_text(
        "---\nsource: job_post\ntitle: SBE\ncompany: Littlebird\nsource_ref: https://x/1\nremote: true\n---\n## About\nWe build memory."
    )
    d = load_file(p)
    assert (d.source, d.title, d.company, d.source_ref, d.metadata) == (
        "job_post",
        "SBE",
        "Littlebird",
        "https://x/1",
        {"remote": True},
    )


def test_load_defaults_and_bad_bytes(tmp_path):
    p = tmp_path / "note.txt"
    p.write_bytes(b"caf\xe9 notes")
    d = load_file(p, root=tmp_path)
    assert d.source == "note" and d.title == "note" and "�" in d.body


def test_source_ref_relative_to_root(tmp_path):
    (tmp_path / "sub").mkdir()
    p = tmp_path / "sub" / "a.txt"
    p.write_text("x")
    assert load_file(p, root=tmp_path).source_ref == "sub/a.txt"
    assert load_file(p).source_ref == "a.txt"


def test_invalid_source_raises(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("---\nsource: tweet\n---\nhi")
    with pytest.raises(ValueError, match="x.md"):
        load_file(p)


def test_invalid_yaml_raises(tmp_path):
    p = tmp_path / "y.md"
    p.write_text("---\nsource: [unclosed\n---\nhi")
    with pytest.raises(ValueError, match="y.md"):
        load_file(p)


def test_discover_filters_and_sorts(tmp_path):
    for n in ["b.md", "a.txt", ".hidden.md", "c.pdf"]:
        (tmp_path / n).write_text("x")
    assert [p.name for p in discover(tmp_path)] == ["a.txt", "b.md"]
