from recall.ingest.chunk import chunk_document


def words(n, w="lorem"):
    return " ".join(f"{w}{i}" for i in range(n))


def test_small_sections_merge():
    text = f"## About\n{words(50, 'a')}\n## Requirements\n{words(50, 'r')}\n## Benefits\n{words(50, 'b')}"
    out = chunk_document(
        text, source="job_post", title="Senior Backend Engineer", company="Littlebird"
    )
    assert len(out) == 1 and "a0" in out[0].text and "b49" in out[0].text


def test_large_sections_split_with_prefix():
    text = f"## About\n{words(250, 'a')}\n## Requirements\n{words(250, 'r')}\n## Benefits\n{words(250, 'b')}"
    out = chunk_document(
        text, source="job_post", title="Senior Backend Engineer", company="Littlebird"
    )
    assert [c.ord for c in out] == [0, 1, 2]
    assert all(c.text.startswith("Senior Backend Engineer — Littlebird\n\n") for c in out)


def test_long_unstructured_text_windows():
    out = chunk_document(words(1000), source="note", title=None, company=None)
    bodies = [c.text.split() for c in out]
    assert all(len(b) <= 380 for b in bodies)
    assert bodies[0][-40:] == bodies[1][:40]
    assert bodies[-1][-1] == "lorem999"


def test_email_splits_by_message():
    text = f"From: Jane\n{words(200, 'x')}\nFrom: Me\n{words(200, 'y')}"
    assert len(chunk_document(text, source="email", title="Re: next round", company="MongoDB")) == 2


def test_empty_text():
    assert chunk_document("  \n", source="note", title="t", company=None) == []


def test_no_title_no_prefix():
    assert (
        chunk_document("hello there", source="note", title=None, company=None)[0].text
        == "hello there"
    )


def test_known_section_names_split_job_post():
    text = f"Requirements\n{words(250, 'r')}\nBenefits:\n{words(250, 'b')}"
    assert len(chunk_document(text, source="job_post", title="T", company=None)) == 2


def test_title_only_prefix():
    out = chunk_document("hello", source="note", title="T", company=None)
    assert out[0].text == "T\n\nhello"
