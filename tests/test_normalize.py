from recall.ingest.normalize import content_hash, normalize


def test_html_block_and_inline():
    assert normalize("<p>Hello <b>world</b></p><p>Bye</p>") == "Hello world\nBye"


def test_script_and_style_removed():
    assert normalize("<div>Job<script>x=1</script><style>.a{}</style></div>") == "Job"


def test_signature_cut():
    assert normalize("Hi,\nThanks\n-- \nJane Doe\nRecruiter") == "Hi,\nThanks"


def test_quoted_reply_cut():
    raw = "Sounds good.\n\nOn Mon, 3 Nov 2026 at 10:00, Jane <j@x.com> wrote:\n> earlier text"
    assert normalize(raw) == "Sounds good."


def test_inline_quotes_dropped():
    assert normalize("Yes\n> old line\nThursday works") == "Yes\nThursday works"


def test_whitespace_collapsed():
    assert normalize("  a   b \n\n\n\n c ") == "a b\n\nc"


def test_html_only_whitespace_is_empty():
    assert normalize("<div>   </div>") == ""


def test_hash_ignores_wrapping():
    assert content_hash(normalize("<p>Same  text</p>")) == content_hash(normalize("Same text\n"))


def test_hash_is_sha256_hex():
    assert content_hash("a") == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"


def test_less_than_in_plain_text_is_not_html():
    assert normalize("salary < 50k, team > 5") == "salary < 50k, team > 5"
