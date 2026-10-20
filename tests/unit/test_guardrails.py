import pytest

from careerpilot.core.chunking import chunk_document
from careerpilot.integrations.sql_db.postgres_client import ReadOnlyViolation, validate_readonly


def test_validate_readonly_allows_select():
    assert validate_readonly("SELECT * FROM applications;").lower().startswith("select")


def test_validate_readonly_allows_cte():
    assert validate_readonly("WITH t AS (SELECT 1) SELECT * FROM t")


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM applications",
        "UPDATE applications SET status='Offer'",
        "INSERT INTO applications VALUES (1)",
        "DROP TABLE applications",
        "SELECT 1; DROP TABLE applications",
    ],
)
def test_validate_readonly_blocks_writes(query):
    with pytest.raises(ReadOnlyViolation):
        validate_readonly(query)


def test_chunking_overlap_and_size():
    text = "\n\n".join(f"Paragraph {i} with some filler text." for i in range(20))
    chunks = chunk_document(text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_document("x", chunk_size=50, chunk_overlap=50)
