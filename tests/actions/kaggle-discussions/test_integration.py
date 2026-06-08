"""Integration tests for kaggle-discussions (hit real Kaggle API).

Run with: uv run pytest tests/actions/kaggle-discussions/test_integration.py --run-integration -v
"""

import os

import pytest

from discussions.database import DiscussionDatabase
from discussions.discussion_client import DiscussionClient
from discussions.models import DiscussionRecord

COMPETITION = "titanic"

# Competitions where slug != title (slug-to-title resolution required)
SLUG_TITLE_CASES = [
    ("playground-series-s6e5", "Predicting F1 Pit Stops"),
    ("birdclef-2026", "BirdCLEF+ 2026"),
]


@pytest.fixture(scope="module")
def client():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        pytest.skip("KAGGLE_API_TOKEN not set")
    with DiscussionClient() as c:
        yield c


@pytest.fixture(scope="module")
def discussions(client):
    """Fetch a small batch of discussions once for the module."""
    results = client.list_discussions(COMPETITION, max_pages=1, page_size=5)
    if not results:
        pytest.skip(f"No discussions found for {COMPETITION}")
    return results


@pytest.mark.integration
class TestListDiscussions:
    def test_returns_records(self, discussions):
        assert len(discussions) > 0
        assert all(isinstance(d, DiscussionRecord) for d in discussions)

    def test_fields_populated(self, discussions):
        d = discussions[0]
        assert d.competition_id == COMPETITION
        assert d.discussion_id > 0
        assert len(d.title) > 0
        assert len(d.author) > 0


@pytest.mark.integration
class TestCommentFetching:
    def test_get_discussion_detail(self, client, discussions):
        disc_id = discussions[0].discussion_id
        total, comments, first_md, op_author = client.get_discussion_detail(disc_id)
        assert isinstance(total, int)
        assert isinstance(comments, list)


@pytest.mark.integration
class TestSlugToTitleResolution:
    @pytest.mark.parametrize("slug,expected_title", SLUG_TITLE_CASES)
    def test_resolve_competition_title(self, client, slug, expected_title):
        title = client._resolve_competition_title(slug)
        assert title == expected_title

    @pytest.mark.parametrize("slug,expected_title", SLUG_TITLE_CASES)
    def test_list_discussions_with_slug(self, client, slug, expected_title):
        """Slugs where slug != title should still return discussions."""
        results = client.list_discussions(slug, max_pages=1, page_size=5)
        assert len(results) > 0, f"Expected discussions for {slug} ({expected_title})"
        assert all(isinstance(d, DiscussionRecord) for d in results)

    def test_nonexistent_slug_returns_none(self, client):
        assert client._resolve_competition_title("this-competition-does-not-exist-99999") is None


@pytest.mark.integration
class TestCompetitionInfo:
    def test_get_competition_info(self, client):
        info = client.get_competition_info(COMPETITION)
        assert info.competition_id == COMPETITION
        assert len(info.title) > 0


@pytest.mark.integration
class TestFullPipeline:
    def test_ingest_query_roundtrip(self, client, discussions, tmp_path):
        db_path = tmp_path / "integration.db"
        with DiscussionDatabase(db_path) as db:
            inserted, updated = db.upsert_discussions(discussions)
            assert inserted == len(discussions)

            results = db.query_discussions(COMPETITION)
            assert len(results) == len(discussions)

            disc = db.get_discussion(COMPETITION, discussions[0].discussion_id)
            assert disc is not None
            assert disc.title == discussions[0].title
