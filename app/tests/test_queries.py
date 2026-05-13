"""Tests for database query functions."""
import pytest
from app.db.queries import (
    get_article,
    get_article_summary,
    get_comment,
    get_connection,
    initialize_schema,
    list_articles,
    list_comments,
    serialize_article_summary,
    serialize_comment,
    upsert_article,
    update_thresholds,
)


SAMPLE_ARTICLE = {
    "id": "test_001",
    "title": "Test Article",
    "url": "https://en.wikipedia.org/wiki/Test_Article",
    "summary": "A test article summary.",
    "created_at": "2024-01-01T00:00:00",
    "model_name": "test_model",
    "auto_threshold": 0.5,
    "manual_threshold": 0.7,
    "flagged_count": 2,
    "trend": [],
    "inference_stats": {},
}

SAMPLE_COMMENTS = [
    {
        "id": "c1",
        "author": "user1",
        "timestamp": "2024-01-01T00:01:00",
        "text": "This is fine.",
        "toxicity": 0.1,
        "decision": "approved",
        "is_flagged": False,
        "top_features": None,
        "model_version": "v1",
        "explain_version": "",
        "inference_ms": 0.0,
    },
    {
        "id": "c2",
        "author": "user2",
        "timestamp": "2024-01-01T00:02:00",
        "text": "This is toxic.",
        "toxicity": 0.9,
        "decision": "flagged",
        "is_flagged": True,
        "top_features": None,
        "model_version": "v1",
        "explain_version": "",
        "inference_ms": 0.0,
    },
]


@pytest.fixture(autouse=True)
def clean_db(app):
    """Ensure schema is initialized and DB is clean before each test."""
    with app.app_context():
        initialize_schema()
        conn = get_connection()
        conn.execute("DELETE FROM articles")
        conn.execute("DELETE FROM comments")
        conn.commit()
        conn.close()
    yield


class TestInitializeSchema:
    def test_creates_tables(self, app):
        with app.app_context():
            initialize_schema()
            conn = get_connection()
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            conn.close()
            assert "articles" in tables
            assert "comments" in tables

    def test_idempotent(self, app):
        with app.app_context():
            initialize_schema()
            initialize_schema()  # should not raise


class TestUpsertAndListArticles:
    def test_upsert_inserts_article(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            articles = list_articles()
            assert len(articles) == 1
            assert articles[0]["id"] == "test_001"

    def test_upsert_updates_existing_article(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            updated = {**SAMPLE_ARTICLE, "title": "Updated Title"}
            upsert_article(updated, SAMPLE_COMMENTS)
            articles = list_articles()
            assert len(articles) == 1
            assert articles[0]["title"] == "Updated Title"

    def test_list_articles_empty(self, app):
        with app.app_context():
            assert list_articles() == []


class TestGetArticle:
    def test_get_existing_article(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            result = get_article("test_001")
            assert result != {}
            assert result["id"] == "test_001"

    def test_get_nonexistent_article_returns_empty(self, app):
        with app.app_context():
            result = get_article("does_not_exist")
            assert result == {}

    def test_get_article_includes_comments_by_default(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            result = get_article("test_001")
            assert "comments" in result
            assert len(result["comments"]) == 2


class TestGetArticleSummary:
    def test_get_summary_existing(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            result = get_article_summary("test_001")
            assert result is not None
            assert result["id"] == "test_001"

    def test_get_summary_nonexistent_returns_empty(self, app):
        with app.app_context():
            result = get_article_summary("does_not_exist")
            assert result == {}


class TestListComments:
    def test_list_comments_for_article(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            comments, total = list_comments("test_001")
            assert len(comments) == 2
            assert total == 2

    def test_list_comments_filter_by_decision(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            flagged, total = list_comments("test_001", decision="flagged")
            assert all(c["decision"] == "flagged" for c in flagged)
            assert len(flagged) == 1

    def test_list_comments_empty_article(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, [])
            comments, total = list_comments("test_001")
            assert comments == []
            assert total == 0


class TestGetComment:
    def test_get_existing_comment(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            result = get_comment("test_001", "c1")
            assert result != {}
            assert result["comment"]["id"] == "c1"
            assert result["article"]["id"] == "test_001"

    def test_get_nonexistent_comment_returns_empty(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            result = get_comment("test_001", "does_not_exist")
            assert result == {}


class TestUpdateThresholds:
    def test_update_thresholds(self, app):
        with app.app_context():
            upsert_article(SAMPLE_ARTICLE, SAMPLE_COMMENTS)
            update_thresholds("test_001", 0.3, 0.8)
            result = get_article_summary("test_001")
            assert result["auto_threshold"] == 0.3
            assert result["manual_threshold"] == 0.8
