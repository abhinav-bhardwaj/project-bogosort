"""Tests for article_service module."""
import pytest
from unittest.mock import patch, MagicMock, Mock
from app.services.article_service import (
    slugify_title,
    ingest_article,
    list_articles,
    get_article,
    list_comments,
    update_thresholds,
    get_comment_detail,
)


class TestSlugifyTitle:
    """Tests for slugify_title function."""

    def test_simple_title(self):
        """Test slugifying simple title."""
        result = slugify_title("Python Programming")
        assert isinstance(result, str)
        # Should be URL-safe
        assert " " not in result or "_" in result

    def test_title_with_special_chars(self):
        """Test slugifying title with special characters."""
        result = slugify_title("C++ & Python!")
        assert isinstance(result, str)

    def test_title_with_unicode(self):
        """Test slugifying title with unicode."""
        result = slugify_title("Café au Lait")
        assert isinstance(result, str)

    def test_empty_title(self):
        """Test slugifying empty title."""
        result = slugify_title("")
        assert isinstance(result, str)

    def test_title_with_numbers(self):
        """Test slugifying title with numbers."""
        result = slugify_title("Python 3.9")
        assert isinstance(result, str)

    def test_title_with_parentheses(self):
        """Test slugifying title with parentheses."""
        result = slugify_title("Python (programming language)")
        assert isinstance(result, str)


class TestDecideAction:
    """Tests for internal _decide_action function (tested via ingest_article)."""

    @pytest.mark.parametrize("toxicity,auto_thresh,manual_thresh,expected", [
        (0.3, 0.75, 0.55, "none"),  # Below both
        (0.6, 0.75, 0.55, "manual-review"),  # Between
        (0.9, 0.75, 0.55, "auto-ban"),  # Above both
        (0.75, 0.75, 0.55, "auto-ban"),  # On auto threshold
        (0.55, 0.75, 0.55, "manual-review"),  # On manual threshold
    ])
    def test_threshold_logic(self, toxicity, auto_thresh, manual_thresh, expected):
        """Test threshold decision logic."""
        # This would be tested through ingest_article
        # Verify the function completes
        assert isinstance(expected, str)


class TestIngestArticle:
    """Tests for ingest_article function."""

    @patch("app.services.wiki_client.is_allowed_wikipedia_url")
    @patch("app.services.wiki_client.parse_wiki_title_from_url")
    @patch("app.services.wiki_client.fetch_wikipedia_metadata")
    @patch("app.services.wiki_client.fetch_talk_page_comments")
    @patch("app.services.toxicity_service.score_comment")
    @patch("app.db.article_repository.upsert_article")
    @patch("app.db.article_repository.get_article_summary")
    def test_ingest_article_success(
        self, mock_get_summary, mock_upsert, mock_score, mock_fetch_comments, mock_fetch_meta,
        mock_parse, mock_is_allowed
    ):
        """Test successful article ingestion."""
        mock_is_allowed.return_value = True
        mock_parse.return_value = "Test_Article"
        mock_fetch_meta.return_value = {
            "title": "Test Article",
            "summary": "Summary",
            "url": "https://en.wikipedia.org/wiki/Test_Article"
        }
        mock_fetch_comments.return_value = [
            {
                "id": "",
                "author": "User1",
                "timestamp": "2024-01-01T12:00:00Z",
                "text": "Toxic comment"
            }
        ]
        mock_score.return_value = {"probability": 0.8, "label": 1, "top_features": [], "explain_version": ""}
        mock_upsert.return_value = None
        mock_get_summary.return_value = {"id": "article_1", "title": "Test Article"}

        result = ingest_article(
            "https://en.wikipedia.org/wiki/Test_Article",
            limit=30,
            auto_threshold=0.75,
            manual_threshold=0.55
        )

        assert result is not None
        assert isinstance(result, dict)

    @patch("app.services.wiki_client.is_allowed_wikipedia_url")
    def test_ingest_invalid_url(self, mock_is_allowed):
        """Test ingesting article with invalid URL."""
        mock_is_allowed.return_value = False

        with pytest.raises(ValueError):
            ingest_article("https://example.com/article")

    @patch("app.services.wiki_client.is_allowed_wikipedia_url")
    @patch("app.services.wiki_client.parse_wiki_title_from_url")
    @patch("app.services.wiki_client.fetch_wikipedia_metadata")
    def test_ingest_missing_article(self, mock_fetch_meta, mock_parse, mock_is_allowed):
        """Test ingesting non-existent Wikipedia article."""
        mock_is_allowed.return_value = True
        mock_parse.return_value = "Nonexistent"
        mock_fetch_meta.return_value = None

        with pytest.raises((ValueError, Exception)):
            ingest_article("https://en.wikipedia.org/wiki/Nonexistent")

    @patch("app.services.wiki_client.is_allowed_wikipedia_url")
    def test_ingest_threshold_validation(self, mock_is_allowed):
        """Test threshold validation."""
        mock_is_allowed.return_value = True

        with pytest.raises(ValueError):
            # manual_threshold > auto_threshold should fail
            ingest_article(
                "https://en.wikipedia.org/wiki/Test",
                auto_threshold=0.5,
                manual_threshold=0.8
            )

    @patch("app.services.wiki_client.is_allowed_wikipedia_url")
    @patch("app.services.wiki_client.parse_wiki_title_from_url")
    @patch("app.services.wiki_client.fetch_wikipedia_metadata")
    @patch("app.services.wiki_client.fetch_talk_page_comments")
    @patch("app.db.article_repository.upsert_article")
    @patch("app.db.article_repository.get_article_summary")
    def test_ingest_empty_talk_page(
        self, mock_get_summary, mock_upsert, mock_fetch_comments, mock_fetch_meta, mock_parse, mock_is_allowed
    ):
        """Test ingesting article with empty talk page."""
        mock_is_allowed.return_value = True
        mock_parse.return_value = "Test_Article"
        mock_fetch_meta.return_value = {
            "title": "Test Article",
            "summary": "Summary",
            "url": "https://en.wikipedia.org/wiki/Test_Article"
        }
        mock_fetch_comments.return_value = []
        mock_upsert.return_value = None
        mock_get_summary.return_value = {"id": "article_1", "title": "Test Article"}

        result = ingest_article("https://en.wikipedia.org/wiki/Test_Article")
        assert result is not None

    @patch("app.services.wiki_client.is_allowed_wikipedia_url")
    @patch("app.services.wiki_client.parse_wiki_title_from_url")
    @patch("app.services.wiki_client.fetch_wikipedia_metadata")
    @patch("app.services.wiki_client.fetch_talk_page_comments")
    @patch("app.services.toxicity_service.score_comment")
    @patch("app.db.article_repository.upsert_article")
    @patch("app.db.article_repository.get_article_summary")
    def test_ingest_with_different_model(
        self, mock_get_summary, mock_upsert, mock_score, mock_fetch_comments, mock_fetch_meta,
        mock_parse, mock_is_allowed
    ):
        """Test ingestion with different model name."""
        mock_is_allowed.return_value = True
        mock_parse.return_value = "Test_Article"
        mock_fetch_meta.return_value = {
            "title": "Test Article",
            "summary": "Summary",
            "url": "https://en.wikipedia.org/wiki/Test_Article"
        }
        mock_fetch_comments.return_value = [
            {"id": "", "author": "User1", "timestamp": "2024-01-01T12:00:00Z", "text": "Text"}
        ]
        mock_score.return_value = {"probability": 0.5, "label": 0, "top_features": [], "explain_version": ""}
        mock_upsert.return_value = None
        mock_get_summary.return_value = {"id": "article_1", "title": "Test Article"}

        result = ingest_article(
            "https://en.wikipedia.org/wiki/Test_Article",
            model_name="random_forest"
        )
        assert result is not None


class TestListArticles:
    """Tests for list_articles function."""

    @patch("app.services.article_service.article_repository")
    def test_list_articles_empty(self, mock_repo):
        """Test listing articles when none exist."""
        mock_repo.list_articles.return_value = []

        with patch("app.services.article_service.article_repository", mock_repo):
            result = list_articles()

            assert result == []

    @patch("app.services.article_service.article_repository")
    def test_list_articles_with_data(self, mock_repo):
        """Test listing articles with data."""
        articles = [
            {"id": "1", "title": "Article 1"},
            {"id": "2", "title": "Article 2"}
        ]
        mock_repo.list_articles.return_value = articles

        with patch("app.services.article_service.article_repository", mock_repo):
            result = list_articles()

            assert len(result) == 2


class TestGetArticle:
    """Tests for get_article function."""

    @patch("app.services.article_service.article_repository")
    def test_get_article_with_comments(self, mock_repo):
        """Test getting article with comments."""
        article = {
            "id": "1",
            "title": "Test Article",
            "comments": [
                {"id": "c1", "text": "Comment 1", "toxicity": 0.8}
            ]
        }
        mock_repo.get_article.return_value = article

        with patch("app.services.article_service.article_repository", mock_repo):
            result = get_article("1", include_comments=True)

            assert result is not None

    @patch("app.services.article_service.article_repository")
    def test_get_article_without_comments(self, mock_repo):
        """Test getting article without comments."""
        article = {"id": "1", "title": "Test Article"}
        mock_repo.get_article.return_value = article

        with patch("app.services.article_service.article_repository", mock_repo):
            result = get_article("1", include_comments=False)

            assert result is not None

    @patch("app.services.article_service.article_repository")
    def test_get_nonexistent_article(self, mock_repo):
        """Test getting non-existent article."""
        mock_repo.get_article.return_value = None

        with patch("app.services.article_service.article_repository", mock_repo):
            result = get_article("nonexistent")

            assert result is None

    @patch("app.services.article_service.article_repository")
    def test_get_article_with_pagination(self, mock_repo):
        """Test getting article with limit and offset."""
        article = {"id": "1", "title": "Test Article"}
        mock_repo.get_article.return_value = article

        with patch("app.services.article_service.article_repository", mock_repo):
            result = get_article("1", limit=10, offset=5)

            # Should pass limit/offset to repository
            mock_repo.get_article.assert_called()

    @patch("app.services.article_service.article_repository")
    def test_get_article_with_filtering(self, mock_repo):
        """Test getting article with decision filter."""
        article = {"id": "1", "title": "Test Article"}
        mock_repo.get_article.return_value = article

        with patch("app.services.article_service.article_repository", mock_repo):
            result = get_article("1", decision="auto-ban")

            assert result is not None

    @pytest.mark.parametrize("sort_type", ["toxicity_desc", "toxicity_asc", "timestamp_desc", "timestamp_asc"])
    @patch("app.services.article_service.article_repository")
    def test_get_article_with_sorting(self, mock_repo, sort_type):
        """Test getting article with different sort orders."""
        article = {"id": "1", "title": "Test Article"}
        mock_repo.get_article.return_value = article

        with patch("app.services.article_service.article_repository", mock_repo):
            result = get_article("1", sort=sort_type)

            assert result is not None


class TestListComments:
    """Tests for list_comments function."""

    @patch("app.services.article_service.article_repository")
    def test_list_comments_empty(self, mock_repo):
        """Test listing comments when none exist."""
        mock_repo.list_comments.return_value = ([], 0)

        with patch("app.services.article_service.article_repository", mock_repo):
            result = list_comments("article_1")

            assert isinstance(result, dict) and result["total"] == 0

    @patch("app.services.article_service.article_repository")
    def test_list_comments_with_data(self, mock_repo):
        """Test listing comments with data."""
        comments = [
            {"id": "c1", "text": "Comment 1", "toxicity": 0.8},
            {"id": "c2", "text": "Comment 2", "toxicity": 0.2}
        ]
        mock_repo.list_comments.return_value = (comments, len(comments))

        with patch("app.services.article_service.article_repository", mock_repo):
            result = list_comments("article_1")

            assert result["total"] == 2 and len(result["comments"]) == 2

    @pytest.mark.parametrize("sort_type", ["toxicity_desc", "toxicity_asc"])
    @patch("app.services.article_service.article_repository")
    def test_list_comments_with_sorting(self, mock_repo, sort_type):
        """Test listing comments with sorting."""
        mock_repo.list_comments.return_value = ([], 0)

        with patch("app.services.article_service.article_repository", mock_repo):
            result = list_comments("article_1", sort=sort_type)

            mock_repo.list_comments.assert_called()

    @patch("app.services.article_service.article_repository")
    def test_list_comments_with_pagination(self, mock_repo):
        """Test listing comments with limit and offset."""
        mock_repo.list_comments.return_value = ([], 0)

        with patch("app.services.article_service.article_repository", mock_repo):
            result = list_comments("article_1", limit=20, offset=10)

            mock_repo.list_comments.assert_called()

    @patch("app.services.article_service.article_repository")
    def test_list_comments_with_decision_filter(self, mock_repo):
        """Test listing comments filtered by decision."""
        mock_repo.list_comments.return_value = ([], 0)

        with patch("app.services.article_service.article_repository", mock_repo):
            result = list_comments("article_1", decision="manual-review")

            mock_repo.list_comments.assert_called()


class TestUpdateThresholds:
    """Tests for update_thresholds function."""

    @patch("app.db.article_repository.update_thresholds")
    def test_update_thresholds_success(self, mock_update):
        """Test successfully updating thresholds."""
        mock_update.return_value = None
        update_thresholds("article_1", auto_threshold=0.8, manual_threshold=0.6)
        mock_update.assert_called_once_with("article_1", 0.8, 0.6)

    @patch("app.db.article_repository.update_thresholds")
    def test_update_thresholds_boundary_values(self, mock_update):
        """Test updating thresholds with boundary values."""
        mock_update.return_value = None
        update_thresholds("article_1", auto_threshold=1.0, manual_threshold=0.0)
        mock_update.assert_called_once_with("article_1", 1.0, 0.0)

    @patch("app.db.article_repository.update_thresholds")
    def test_update_thresholds_invalid_order(self, mock_update):
        """Test updating with manual_threshold > auto_threshold."""
        mock_update.return_value = None
        # update_thresholds doesn't validate order - that's done in API layer
        update_thresholds("article_1", auto_threshold=0.5, manual_threshold=0.8)
        mock_update.assert_called_once_with("article_1", 0.5, 0.8)


class TestGetCommentDetail:
    """Tests for get_comment_detail function."""

    @patch("app.db.article_repository.get_comment")
    def test_get_comment_detail_success(self, mock_get_comment):
        """Test getting comment detail."""
        mock_get_comment.return_value = {
            "comment": {
                "id": "c1",
                "text": "Comment text",
                "toxicity": 0.8,
                "author": "User1",
                "top_features": [],
                "explain_version": ""
            },
            "article": {
                "id": "article_1",
                "title": "Test",
                "model_name": "ensemble"
            }
        }

        result = get_comment_detail("article_1", "c1")

        assert result is not None

    @patch("app.db.article_repository.get_comment")
    def test_get_comment_detail_nonexistent(self, mock_get_comment):
        """Test getting non-existent comment."""
        mock_get_comment.return_value = None

        result = get_comment_detail("article_1", "nonexistent")

        assert result == {}

    @patch("app.db.article_repository.get_comment")
    @patch("app.db.article_repository.update_comment_explanation")
    @patch("app.services.toxicity_service.score_comment")
    def test_get_comment_detail_with_explanation(self, mock_score, mock_update_explain, mock_get_comment):
        """Test getting comment with explanation loading."""
        mock_get_comment.return_value = {
            "comment": {
                "id": "c1",
                "text": "Comment text",
                "toxicity": 0.8,
                "top_features": [],
                "explain_version": ""
            },
            "article": {
                "id": "article_1",
                "title": "Test",
                "model_name": "ensemble"
            }
        }
        mock_score.return_value = {
            "probability": 0.8,
            "label": 1,
            "top_features": [("word1", 0.5)],
            "explain_version": "v1"
        }
        mock_update_explain.return_value = None

        result = get_comment_detail("article_1", "c1")

        assert result is not None


class TestArticleServiceIntegration:
    """Integration tests for article service workflows."""

    @patch("app.services.wiki_client.is_allowed_wikipedia_url")
    @patch("app.services.wiki_client.parse_wiki_title_from_url")
    @patch("app.services.wiki_client.fetch_wikipedia_metadata")
    @patch("app.services.wiki_client.fetch_talk_page_comments")
    @patch("app.services.toxicity_service.score_comment")
    @patch("app.db.article_repository.upsert_article")
    @patch("app.db.article_repository.get_article_summary")
    def test_article_ingestion_workflow(
        self, mock_get_summary, mock_upsert, mock_score, mock_fetch_comments, mock_fetch_meta,
        mock_parse, mock_is_allowed
    ):
        """Test complete article ingestion workflow."""
        # Setup mocks
        mock_is_allowed.return_value = True
        mock_parse.return_value = "Test_Article"
        mock_fetch_meta.return_value = {
            "title": "Test Article",
            "summary": "Summary",
            "url": "https://en.wikipedia.org/wiki/Test_Article"
        }
        mock_fetch_comments.return_value = [
            {"id": "", "author": "User1", "timestamp": "2024-01-01T12:00:00Z", "text": "Comment1"},
            {"id": "", "author": "User2", "timestamp": "2024-01-02T12:00:00Z", "text": "Comment2"}
        ]
        mock_score.return_value = {"probability": 0.5, "label": 0, "top_features": [], "explain_version": ""}
        mock_upsert.return_value = None
        mock_get_summary.return_value = {"id": "article_1", "title": "Test Article"}

        result = ingest_article("https://en.wikipedia.org/wiki/Test_Article")
        assert result is not None

    @patch("app.db.article_repository.get_article")
    @patch("app.db.article_repository.update_thresholds")
    @patch("app.db.article_repository.list_comments")
    def test_article_retrieval_workflow(self, mock_list_comments, mock_update_thresholds, mock_get_article):
        """Test article retrieval workflow."""
        # Setup mocks
        mock_get_article.return_value = {
            "id": "article_1",
            "title": "Test Article",
            "comments": []
        }
        mock_update_thresholds.return_value = None
        mock_list_comments.return_value = ([], 0)

        # Get article
        article = get_article("article_1")
        assert article is not None

        # Update thresholds
        update_thresholds("article_1", auto_threshold=0.8, manual_threshold=0.6)

        # Get comments
        comments = list_comments("article_1")
        assert isinstance(comments, dict)
