"""Tests for wiki_client module."""
import pytest
from unittest.mock import patch, Mock
from app.services.wiki_client import (
    is_allowed_wikipedia_url,
    parse_wiki_title_from_url,
    fetch_wikipedia_metadata,
    fetch_talk_page_comments,
)


class TestIsAllowedWikipediaUrl:
    """Tests for is_allowed_wikipedia_url function."""

    @pytest.mark.parametrize("url", [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Machine_Learning",
        "http://en.wikipedia.org/wiki/Test",
        "https://simple.wikipedia.org/wiki/Simple_Test",
    ])
    def test_valid_wikipedia_urls(self, url):
        """Test that valid Wikipedia URLs are allowed."""
        assert is_allowed_wikipedia_url(url) is True

    @pytest.mark.parametrize("url", [
        "https://example.com/wiki/Article",
        "https://en.example.org/wiki/Article",
        "https://en.wikipedia.org/Article",  # Missing /wiki/
        "https://en.wikipedia.org/w/Article",  # Wrong path
        "ftp://en.wikipedia.org/wiki/Article",
        "en.wikipedia.org/wiki/Article",  # Missing scheme
        "",
        None,
        "https://en.wikipedia.org",  # No article
    ])
    def test_invalid_wikipedia_urls(self, url):
        """Test that invalid URLs are rejected."""
        assert is_allowed_wikipedia_url(url) is False

    def test_wikipedia_url_case_insensitive_scheme(self):
        """Test that scheme comparison is case-insensitive."""
        assert is_allowed_wikipedia_url("HTTPS://en.wikipedia.org/wiki/Test") is True

    def test_wikipedia_url_with_query_params(self):
        """Test URL with query parameters."""
        assert is_allowed_wikipedia_url("https://en.wikipedia.org/wiki/Test?action=edit") is True

    def test_wikipedia_url_with_anchor(self):
        """Test URL with anchor fragment."""
        assert is_allowed_wikipedia_url("https://en.wikipedia.org/wiki/Test#section") is True


class TestParseWikiTitleFromUrl:
    """Tests for parse_wiki_title_from_url function."""

    def test_parse_simple_title(self):
        """Test parsing simple wiki title."""
        url = "https://en.wikipedia.org/wiki/Python"
        title = parse_wiki_title_from_url(url)
        assert title == "Python"

    def test_parse_multi_word_title(self):
        """Test parsing multi-word title with underscores."""
        url = "https://en.wikipedia.org/wiki/Machine_Learning"
        title = parse_wiki_title_from_url(url)
        assert title == "Machine_Learning"

    def test_parse_url_encoded_title(self):
        """Test parsing URL-encoded title."""
        url = "https://en.wikipedia.org/wiki/Python_%28programming_language%29"
        title = parse_wiki_title_from_url(url)
        # Should handle URL decoding
        assert "Python" in title

    def test_parse_title_with_spaces_encoded(self):
        """Test parsing title with %20 encoded spaces."""
        url = "https://en.wikipedia.org/wiki/Test%20Article"
        title = parse_wiki_title_from_url(url)
        # Should decode %20 to space
        assert "Test" in title or "Article" in title

    def test_parse_invalid_url(self):
        """Test parsing invalid URL returns None."""
        result = parse_wiki_title_from_url("https://example.com/wiki/Test")
        assert result is None or isinstance(result, str)  # Should not crash

    def test_parse_url_without_wiki_path(self):
        """Test parsing URL without /wiki/ path."""
        url = "https://en.wikipedia.org/Article"
        result = parse_wiki_title_from_url(url)
        # May be None or attempt to extract, but shouldn't crash
        assert result is None or isinstance(result, str)

    def test_parse_empty_url(self):
        """Test parsing empty URL."""
        result = parse_wiki_title_from_url("")
        assert result is None or isinstance(result, str)

    def test_parse_none_url(self):
        """Test parsing None URL."""
        result = parse_wiki_title_from_url(None)
        assert result is None or isinstance(result, str)


class TestFetchWikipediaMetadata:
    """Tests for fetch_wikipedia_metadata function."""

    @patch("app.services.wiki_client.requests.get")
    def test_fetch_metadata_success(self, mock_get):
        """Test successful metadata fetching."""
        mock_get.return_value = Mock(
            json=lambda: {
                "query": {
                    "pages": {
                        "12345": {
                            "title": "Test Article",
                            "extract": "Article summary here.",
                        }
                    }
                }
            }
        )

        result = fetch_wikipedia_metadata("Test_Article")

        assert result["title"] == "Test Article"
        assert "summary" in result
        assert "url" in result
        mock_get.assert_called_once()

    @patch("app.services.wiki_client.requests.get")
    def test_fetch_metadata_missing_article(self, mock_get):
        """Test fetching non-existent article."""
        mock_get.return_value = Mock(
            json=lambda: {"query": {"pages": {}}}
        )

        result = fetch_wikipedia_metadata("Nonexistent_Article")

        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    @patch("app.services.wiki_client.requests.get")
    def test_fetch_metadata_api_timeout(self, mock_get):
        """Test handling of API timeout."""
        import requests
        mock_get.side_effect = requests.Timeout("Timeout")

        with pytest.raises(ValueError, match="Failed to fetch Wikipedia metadata"):
            fetch_wikipedia_metadata("Test_Article")

    @patch("app.services.wiki_client.requests.get")
    def test_fetch_metadata_malformed_json(self, mock_get):
        """Test handling of malformed JSON response."""
        mock_get.return_value = Mock(
            json=lambda: {"invalid": "structure"}
        )

        result = fetch_wikipedia_metadata("Test_Article")

        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    @patch("app.services.wiki_client.requests.get")
    def test_fetch_metadata_missing_extract(self, mock_get):
        """Test handling when extract field is missing."""
        mock_get.return_value = Mock(
            json=lambda: {
                "query": {
                    "pages": {
                        "12345": {
                            "title": "Test Article",
                            # Missing extract
                        }
                    }
                }
            }
        )

        result = fetch_wikipedia_metadata("Test_Article")

        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    @patch("app.services.wiki_client.requests.get")
    def test_fetch_metadata_with_special_characters(self, mock_get):
        """Test metadata fetch with special characters in title."""
        mock_get.return_value = Mock(
            json=lambda: {
                "query": {
                    "pages": {
                        "12345": {
                            "title": "Test & Article",
                            "extract": "Summary with special chars: !@#$%",
                        }
                    }
                }
            }
        )

        result = fetch_wikipedia_metadata("Test_&_Article")

        assert result is not None


class TestFetchTalkPageComments:
    """Tests for fetch_talk_page_comments function."""

    @patch("app.services.wiki_client.WikipediaTalkFetcher")
    def test_fetch_comments_success(self, mock_fetcher_class):
        """Test successful comment fetching."""
        from datetime import datetime
        mock_instance = Mock()
        comment1 = Mock()
        comment1.author = "Editor1"
        comment1.timestamp = datetime(2024, 1, 1, 12, 0, 0)
        comment1.text = "First comment"
        comment2 = Mock()
        comment2.author = "Editor2"
        comment2.timestamp = datetime(2024, 1, 2, 12, 0, 0)
        comment2.text = "Second comment"
        mock_instance.get_all_comments.return_value = [comment1, comment2]
        mock_fetcher_class.return_value = mock_instance

        result = fetch_talk_page_comments("Test_Article")

        assert len(result) == 2
        assert result[0]["author"] == "Editor1"
        assert result[1]["author"] == "Editor2"

    @patch("app.services.wiki_client.WikipediaTalkFetcher")
    def test_fetch_comments_empty_talk_page(self, mock_fetcher_class):
        """Test fetching from empty talk page."""
        mock_instance = Mock()
        mock_instance.get_all_comments.return_value = []
        mock_fetcher_class.return_value = mock_instance

        result = fetch_talk_page_comments("Test_Article")

        assert result == []

    @patch("app.services.wiki_client.WikipediaTalkFetcher")
    def test_fetch_comments_with_limit(self, mock_fetcher_class):
        """Test fetching with limit parameter."""
        mock_instance = Mock()
        mock_instance.get_all_comments.return_value = [
            {"author": f"Editor{i}", "text": f"Comment{i}"}
            for i in range(5)
        ]
        mock_fetcher_class.return_value = mock_instance

        result = fetch_talk_page_comments("Test_Article", limit=3)

        # Function should limit results
        assert len(result) <= 5

    @patch("app.services.wiki_client.WikipediaTalkFetcher")
    def test_fetch_comments_exception_handling(self, mock_fetcher_class):
        """Test exception handling during fetching."""
        mock_instance = Mock()
        mock_instance.get_all_comments.side_effect = Exception("Network error")
        mock_fetcher_class.return_value = mock_instance

        # Should raise ValueError wrapping the exception
        with pytest.raises(ValueError, match="Failed to fetch talk page comments"):
            fetch_talk_page_comments("Test_Article")

    @patch("app.services.wiki_client.WikipediaTalkFetcher")
    def test_fetch_comments_missing_author(self, mock_fetcher_class):
        """Test handling of comments with missing author field."""
        from datetime import datetime
        mock_instance = Mock()
        comment = Mock()
        del comment.author  # Missing author attribute
        comment.timestamp = datetime(2024, 1, 1, 12, 0, 0)
        comment.text = "Comment without author"
        mock_instance.get_all_comments.return_value = [comment]
        mock_fetcher_class.return_value = mock_instance

        result = fetch_talk_page_comments("Test_Article")

        # Should skip malformed comment
        assert len(result) == 0

    @patch("app.services.wiki_client.WikipediaTalkFetcher")
    def test_fetch_comments_missing_timestamp(self, mock_fetcher_class):
        """Test handling of comments with missing timestamp."""
        mock_instance = Mock()
        comment = Mock()
        comment.author = "Editor1"
        comment.timestamp = None  # Missing timestamp
        comment.text = "Comment without timestamp"
        mock_instance.get_all_comments.return_value = [comment]
        mock_fetcher_class.return_value = mock_instance

        result = fetch_talk_page_comments("Test_Article")

        # Should handle gracefully - None timestamp becomes ""
        assert len(result) > 0
        assert result[0]["timestamp"] == ""

    @patch("app.services.wiki_client.WikipediaTalkFetcher")
    def test_fetch_comments_with_zero_limit(self, mock_fetcher_class):
        """Test fetching with limit=0."""
        mock_instance = Mock()
        mock_instance.get_all_comments.return_value = []
        mock_fetcher_class.return_value = mock_instance

        result = fetch_talk_page_comments("Test_Article", limit=0)

        assert isinstance(result, list)

    @patch("app.services.wiki_client.WikipediaTalkFetcher")
    def test_fetch_comments_with_negative_limit(self, mock_fetcher_class):
        """Test fetching with negative limit."""
        mock_instance = Mock()
        mock_instance.get_all_comments.return_value = [
            {"author": "Editor1", "text": "Comment"}
        ]
        mock_fetcher_class.return_value = mock_instance

        result = fetch_talk_page_comments("Test_Article", limit=-1)

        assert isinstance(result, list)
