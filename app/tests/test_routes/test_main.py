"""Tests for main routes."""
import pytest


class TestMainRoutes:
    """Test main blueprint routes."""

    def test_index_renders_landing(self, client):
        """Test that index renders the landing page."""
        response = client.get("/")
        assert response.status_code == 200

    def test_index_redirect_follow(self, client):
        """Test following the redirect from index."""
        response = client.get("/", follow_redirects=True)
        assert response.status_code == 200
