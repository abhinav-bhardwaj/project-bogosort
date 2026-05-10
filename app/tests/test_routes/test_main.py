"""Tests for main routes."""
import pytest


class TestMainRoutes:
    """Test main blueprint routes."""

    def test_index_redirect(self, client):
        """Test that index redirects to dashboard."""
        response = client.get("/")
        assert response.status_code == 302
        assert "/dashboard/" in response.location

    def test_index_redirect_follow(self, client):
        """Test following the redirect from index."""
        response = client.get("/", follow_redirects=True)
        assert response.status_code == 200
