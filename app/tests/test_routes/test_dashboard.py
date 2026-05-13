"""Tests for dashboard routes."""
import pytest


class TestDashboardRoutes:
    """Test dashboard blueprint routes."""

    #def test_dashboard_page_renders(self, client):
    #    """Test that dashboard page renders successfully."""
    #    response = client.get("/dashboard/")
    #    assert response.status_code == 200
    #    assert b"<!DOCTYPE html>" in response.data or b"<html" in response.data#

    #def test_dashboard_returns_html(self, client):
    #    """Test that dashboard returns HTML content type."""
    #    response = client.get("/dashboard/")
    #    assert "text/html" in response.content_type

    def test_nerdy_dashboard_renders(self, client):
        """Test that nerdy dashboard page renders successfully."""
        response = client.get("/dashboard/nerdy/")
        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data or b"<html" in response.data

    def test_nerdy_dashboard_returns_html(self, client):
        """Test that nerdy dashboard returns HTML content type."""
        response = client.get("/dashboard/nerdy/")
        assert "text/html" in response.content_type

    def test_dashboard_not_found(self, client):
        """Test that non-existent dashboard route returns 404."""
        response = client.get("/dashboard/nonexistent/")
        assert response.status_code == 404
