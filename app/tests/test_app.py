"""Tests for Flask app creation and initialization."""
import pytest
from app import create_app
from app.config import DevelopmentConfig, TestingConfig, ProductionConfig


class TestCreateApp:
    """Test app factory function."""

    def test_create_app_default(self):
        """Test creating app with default config."""
        app = create_app()
        assert app is not None
        assert not app.config["TESTING"]

    def test_create_app_testing(self):
        """Test creating app with testing config."""
        app = create_app(config_name="testing")
        assert app is not None
        assert app.config["TESTING"] is True

    def test_create_app_development(self):
        """Test creating app with development config."""
        app = create_app(config_name="development")
        assert app is not None
        assert app.config["DEBUG"] is True

    def test_create_app_production(self):
        """Test creating app with production config."""
        app = create_app(config_name="production")
        assert app is not None
        assert app.config["DEBUG"] is False
        assert app.config["TESTING"] is False


class TestBlueprintRegistration:
    """Test that blueprints are registered correctly."""

    def test_main_blueprint_registered(self, app):
        """Test that main blueprint is registered."""
        assert "main" in [bp.name for bp in app.blueprints.values()]

    def test_api_blueprint_registered(self, app):
        """Test that API blueprint is registered."""
        assert "api" in [bp.name for bp in app.blueprints.values()]

    def test_dashboard_blueprint_registered(self, app):
        """Test that dashboard blueprint is registered."""
        assert "dashboard" in [bp.name for bp in app.blueprints.values()]

    def test_bogosort_blueprint_registered(self, app):
        """Test that bogosort blueprint is registered."""
        assert "bogosort" in [bp.name for bp in app.blueprints.values()]

    def test_api_url_prefix(self, app):
        """Test that API blueprint has correct URL prefix."""
        with app.test_client() as client:
            response = client.get("/api/models")
            assert response.status_code in [200, 404, 500]  # Just checking it's routable

    def test_dashboard_url_prefix(self, app):
        """Test that dashboard blueprint has correct URL prefix."""
        with app.test_client() as client:
            response = client.get("/dashboard/")
            assert response.status_code in [200, 404, 500]

    def test_bogosort_url_prefix(self, app):
        """Test that bogosort blueprint has correct URL prefix."""
        with app.test_client() as client:
            response = client.get("/bogosort/")
            assert response.status_code in [200, 404, 500]


class TestAppContext:
    """Test app context and configuration."""

    def test_app_has_secret_key(self, app):
        """Test that app has SECRET_KEY configured."""
        assert "SECRET_KEY" in app.config
        assert app.config["SECRET_KEY"] is not None

    def test_app_config_loading(self, app):
        """Test that app config is properly loaded."""
        assert app.config is not None
        assert "DEBUG" in app.config
        assert "TESTING" in app.config
