"""Tests for Flask configuration."""
import pytest
import os
from app.config import Config, DevelopmentConfig, TestingConfig, ProductionConfig


class TestBaseConfig:
    """Test base configuration."""

    def test_config_has_secret_key(self):
        """Test that Config has a SECRET_KEY."""
        assert Config.SECRET_KEY is not None

    def test_debug_is_false(self):
        """Test that debug is False by default."""
        assert Config.DEBUG is False

    def test_testing_is_false(self):
        """Test that testing is False by default."""
        assert Config.TESTING is False


class TestDevelopmentConfig:
    """Test development configuration."""

    def test_inherits_from_config(self):
        """Test that DevelopmentConfig inherits from Config."""
        assert issubclass(DevelopmentConfig, Config)

    def test_debug_is_true(self):
        """Test that debug is True for development."""
        assert DevelopmentConfig.DEBUG is True

    def test_testing_is_false(self):
        """Test that testing is False for development."""
        assert DevelopmentConfig.TESTING is False


class TestTestingConfig:
    """Test testing configuration."""

    def test_inherits_from_config(self):
        """Test that TestingConfig inherits from Config."""
        assert issubclass(TestingConfig, Config)

    def test_testing_is_true(self):
        """Test that testing is True for testing config."""
        assert TestingConfig.TESTING is True

    def test_debug_is_false(self):
        """Test that debug is False for testing config."""
        assert TestingConfig.DEBUG is False


class TestProductionConfig:
    """Test production configuration."""

    def test_inherits_from_config(self):
        """Test that ProductionConfig inherits from Config."""
        assert issubclass(ProductionConfig, Config)

    def test_debug_is_false(self):
        """Test that debug is False for production."""
        assert ProductionConfig.DEBUG is False

    def test_testing_is_false(self):
        """Test that testing is False for production."""
        assert ProductionConfig.TESTING is False
