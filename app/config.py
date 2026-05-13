"""
config.py - application configuration module

This module defines environment-specific configuration classes used across
the application. It centralizes runtime settings such as database connection
paths, secret keys, debug flags, and testing behavior.

Used by:
    - Flask application factory
    - Database initialization
    - Session management
    - Testing framework
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    """Base configuration."""
    SQL_URI = os.environ.get("SQL_URI", f"sqlite:///{_PROJECT_ROOT}/app/db/articles.db")
    SECRET_KEY = os.environ.get("SECRET_KEY", "jigsaw_secret_key")
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    SQL_URI = os.environ.get("TEST_SQL_URI", f"sqlite:///{_PROJECT_ROOT}/app/db/test_articles.db")

class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")  # Must be set in environment