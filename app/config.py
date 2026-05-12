import os

class Config:
    """Base configuration."""
    SQL_URI = os.environ.get("SQL_URI", "sqlite:///app/db/articles.db")
    SECRET_KEY = os.environ.get("SECRET_KEY", "jigsaw_secret_key")
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    SQL_URI = os.environ.get("TEST_SQL_URI", "sqlite:///app/data/test_articles.db")

class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")  # Must be set in environment