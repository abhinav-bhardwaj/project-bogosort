import pytest
import json
import os
import tempfile
from pathlib import Path
from app import create_app


@pytest.fixture
def app():
    """Create and configure a test Flask app."""
    # Create a temporary directory for test data
    test_data_dir = tempfile.mkdtemp()

    app = create_app(config_name="testing")
    app.config["TESTING"] = True

    # Store test data dir in app for use by tests
    app.test_data_dir = test_data_dir

    with app.app_context():
        yield app

    # Cleanup
    import shutil
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def sample_evaluations():
    """Sample model evaluation data."""
    return {
        "models": [
            {
                "model_id": "model_001",
                "model_name": "Logistic Regression",
                "version": "1.0",
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.88,
                    "f1_score": 0.85
                }
            },
            {
                "model_id": "model_002",
                "model_name": "Random Forest",
                "version": "1.0",
                "metrics": {
                    "accuracy": 0.88,
                    "precision": 0.86,
                    "recall": 0.90,
                    "f1_score": 0.88
                }
            }
        ]
    }


@pytest.fixture
def evaluations_file(tmp_path, sample_evaluations):
    """Create a temporary evaluations JSON file."""
    eval_file = tmp_path / "model_evaluations.json"
    with open(eval_file, "w") as f:
        json.dump(sample_evaluations, f)
    return eval_file


@pytest.fixture
def monkeypatch_data_path(monkeypatch, evaluations_file):
    """Monkeypatch the DATA_PATH in queries module."""
    monkeypatch.setattr(
        "app.db.queries.DATA_PATH",
        evaluations_file
    )
    return evaluations_file


@pytest.fixture
def mock_toxic_words():
    """Sample toxic words data for bogosort tests."""
    return [
        ("toxic", 150),
        ("hate", 120),
        ("abuse", 100),
        ("spam", 80),
        ("rude", 60)
    ]
