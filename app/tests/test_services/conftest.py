"""Fixtures for service layer tests."""
import json
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from app.services.session_manager import SessionManager


# ===== Sample Data Fixtures =====

@pytest.fixture
def sample_toxic_words():
    """Sample toxic words with counts for sorting tests."""
    return [
        ("toxic", 150),
        ("hate", 120),
        ("abuse", 100),
        ("spam", 80),
        ("rude", 60),
        ("bad", 50),
    ]


@pytest.fixture
def sample_article_data():
    """Sample article with comments for ingestion tests."""
    return {
        "title": "Test Article",
        "url": "https://en.wikipedia.org/wiki/Test_Article",
        "summary": "A test article summary.",
        "comments": [
            {
                "id": "comment1",
                "author": "User1",
                "timestamp": "2024-01-01T12:00:00",
                "text": "This is a toxic comment.",
            },
            {
                "id": "comment2",
                "author": "User2",
                "timestamp": "2024-01-02T12:00:00",
                "text": "This is a normal comment.",
            },
        ],
    }


@pytest.fixture
def sample_evaluation():
    """Sample model evaluation JSON structure."""
    return {
        "model_id": "ensemble",
        "model_name": "Ensemble Model",
        "version": "1.0.0",
        "metrics": {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.85,
            "f1_score": 0.87,
            "roc_auc": 0.95,
        },
        "threshold": 0.5,
        "training_date": "2024-01-01",
    }


@pytest.fixture
def sample_evaluations_dict(sample_evaluation):
    """Dictionary with multiple model evaluations."""
    ensemble_eval = sample_evaluation.copy()
    rf_eval = sample_evaluation.copy()
    rf_eval["model_id"] = "random_forest"
    rf_eval["model_name"] = "Random Forest"
    return {
        "models": [ensemble_eval, rf_eval],
        "timestamp": "2024-01-01T00:00:00",
    }


@pytest.fixture
def sample_wikipedia_metadata():
    """Sample Wikipedia API response format."""
    return {
        "title": "Test Article",
        "summary": "A comprehensive summary of the test article.",
        "url": "https://en.wikipedia.org/wiki/Test_Article",
    }


@pytest.fixture
def sample_wiki_comments():
    """Sample talk page comment list."""
    return [
        {
            "id": "",
            "author": "Editor1",
            "timestamp": "2024-01-01T12:00:00Z",
            "text": "This section needs improvement.",
        },
        {
            "id": "",
            "author": "Editor2",
            "timestamp": "2024-01-02T12:00:00Z",
            "text": "I disagree with this interpretation.",
        },
    ]


# ===== File/Path Fixtures =====

@pytest.fixture
def temp_npy_file(tmp_path, sample_toxic_words):
    """Create a temporary .npy file with toxic words data."""
    npy_path = tmp_path / "toxic_words.npy"
    data = np.array(sample_toxic_words, dtype=object)
    np.save(npy_path, data)
    return npy_path


@pytest.fixture
def temp_evaluation_json(tmp_path, sample_evaluations_dict):
    """Create a temporary evaluation JSON file."""
    json_path = tmp_path / "model_evaluations.json"
    json_path.write_text(json.dumps(sample_evaluations_dict))
    return json_path


@pytest.fixture
def temp_model_artifacts_dir(tmp_path):
    """Create a temporary model artifacts directory structure."""
    artifacts_dir = tmp_path / "analysis_and_inference" / "models" / "ensemble" / "outputs" / "evaluation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create sample image files (empty PNG stubs)
    for image_name in ["roc_curve.png", "pr_curve.png", "confusion_matrix.png", "calibration.png"]:
        (artifacts_dir / image_name).write_text("PNG_STUB")

    # Create sample CSV file
    csv_file = artifacts_dir / "false_positives.csv"
    csv_file.write_text("id,text,toxicity\n1,bad,0.8\n2,mean,0.75\n")

    return artifacts_dir


# ===== Mock Fixtures =====

@pytest.fixture
def mock_predict_comment():
    """Mock toxicity_service.predict_comment function."""
    with patch("app.services.toxicity_service.predict_comment") as mock:
        mock.return_value = {
            "label": 1,
            "probability": 0.85,
            "top_features": [("word1", 0.5), ("word2", 0.3)],
            "explain_version": "v1",
        }
        yield mock


@pytest.fixture
def mock_wikipedia_api():
    """Mock requests library for Wikipedia API calls."""
    with patch("app.services.wiki_client.requests.get") as mock:
        mock.return_value = Mock(
            status_code=200,
            json=lambda: {
                "query": {
                    "pages": {
                        "12345": {
                            "title": "Test Article",
                            "extract": "A comprehensive summary.",
                        }
                    }
                }
            },
        )
        yield mock


@pytest.fixture
def mock_talk_fetcher():
    """Mock WikipediaTalkFetcher for comment fetching."""
    with patch("app.services.article_service.WikipediaTalkFetcher") as mock_class:
        mock_instance = MagicMock()
        mock_instance.get_all_comments.return_value = [
            {
                "id": "",
                "author": "Editor1",
                "timestamp": "2024-01-01T12:00:00Z",
                "text": "First comment",
            },
            {
                "id": "",
                "author": "Editor2",
                "timestamp": "2024-01-02T12:00:00Z",
                "text": "Second comment",
            },
        ]
        mock_class.return_value = mock_instance
        yield mock_class


@pytest.fixture
def mock_article_repository():
    """Mock article_repository for database operations."""
    mock = MagicMock()
    mock.list_articles.return_value = []
    mock.list_comments.return_value = ([], 0)
    mock.get_article.return_value = None
    mock.upsert_article.return_value = {"id": "test_id", "title": "Test"}
    yield mock


# ===== Session Fixtures =====

@pytest.fixture
def session_manager():
    """Pre-configured SessionManager instance for testing."""
    return SessionManager(timeout_minutes=10)


# ===== Parametrize Fixtures =====

@pytest.fixture(
    params=[
        "https://en.wikipedia.org/wiki/Test_Article",
        "https://simple.wikipedia.org/wiki/Simple_Test",
    ]
)
def valid_wikipedia_urls(request):
    """Valid Wikipedia URLs for parametrized tests."""
    return request.param


@pytest.fixture(
    params=[
        "https://example.com/wiki/Article",
        "https://en.wikipedia.org/Article",
        "ftp://en.wikipedia.org/wiki/Article",
        "",
        None,
    ]
)
def invalid_wikipedia_urls(request):
    """Invalid Wikipedia URLs for parametrized tests."""
    return request.param


@pytest.fixture(
    params=[
        (0.3, 0.1),  # Below both thresholds
        (0.5, 0.7),  # Between thresholds
        (0.9, 0.95),  # Above both
        (0.75, 0.75),  # On both thresholds
        (0.0, 1.0),  # Boundary values
    ]
)
def threshold_pairs(request):
    """Parametrized threshold pairs (auto, manual) for decision logic testing."""
    return request.param


# ===== Support Functions =====

def create_temporary_model_structure(tmp_path, model_id="test_model"):
    """Helper to create a complete temporary model artifact structure."""
    base_path = tmp_path / "analysis_and_inference" / "models" / model_id / "outputs" / "evaluation"
    base_path.mkdir(parents=True, exist_ok=True)

    # Create image files
    for img_file in ["roc_curve.png", "pr_curve.png", "confusion_matrix.png"]:
        (base_path / img_file).write_text("PNG_DATA")

    # Create CSV files
    (base_path / "false_positives.csv").write_text("id,text\n1,bad\n2,mean\n")
    (base_path / "false_negatives.csv").write_text("id,text\n1,good\n")

    return base_path


# Make helper available to tests
pytest_plugins = []
