"""
evaluation_service.py - service module for loading model evaluation metadata 
and serving evaluation artifacts 

This module provides model evaluation retrieval, artifact path resolution, 
and security validation for the Flask application. It loads evaluation metadata 
from a JSON file, resolves paths to evaluation artifacts like ROC curves and 
confusion matrices, and ensures that all file access is securely validated 
to prevent path traversal attacks.

Used by:
- api.py
- model evaluation dashboards
"""


import csv
import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_PATHS = [
    Path("data/model_evaluations.json"),
    Path("app/data/model_evaluations.json"),
]

MODEL_DIR_ALIASES = {
    "random_forest": "random_forest",
    "lasso_log_reg": "lasso_log_reg",
    "svm": "svm",
    "ridge_log_reg": "ridge_log_reg",
    "ensemble": "ensemble",
}

ARTIFACT_FILES = {
    "roc_curve": "roc_curve.png",
    "pr_curve": "pr_curve.png",
    "confusion_matrix": "confusion_matrix.png",
    "calibration": "calibration.png",
    "feature_importance": "feature_importance.png",
    "error_confidence_distribution": "error_confidence_distribution.png",
    "error_patterns_by_feature": "error_patterns_by_feature.png",
}

CSV_FILES = {
    "false_positives": "false_positives.csv",
    "false_negatives": "false_negatives.csv",
    "error_patterns_by_feature": "error_patterns_by_feature.csv",
}

ALLOWED_ARTIFACT_FILES = set(ARTIFACT_FILES.values())

SAFE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

def is_safe_model_id(model_id):
    return bool(model_id) and bool(SAFE_MODEL_ID_RE.match(model_id))


def _select_data_path():
    for path in DATA_PATHS:
        if path.exists():
            return path
    return DATA_PATHS[0]


def load_all_evaluations():
    data_path = _select_data_path()
    if not data_path.exists():
        logger.warning(f"Data file not found at {data_path}")
        return {"models": []}
    
    try:
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            logger.error(f"Data file contains non-dict type: {type(data)}")
            return {"models": []}
        
        if "models" not in data:
            logger.warning(f"Data file missing 'models' key. Keys: {list(data.keys())}")
            return {"models": []}
        
        if not isinstance(data["models"], list):
            logger.error(f"'models' is not a list: {type(data['models'])}")
            return {"models": []}
        
        return data
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSON from {data_path}: {exc}")
        return {"models": []}
    except Exception as exc:
        logger.error(f"Unexpected error loading evaluations: {exc}", exc_info=True)
        return {"models": []}


def get_model_evaluation(model_id=None):
    try:
        data = load_all_evaluations()
        models = data.get("models", [])
        
        if not models:
            logger.warning("No models available in evaluations")
            return {}
        
        if not model_id:
            return models[0]
        
        for model in models:
            if isinstance(model, dict) and model.get("model_id") == model_id:
                return model
        
        logger.warning(f"Model {model_id} not found in evaluations")
        return {}
    except Exception as exc:
        logger.error(f"Error in get_model_evaluation({model_id}): {exc}", exc_info=True)
        return {}


def get_model_version(model_id):
    try:
        evaluation = get_model_evaluation(model_id)
        version = evaluation.get("version", "") if evaluation else ""
        if version:
            logger.debug(f"Model {model_id} version: {version}")
        return version
    except Exception as exc:
        logger.error(f"Error in get_model_version({model_id}): {exc}")
        return ""


def resolve_artifact_dir(model_id):
    if not is_safe_model_id(model_id):
        logger.warning(f"Unsafe model_id: {model_id}")
        return None
    
    try:
        base = Path("analysis_and_inference/models").resolve()
        candidate = (base / model_id / "outputs/evaluation").resolve()
        
        if not str(candidate).startswith(str(base)):
            logger.warning(f"Path traversal attempt detected: {candidate}")
            return None
        
        if candidate.exists():
            logger.debug(f"Found artifact dir: {candidate}")
            return candidate
        
        alias = MODEL_DIR_ALIASES.get(model_id)
        if alias:
            alias_path = (base / alias / "outputs/evaluation").resolve()
            if not str(alias_path).startswith(str(base)):
                logger.warning(f"Path traversal attempt in alias: {alias_path}")
                return None
            if alias_path.exists():
                logger.debug(f"Found alias artifact dir: {alias_path}")
                return alias_path
        
        logger.debug(f"No artifact dir found for {model_id}")
        return None
    except Exception as exc:
        logger.error(f"Error resolving artifact dir for {model_id}: {exc}")
        return None


def _load_csv_sample(path, limit=5):
    try:
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                if len(rows) >= limit:
                    break
        return rows
    except Exception as exc:
        logger.error(f"Error loading CSV sample from {path}: {exc}")
        return []


def get_model_artifacts(model_id):
    try:
        base_dir = resolve_artifact_dir(model_id)
        if not base_dir or not base_dir.exists():
            logger.debug(f"No artifact directory for {model_id}")
            return {"images": {}, "samples": {}}

        images = {}
        for key, filename in ARTIFACT_FILES.items():
            try:
                path = base_dir / filename
                if path.exists():
                    images[key] = f"/api/models/{model_id}/artifacts/{filename}"
                    logger.debug(f"Found artifact {filename} for {model_id}")
            except Exception as exc:
                logger.warning(f"Error processing artifact {key} for {model_id}: {exc}")
                continue

        samples = {}
        for key, filename in CSV_FILES.items():
            try:
                path = base_dir / filename
                if path.exists():
                    samples[key] = _load_csv_sample(path)
                    logger.debug(f"Loaded CSV sample {key} for {model_id}")
            except Exception as exc:
                logger.warning(f"Error loading CSV sample {key} for {model_id}: {exc}")
                continue

        return {"images": images, "samples": samples}
    except Exception as exc:
        logger.error(f"Error in get_model_artifacts({model_id}): {exc}", exc_info=True)
        return {"images": {}, "samples": {}}