"""
toxicity_service.py - service module for toxicity inference and explanation generation

This module provides toxicity prediction wrappers, model inference timing, explanation 
feature extraction, standardized inference outputs,and inference error handling.

The service acts as an abstraction layer over the model inference
pipeline located in:
    analysis_and_inference.models.inference

Used by:
- article_service.py
- moderation workflows
- API endpoints
"""

import logging
import time
from analysis_and_inference.models.inference import predict_comment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ensemble"
EXPLAIN_VERSION = "v1"


def score_comment(text, model_name=DEFAULT_MODEL, explain=False):
    try:
        if model_name is None:
            model_name = DEFAULT_MODEL
        t0 = time.perf_counter()
        result = predict_comment(text, model_name=model_name, explain=explain)
        inference_ms = (time.perf_counter() - t0) * 1000
        return {
            "label": int(result.get("label", 0)),
            "probability": float(result.get("probability") or 0.0),
            "top_features": result.get("top_features", []) if explain else [],
            "explain_version": EXPLAIN_VERSION if explain else "",
            "inference_ms": round(inference_ms, 2),
        }
    except Exception as exc:
        logger.error(f"Failed to score comment: {exc}", exc_info=True)
        return {
            "label": 0,
            "probability": 0.0,
            "top_features": [],
            "explain_version": "",
            "inference_ms": 0.0,
        }