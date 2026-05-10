import logging
from analysis_and_inference.models.inference import predict_comment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ensemble"
EXPLAIN_VERSION = "v1"


def score_comment(text, model_name=DEFAULT_MODEL, explain=False):
    try:
        if model_name is None:
            model_name = DEFAULT_MODEL
        result = predict_comment(text, model_name=model_name, explain=explain)
        return {
            "label": int(result.get("label", 0)),
            "probability": float(result.get("probability") or 0.0),
            "top_features": result.get("top_features", []) if explain else [],
            "explain_version": EXPLAIN_VERSION if explain else "",
        }
    except Exception as exc:
        logger.error(f"Failed to score comment: {exc}", exc_info=True)
        return {
            "label": 0,
            "probability": 0.0,
            "top_features": [],
            "explain_version": "",
        }