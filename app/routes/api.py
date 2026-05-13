"""
api.py - REST API routes for moderation, evaluation and article management

This module centralizes all HTTP API endpoints behind a single Flask Blueprint.
Routes delegate business logic to service-layer modules instead of implementing
database, evaluation, or inference logic directly inside controllers.

This separation keeps routing lightweight and makes services independently testable,
reusable, and replaceable without changing endpoint structure.

Model artifact access is restricted through allowlists and path validation to
prevent unsafe filesystem access. All endpoints return structured JSON errors
instead of exposing internal exceptions directly.

The API layer also isolates Flask-specific request handling from the underlying
moderation and evaluation pipelines, simplifying future migration to other
interfaces if needed.
"""

import logging
from flask import Blueprint, request, jsonify, abort, send_from_directory

logger = logging.getLogger(__name__)

from app.services.article_service import (
    DEFAULT_MODEL,
    get_article,
    get_comment_detail,
    ingest_article,
    list_articles,
    list_comments,
    update_comment_decision,
    update_thresholds,
)

from app.services.evaluation_service import (
    ALLOWED_ARTIFACT_FILES,
    get_model_artifacts,
    get_model_evaluation,
    is_safe_model_id,
    load_all_evaluations,
    resolve_artifact_dir,
)
from app.services.wiki_client import is_allowed_wikipedia_url
from app.services.toxicity_service import score_comment

api = Blueprint('api', __name__)

MIN_LIMIT = 1
MAX_LIMIT = 200
DEFAULT_LIMIT = 30
DEFAULT_COMMENT_LIMIT = 50
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0
DEFAULT_AUTO_THRESHOLD = 0.75
DEFAULT_MANUAL_THRESHOLD = 0.55
MAX_OFFSET = 1000000
VALID_DECISIONS = {"auto-ban", "manual-ban", "manual-review", "none", "flagged"}
VALID_SORTS = {"toxicity_desc", "toxicity_asc", "timestamp_desc", "timestamp_asc", "decision_asc"}

def _parse_int(value, default, min_value, max_value, field_name):
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")
    return parsed


def _parse_float(value, default, min_value, max_value, field_name):
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")
    return parsed

def _parse_sort(value):
    if not value:
        return "toxicity_desc"
    if value not in VALID_SORTS:
        raise ValueError(f"sort must be one of {sorted(VALID_SORTS)}")
    return value


def _parse_decision(value):
    if not value:
        return None
    if value not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
    return value


def _attach_artifacts(evaluation, model_id):
    artifacts = get_model_artifacts(model_id)
    payload = {**evaluation, "artifacts": artifacts}
    images = artifacts.get("images", {})
    if images.get("roc_curve"):
        payload.setdefault("roc_curve", {})["image_url"] = images["roc_curve"]
        payload["roc_image_url"] = images["roc_curve"]
    if images.get("pr_curve"):
        payload.setdefault("pr_curve", {})["image_url"] = images["pr_curve"]
        payload["pr_image_url"] = images["pr_curve"]
    if images.get("confusion_matrix"):
        payload["confusion_matrix_image_url"] = images["confusion_matrix"]
    if images.get("calibration"):
        payload["calibration_image_url"] = images["calibration"]
    if images.get("feature_importance"):
        payload["feature_importance_image_url"] = images["feature_importance"]
    if images.get("error_confidence_distribution"):
        payload["error_confidence_distribution_url"] = images["error_confidence_distribution"]
    if images.get("error_patterns_by_feature"):
        payload["error_patterns_by_feature_url"] = images["error_patterns_by_feature"]
    return payload


@api.route("/demo/infer", methods=["POST"])
def demo_infer():
    payload = request.get_json() or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400
    if len(text) > 10000:
        return jsonify({"error": "Text too long (max 10 000 characters)"}), 400

    model_name = payload.get("model_name") or DEFAULT_MODEL
    try:
        auto_threshold = _parse_float(
            payload.get("auto_threshold"), DEFAULT_AUTO_THRESHOLD, MIN_THRESHOLD, MAX_THRESHOLD, "auto_threshold"
        )
        manual_threshold = _parse_float(
            payload.get("manual_threshold"), DEFAULT_MANUAL_THRESHOLD, MIN_THRESHOLD, MAX_THRESHOLD, "manual_threshold"
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = score_comment(text, model_name=model_name, explain=True)
    except Exception as exc:
        logger.error(f"Demo inference error: {exc}", exc_info=True)
        return jsonify({"error": "Inference failed. Please try again."}), 500

    probability = result["probability"]
    if probability >= auto_threshold:
        decision = "auto-ban"
    elif probability >= manual_threshold:
        decision = "manual-review"
    else:
        decision = "none"

    return jsonify({
        "text": text,
        "model_name": model_name,
        "toxicity": probability,
        "label": result["label"],
        "decision": decision,
        "auto_threshold": auto_threshold,
        "manual_threshold": manual_threshold,
        "inference_ms": result["inference_ms"],
        "top_features": result["top_features"],
    })


@api.route('/models', methods=['GET'])
def list_models():
    try:
        data = load_all_evaluations()
        models = data.get("models", [])
        
        if not models:
            return jsonify({"models": [], "error": "No models found"}), 400
        
        return jsonify({
            "models": [
                {
                    "model_id": m.get("model_id", ""),
                    "model_name": m.get("model_name", ""),
                    "version": m.get("version", ""),
                    "metrics": m.get("metrics", {}),
                }
                for m in models
                if isinstance(m, dict)
            ]
        })
    except Exception as exc:
        return jsonify({"models": [], "error": str(exc)}), 500


@api.route("/models/<model_id>/evaluation", methods=["GET"])
def model_evaluation(model_id):
    try:
        evaluation = get_model_evaluation(model_id)
        if not evaluation:
            return jsonify({})
        return jsonify(_attach_artifacts(evaluation, model_id))
    except Exception as exc:
        logger.error(f"Error fetching evaluation for {model_id}: {exc}", exc_info=True)
        return jsonify({"error": "Failed to fetch model evaluation"}), 500

@api.route("/evaluation", methods=["GET"])
def default_evaluation():
    try:
        model_id = request.args.get("model_id")
        evaluation = get_model_evaluation(model_id)
        if not evaluation:
            return jsonify({})
        return jsonify(_attach_artifacts(evaluation, evaluation.get("model_id")))
    except Exception as exc:
        logger.error(f"Error fetching default evaluation: {exc}", exc_info=True)
        return jsonify({"error": "Failed to fetch model evaluation"}), 500

@api.route("/models/<model_id>/artifacts/<path:filename>", methods=["GET"])
def model_artifact(model_id, filename):
    if filename not in ALLOWED_ARTIFACT_FILES:
        abort(404)
    if not is_safe_model_id(model_id):
        abort(404)
    base_dir = resolve_artifact_dir(model_id)
    if not base_dir:
        abort(404)
    try:
        return send_from_directory(base_dir, filename)
    except FileNotFoundError:
        logger.warning(f"Artifact not found: {model_id}/{filename}")
        abort(404)
    except (PermissionError, OSError) as exc:
        logger.error(f"Error accessing artifact {model_id}/{filename}: {exc}")
        abort(500)

@api.route("/articles", methods=["GET"])
def articles_list():
    try:
        return jsonify({"articles": list_articles()})
    except Exception as exc:
        logger.error(f"Failed to list articles: {exc}", exc_info=True)
        return jsonify({"error": "Failed to retrieve articles"}), 500

@api.route("/articles/ingest", methods=["POST"])
def articles_ingest():
    payload = request.get_json() or {}
    url = payload.get("url", "").strip()
    try:
        limit = _parse_int(payload.get("limit"), DEFAULT_LIMIT, MIN_LIMIT, MAX_LIMIT, "limit")
        auto_threshold = _parse_float(
            payload.get("auto_threshold"), DEFAULT_AUTO_THRESHOLD, MIN_THRESHOLD, MAX_THRESHOLD, "auto_threshold"
        )
        manual_threshold = _parse_float(
            payload.get("manual_threshold"),
            DEFAULT_MANUAL_THRESHOLD,
            MIN_THRESHOLD,
            MAX_THRESHOLD,
            "manual_threshold",
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    model_name = payload.get("model_name") or DEFAULT_MODEL

    if not url:
        return jsonify({"error": "Missing url"}), 400
    if not is_allowed_wikipedia_url(url):
        return jsonify({"error": "URL must be a Wikipedia article"}), 400
    if manual_threshold > auto_threshold:
        return jsonify({"error": "manual_threshold must be <= auto_threshold"}), 400

    try:
        article = ingest_article(
            url,
            limit=limit,
            auto_threshold=auto_threshold,
            manual_threshold=manual_threshold,
            model_name=model_name
        )
    except ValueError as exc:
        logger.warning(f"Article ingestion validation error: {exc}")
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        logger.error(f"Model unavailable during article ingestion: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.error(f"Unexpected error during article ingestion: {exc}", exc_info=True)
        return jsonify({"error": "Failed to ingest article. Please try again."}), 500
    return jsonify(article)

@api.route("/articles/<article_id>", methods=["GET"])
def article_detail(article_id):
    include_comments = request.args.get("include_comments", "true").lower() != "false"
    decision = request.args.get("decision")
    sort = request.args.get("sort")
    try:
        decision = _parse_decision(decision)
        sort = _parse_sort(sort)
        limit = _parse_int(
            request.args.get("limit"), DEFAULT_COMMENT_LIMIT, MIN_LIMIT, MAX_LIMIT, "limit"
        )
        offset = _parse_int(request.args.get("offset"), 0, 0, MAX_OFFSET, "offset")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        article = get_article(
            article_id,
            include_comments=include_comments,
            limit=limit,
            offset=offset,
            decision=decision,
            sort=sort,
        )
    except Exception as exc:
        logger.error(f"Failed to fetch article {article_id}: {exc}", exc_info=True)
        return jsonify({"error": "Failed to retrieve article"}), 500
    if not article:
        return jsonify({"error": "Article not found"}), 404
    return jsonify(article)

@api.route("/articles/<article_id>/thresholds", methods=["PUT"])
def article_thresholds(article_id):
    payload = request.get_json() or {}
    try:
        auto_threshold = _parse_float(
            payload.get("auto_threshold"), DEFAULT_AUTO_THRESHOLD, MIN_THRESHOLD, MAX_THRESHOLD, "auto_threshold"
        )
        manual_threshold = _parse_float(
            payload.get("manual_threshold"),
            DEFAULT_MANUAL_THRESHOLD,
            MIN_THRESHOLD,
            MAX_THRESHOLD,
            "manual_threshold",
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if manual_threshold > auto_threshold:
        return jsonify({"error": "manual_threshold must be <= auto_threshold"}), 400

    try:
        update_thresholds(article_id, auto_threshold, manual_threshold)
    except Exception as exc:
        logger.error(f"Failed to update thresholds for {article_id}: {exc}", exc_info=True)
        return jsonify({"error": "Failed to update thresholds"}), 500
    return jsonify({"status": "ok"})

@api.route("/articles/<article_id>/comments", methods=["GET"])
def article_comments(article_id):
    try:
        decision = _parse_decision(request.args.get("decision"))
        sort = _parse_sort(request.args.get("sort"))
        limit = _parse_int(
            request.args.get("limit"), DEFAULT_COMMENT_LIMIT, MIN_LIMIT, MAX_LIMIT, "limit"
        )
        offset = _parse_int(request.args.get("offset"), 0, 0, MAX_OFFSET, "offset")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        return jsonify(list_comments(article_id, limit=limit, offset=offset, decision=decision, sort=sort))
    except Exception as exc:
        logger.error(f"Failed to list comments for {article_id}: {exc}", exc_info=True)
        return jsonify({"error": "Failed to retrieve comments"}), 500

@api.route("/articles/<article_id>/comments/<comment_id>", methods=["GET"])
def comment_detail(article_id, comment_id):
    try:
        payload = get_comment_detail(article_id, comment_id)
    except Exception as exc:
        logger.error(f"Failed to fetch comment {comment_id}: {exc}", exc_info=True)
        return jsonify({"error": "Failed to retrieve comment"}), 500
    if not payload:
        return jsonify({"error": "Comment not found"}), 404
    return jsonify(payload)


@api.route("/articles/<article_id>/comments/<comment_id>", methods=["PATCH"])
def update_comment(article_id, comment_id):
    payload = request.get_json() or {}
    decision = payload.get("decision", "").strip()
    if decision not in VALID_DECISIONS:
        return jsonify({"error": f"decision must be one of {sorted(VALID_DECISIONS)}"}), 400
    try:
        update_comment_decision(article_id, comment_id, decision)
    except Exception as exc:
        logger.error(f"Failed to update decision for comment {comment_id}: {exc}", exc_info=True)
        return jsonify({"error": "Failed to update comment decision"}), 500
    return jsonify({"status": "ok", "decision": decision})