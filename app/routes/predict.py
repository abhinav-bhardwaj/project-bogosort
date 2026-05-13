"""Routes: home page (form) + /predict endpoint."""

from flask import Blueprint, render_template, request, jsonify

from analysis_and_inference.models.inference import predict_comment


bp = Blueprint("predict", __name__)


@bp.route("/", methods=["GET", "POST"])
def index():
    """Home page: shows a form, and on POST shows the classification result."""
    result = None
    text   = ""
    if request.method == "POST":
        text   = request.form.get("comment", "").strip()
        if text:
            result = predict_comment(text, top_k=8)
    return render_template("index.html", text=text, result=result)


@bp.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint: POST {"comment": "..."} → {"label", "probability", "top_features"}."""
    payload = request.get_json(silent=True) or {}
    text    = (payload.get("comment") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'comment' field"}), 400
    return jsonify(predict_comment(text, top_k=int(payload.get("top_k", 10))))
