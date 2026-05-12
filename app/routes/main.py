"""
main.py — core page routes for the Flask frontend

This module defines lightweight navigation routes for the main user-facing
pages of the application. Routes are intentionally limited to template rendering, while moderation,
evaluation, and inference logic remain isolated in dedicated service and API
layers. This separation keeps frontend navigation independent from backend
processing pipelines.

Dynamic article and comment routes pass identifiers directly into templates
instead of resolving data server-side. This allows frontend components to fetch
data asynchronously through the API layer, reducing template complexity and
improving modularity.
"""
from flask import Blueprint, render_template

main = Blueprint("main", __name__)

@main.route("/")
def landing():
    return render_template("landing.html")

@main.route("/analyze/")
def index():
    return render_template("index.html")

@main.route("/about/")
def about():
    return render_template("about.html")

@main.route("/articles/<article_id>/")
def article_page(article_id):
    return render_template("article.html", article_id=article_id)

@main.route("/articles/<article_id>/comments/<comment_id>/")
def comment_page(article_id, comment_id):
    return render_template("comment.html", article_id=article_id, comment_id=comment_id)

@main.route("/demo/")
def demo_page():
    return render_template("demo.html")