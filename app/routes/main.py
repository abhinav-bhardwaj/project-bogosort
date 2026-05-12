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