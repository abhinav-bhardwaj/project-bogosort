"""Flask app entry point.

Run from the project root:
    flask --app app/main.py run

Or:
    python app/main.py
"""

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so analysis_and_inference imports work
PROJECT_ROOT = next(str(p) for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from flask import Flask

from app.routes.predict import bp as predict_bp


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_ROOT, "app/templates"),
        static_folder=os.path.join(PROJECT_ROOT, "app/static"),
    )
    app.register_blueprint(predict_bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
