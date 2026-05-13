---
title: Wikipedia Toxic Comment Classifier
sdk: docker
app_port: 7860
---

# Project Bogosort - Toxic Comment Classifier

## Overview

This project is a comprehensive machine learning web application that automatically detects toxic comments in text and provides tools for content moderation. It was developed in Spring 2026 by Team Bogosort for the *Data Structures and Algorithms* course at the Hertie School in Berlin, in partial fulfilment of the Master of Data Science for Public Policy (MDS) programme.

**Team:** Bianca Rosca-Mayer, Helena Kandjumbwa, David Moth, Alexis Grangier, Aarushi Mahajan, Abhinav Dubey, Klaas Wolff

---

## Purpose and Target Audience

Online platforms - social media sites, comment sections, Wikipedia talk pages, and community forums - generate enormous volumes of user-generated text. Human moderation at that scale is slow, costly, and psychologically taxing for moderators. This application provides an automated first line of defence: it classifies whether a comment is toxic and, crucially, explains *why* the model reached that decision.

The tool is primarily aimed at:

- **Platform moderators and trust-and-safety teams** who need to triage large queues of flagged content quickly and want to understand which signals triggered a decision before acting on it.
- **Policy researchers and public-sector analysts** who study online toxicity and need an interpretable, auditable classifier - not a black box - to support their analysis.
- **Students and educators** learning about applied NLP and machine learning, who can inspect every step of the pipeline from raw text to final prediction.

Unlike commercial moderation APIs, every component here is open, inspectable, and reproducible: the features are hand-engineered and fully documented, the models are standard scikit-learn estimators, and each prediction comes with SHAP-based feature attributions so a human reviewer can see exactly which words, patterns, or linguistic signals pushed the model toward a toxic or non-toxic verdict.

---

## Key Features

### 1. **Interactive Toxicity Classifier**
Submit a comment and get:
- Binary toxicity verdict (toxic / non-toxic)
- Confidence probability (0.0–1.0)
- Top N features ranked by SHAP value, showing which linguistic signals influenced the decision
- Real-time inference powered by an ensemble of five trained models

### 2. **Article & Comment Management**
- Ingest Wikipedia or custom articles via URL
- Automatically fetch and classify all comments in an article
- Browse articles with toxicity statistics
- View detailed comment-level predictions with full explainability
- Manually override automated decisions
- Adjust toxicity thresholds on-the-fly and re-evaluate articles

### 3. **Wikipedia Integration**
- Direct integration with Wikipedia API to fetch articles and talk pages
- Automatic ingestion of all comments from a Wikipedia article's discussion page
- URL validation and safe fetching

### 4. **Model Evaluation Dashboard**
- View performance metrics for all six trained models (confusion matrices, ROC/PR curves, calibration plots)
- Browse per-model feature importance and error analysis
- Inspect false positives and false negatives with detailed patterns
- Download evaluation artifacts

### 5. **Exploratory Data Analysis (EDA) Viewer**
- Interactive visualization of the Jigsaw dataset characteristics
- Feature distributions, class imbalance analysis, toxicity patterns
- Cached EDA to avoid recomputation

### 6. **Bogosort Sorting Algorithm Demo**
- Visual comparison of Bogosort (randomized) vs. MergeSort (divide-and-conquer)
- Demonstrates algorithmic complexity differences interactively
- Session-based state management for concurrent users

---

## Dataset

The project uses the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset from Kaggle. It contains ~160,000 Wikipedia talk-page comments, each labelled for six toxicity categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. This project treats toxicity as a binary classification task (toxic vs. non-toxic).

The dataset is class-imbalanced: toxic comments are a minority. The pipeline handles this through stratified splitting, class-balanced model training, and F1-optimised decision thresholds.

Place the raw data at:
```
data/raw/jigsaw-dataset/train.csv.zip
```

---

## Project Structure

```
project-bogosort/
├── analysis_and_inference/          # Model training, feature engineering, evaluation
│   ├── EDA/
│   │   ├── eda_results.ipynb        # Exploratory data analysis notebook
│   │   ├── eda_processor.py          # EDA cache computation
│   │   └── eda_cache.json            # Cached EDA results
│   ├── features/
│   │   ├── build_features.py         # DenseFeatureTransformer (~32 features)
│   │   └── tests/
│   ├── models/
│   │   ├── _common.py                # Shared training, tuning, evaluation utilities
│   │   ├── run_all.py                # End-to-end training orchestrator (8 steps)
│   │   ├── inference.py              # Single-comment prediction API for Flask
│   │   ├── split_and_features/       # Data splitting and feature caching
│   │   ├── baseline/baseline.py      # Dummy stratified classifier
│   │   ├── lasso_log_reg/            # L1 logistic regression (custom implementation)
│   │   ├── ridge_log_reg/            # L2 logistic regression (sklearn)
│   │   ├── random_forest/            # RandomForest with feature ablation
│   │   ├── svm/svm.py                # Linear SVM
│   │   └── ensemble/ensemble.py      # Soft-vote ensemble (final classifier)
│   └── evaluation_code/
│       ├── evaluator.py              # Metrics, confusion matrix, plots
│       ├── error_analysis.py         # FP/FN inspection, error patterns
│       └── feature_evaluation.py     # SHAP and permutation importance
├── app/                             # Flask web application
│   ├── config.py                    # Environment-based configs (dev/test/prod)
│   ├── __init__.py                  # Flask app factory (create_app)
│   ├── db/
│   │   ├── __init__.py              # Database initialization
│   │   ├── queries.py               # SQL query builders
│   │   └── article_repository.py    # Article data access layer
│   ├── routes/
│   │   ├── main.py                  # Core navigation pages
│   │   ├── api.py                   # REST API endpoints (moderation, evaluation, articles)
│   │   ├── dashboard.py             # Model evaluation dashboard
│   │   ├── bogosort.py              # Sorting algorithm demo
│   │   └── eda.py                   # EDA visualization routes
│   ├── services/
│   │   ├── toxicity_service.py      # Toxicity scoring (inference wrapper)
│   │   ├── article_service.py       # Article ingestion, comment management
│   │   ├── evaluation_service.py    # Model evaluation metrics and artifacts
│   │   ├── eda_service.py           # EDA cache loading and serving
│   │   ├── wiki_client.py           # Wikipedia API integration
│   │   ├── wikipedia_talk_fetcher.py# Fetches Wikipedia talk pages
│   │   ├── sorting_service.py       # Bogosort/MergeSort implementation
│   │   ├── session_manager.py       # Session state for sorting demo
│   │   └── compute_eda_cache.py     # Generate EDA cache offline
│   ├── templates/                   # HTML Jinja2 templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── error.html
│   │   └── [feature-specific templates]
│   ├── static/
│   │   ├── styles/                  # CSS stylesheets
│   │   ├── js/                      # Client-side JavaScript
│   │   └── team_bio/team_bio.json   # Team member profiles
│   └── tests/
│       ├── test_routes/             # Route integration tests
│       ├── test_services/           # Service unit tests
│       ├── test_db/                 # Database tests
│       └── conftest.py              # Pytest fixtures
├── data/
│   ├── raw/jigsaw-dataset/          # Raw Kaggle CSVs (untracked, .gitignore)
│   └── processed/                   # Cached splits, matrices, embeddings
├── admin/
│   ├── meeting_minutes/             # Team meeting notes
│   ├── documentations/              # Git and collaboration guidelines
│   ├── design_sprint/               # Design sprint artifacts
│   └── human_centered_design_sprint/
├── .github/workflows/ci.yml         # CI test + CD deploy to Hugging Face Spaces
├── Dockerfile                       # Container image for Hugging Face Spaces
├── pyproject.toml                   # Project dependencies and metadata
├── uv.lock                          # Locked dependency versions (uv)
├── wsgi.py                          # WSGI entry point for production
├── run.py                           # Development server entry point
├── license.txt                      # CC BY-NC 4.0 license
└── README.md                        # This file
```

---

## Feature Engineering

All text features are computed by `DenseFeatureTransformer` - a stateless, sklearn-compatible transformer that produces ~32 numerical features per comment in a single pass. Features are grouped as follows:

| Group | Features |
|---|---|
| **Sentiment (VADER)** | `vader_compound`, `vader_neg`, `vader_pos`, `vader_is_negative`, `vader_intensity`, `vader_pos_minus_neg` |
| **Second-person pronouns** | `has_second_person`, `second_person_count`, `second_person_density` |
| **Profanity** | `profanity_count`, `obfuscated_profanity_count` (detects leetspeak substitutions like `$` → s, `@` → a) |
| **Slang** | `slang_count` (covers aggressive abbreviations, death wishes, derogatory terms) |
| **Text shape** | `char_count`, `word_count`, `exclamation_count`, `uppercase_ratio` |
| **Unique words** | `unique_word_ratio` |
| **Elongation** | `elongated_token_count` (e.g. "cooool"), `consecutive_punct_count` (e.g. "!!!") |
| **URLs / IPs** | `url_count`, `ip_count`, `has_url_or_ip` |
| **Syntactic** | `negation_count`, `sentence_count`, `avg_sentence_length` |
| **Identity mentions** | `identity_mention_count`, `identity_race`, `identity_gender`, `identity_sexuality`, `identity_religion`,`identity_disability`,`identity_nationality`|

Features are computed once, scaled with a `StandardScaler`, and cached to disk - all models share the same pre-computed feature matrix so GridSearchCV folds do not redundantly re-run the transformer.

---

## Models

Five classifiers are trained and evaluated independently, then combined into a final ensemble.

### Baseline - Dummy Classifier
A `DummyClassifier` with stratified strategy. Sets the floor: any real model must beat this.

### Custom Lasso Logistic Regression
Built **from scratch** using gradient descent with soft-thresholding (the L1 penalty is non-differentiable at zero, so standard gradient descent does not apply). The implementation follows the sklearn estimator API and supports sample weighting, configurable decision thresholds, and convergence diagnostics. L1 regularization encourages sparse weights, improving interpretability by zeroing out weak features.

### Ridge Logistic Regression
sklearn's `LogisticRegression` with L2 penalty (`solver="lbfgs"`). L2 regularization stabilizes coefficient estimates while retaining information from correlated features. Class balancing is enabled.

### Random Forest
`RandomForestClassifier` with `class_weight="balanced"`. Captures nonlinear feature interactions and is robust to correlated predictors. A separate `feature_selection.py` script performs post-hoc ablation - comparing full feature set performance against top-5 and top-10 subsets - to quantify which features carry the most predictive weight.

### Linear SVM
`LinearSVC` with class balancing. Well-suited to high-dimensional feature spaces. Because `LinearSVC` does not expose `predict_proba`, the raw `decision_function` output is used as a ranking score for ROC-AUC and PR-AUC, and threshold tuning is skipped automatically.

### Ensemble (final classifier)
A soft-vote ensemble over Lasso, Ridge, and Random Forest (SVM is excluded - no `predict_proba`). Soft voting averages predicted probabilities rather than hard class labels, preserving model confidence and producing smoother decisions on ambiguous comments. The three member models are loaded from their serialized artifacts without retraining; the `VotingClassifier` is manually wired with pre-fit estimators.

---

## Training Pipeline

All steps are orchestrated by `run_all.py`:

```bash
uv run python analysis_and_inference/models/run_all.py
```

The script runs 8 sequential steps, skipping any that already have cached outputs on disk:

| Step | Action |
|---|---|
| 1 | Stratified 80/20 train/test split (seed 42) |
| 1b | Pre-compute dense features + fit `StandardScaler` (cached to `split_and_features/features.pkl`) |
| 2 | Train Baseline |
| 3 | Train Lasso (grid search over `alpha` ∈ {0.001, 0.01, 0.1} × `learning_rate` ∈ {0.01, 0.1}) |
| 4 | Train Random Forest (grid search over `n_estimators`, `max_depth`, `min_samples_split`) |
| 5 | Train Ridge (grid search over `C` ∈ {0.01, 0.1, 1.0, 10.0}) |
| 6 | Train SVM (grid search over `C` ∈ {0.01, 0.1, 1.0, 10.0}) |
| 7 | Build soft-vote Ensemble from saved member artifacts |
| 8 | Run error analysis on all six models |

All models use 3-fold cross-validation with `average_precision` as the scoring metric. After grid search, an F1-optimal decision threshold is tuned on out-of-fold predictions and wrapped via `FixedThresholdClassifier`.

Each trained model is saved as `<model_name>_tuned.pkl` under its `outputs/` folder.

---

## Evaluation

Every model produces the following evaluation artifacts saved to `outputs/evaluation/`:

| Artifact | Description |
|---|---|
| `confusion_matrix.png` | True/false positive and negative counts |
| `roc_curve.png` | ROC curve with AUC |
| `pr_curve.png` | Precision-recall curve with AP score |
| `calibration.png` | Reliability of predicted probabilities (skipped for SVM) |
| `feature_importance.png` | Top 20 features by coefficient magnitude or Gini importance |
| `false_positives.csv` | Highest-confidence non-toxic comments flagged as toxic |
| `false_negatives.csv` | Lowest-confidence toxic comments that were missed |
| `error_patterns_by_feature.png/.csv` | Mean feature values for FP and FN vs. correct predictions |
| `error_confidence_distribution.png` | Score distributions for TP, TN, FP, FN |

Error analysis can also be run standalone for any model:

```bash
uv run python analysis_and_inference/evaluation_code/error_analysis.py lasso_log_reg
```

---

## Inference and Explainability

The `inference.py` module provides a `predict_comment()` function used by the Flask app. It classifies a single raw comment and returns a toxicity label, a probability score, and SHAP-based feature attributions showing which features pushed the prediction toward or away from toxic.

```python
from analysis_and_inference.models.inference import predict_comment

result = predict_comment("you are a complete idiot")
# {
#   "label": 1,
#   "probability": 0.949,
#   "top_features": [
#       {"feature": "profanity_count",     "value": 1.0, "shap": +0.21},
#       {"feature": "vader_compound",      "value": -0.55, "shap": +0.18},
#       {"feature": "second_person_count", "value": 1.0, "shap": +0.09},
#       ...
#   ]
# }
```

The model, scaler, and SHAP explainer are loaded once at first call and cached for all subsequent requests. SHAP uses a `KernelExplainer` with a 30-sample background drawn from the training set. Features are ranked by absolute SHAP magnitude so borderline predictions remain interpretable.

---

## Web Application Architecture

The Flask application (`app/`) is structured in layers:

### **Routes** (`app/routes/`)
- Lightweight controllers that delegate business logic to services
- No database access or inference logic directly in route handlers
- Consistent JSON error handling for API routes

### **Services** (`app/services/`)
- **toxicity_service**: Wraps inference layer, handles model loading and caching
- **article_service**: Manages article ingestion, comment CRUD, decision tracking
- **evaluation_service**: Serves model evaluation artifacts and metrics
- **wiki_client**: Validates and fetches Wikipedia URLs
- **wikipedia_talk_fetcher**: Downloads and parses Wikipedia talk pages
- **eda_service**: Loads and serves cached EDA visualizations
- **sorting_service**: Implements Bogosort and MergeSort algorithms
- **session_manager**: Manages state for concurrent sorting demo users

### **Database** (`app/db/`)
- SQLite by default; configurable via `SQL_URI` in config
- Article repository provides abstraction layer over raw SQL
- Query builders in `queries.py` centralize SQL logic

### **Configuration** (`app/config.py`)
- Environment-specific configs (development, testing, production)
- Database URI, logging, cache paths configurable via environment variables

---

## API Endpoints (REST)

All API endpoints are under `/api/`:

### Toxicity Scoring
- `POST /api/score` — Score a single comment
  - Input: `{"text": "comment text"}`
  - Output: `{"label": 0/1, "probability": 0.X, "features": [...]}`

### Article Management
- `GET /api/articles` — List all articles (paginated)
- `POST /api/articles` — Ingest a new article from URL
- `GET /api/articles/<id>` — Get article details with toxicity summary
- `GET /api/articles/<id>/comments` — List comments in article (paginated, filterable by toxicity)
- `PUT /api/articles/<id>/comments/<comment_id>` — Override comment decision
- `PUT /api/articles/<id>/thresholds` — Re-evaluate article with new thresholds

### Model Evaluation
- `GET /api/evaluations` — Load all model evaluation results
- `GET /api/evaluations/<model>` — Get evaluation for a specific model
- `GET /api/evaluations/<model>/artifacts/<filename>` — Download evaluation artifact (plot, CSV)

---

## Requirements

- Python >= 3.10
- **Core:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `vaderSentiment`, `matplotlib`, `flask`, `requests`
- **Dev/testing:** `pytest >= 9.0.2`
- **Optional - BERT embeddings** (large download, ~2 GB, only needed if `BertTransformer` is used):
  ```bash
  uv add torch transformers
  ```

---

## Installation

```bash
# Install all dependencies
uv sync

# Install dev dependencies (pytest)
uv sync --group dev
```

---

## Running the Application

### Development Server
```bash
python run.py
```
Runs on `http://localhost:5000` by default.

### Production Server
```bash
gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app
```

---

## Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific module
uv run pytest app/tests/test_services/test_toxicity_service.py

# Run with coverage
uv run pytest --cov=app --cov=analysis_and_inference
```

---

## Continuous Integration and Deployment

The project ships with a single GitHub Actions workflow at [.github/workflows/ci.yml](.github/workflows/ci.yml) that handles both CI (running the test suite on every push and pull request) and CD (deploying the live web app to Hugging Face Spaces on every push to `dev`).

**Live app:** https://huggingface.co/spaces/David-Moth/Wikipedia-Toxic-Comment-Classifier

### Continuous Integration — the `test` job

Triggered on every `push` to `dev` and every `pull_request` targeting `dev`. The job runs on `ubuntu-latest` and executes the following steps:

1. **Checkout** the repository.
2. **Install `uv`** via Astral's official action — the same package manager used locally.
3. **Set up Python 3.10**, matching the version pinned in `pyproject.toml`.
4. **Install dependencies** with `uv sync --frozen` so the CI environment matches `uv.lock` exactly. This catches lockfile drift early.
5. **Run the test suite** with `uv run pytest app/tests/ -v`. Any failing test fails the workflow and blocks merges.

This guards against regressions in routes, services, the inference pipeline, and the database layer.

### Continuous Deployment — the `deploy` job

Runs only on `push` events to `dev` (skipped for pull requests) and only after the `test` job passes (`needs: test`). The deployed Space mirrors the latest green build on `dev`.

The deploy job builds a **slim deploy tree** containing only what the running app needs (~10 MB of code) and force-syncs it to the Hugging Face Space's git repository:

1. **Stage a slim copy** of the project in `/tmp/deploy/`:
   - The entire [app/](app/) Flask application (routes, services, templates, static assets).
   - Top-level files needed at build/run time: [Dockerfile](Dockerfile), [wsgi.py](wsgi.py), [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), [README.md](README.md).
   - The full [analysis_and_inference/](analysis_and_inference/) source tree, **excluding** tests, `__pycache__`, `*.pkl`, `*.npz`, and `*.csv`. This includes every model's training and inference code so unpickling on the Space can find all custom classes (e.g. `LassoLogisticRegression`).
2. **Recreate empty `outputs/` directories** for each model so the .pkl artifacts uploaded to the Space land in the expected paths.
3. **Clone the existing HF Space repository** into `/tmp/space/` so previously-uploaded model artifacts and `.gitattributes` are preserved.
4. **Overlay the slim tree** with `rsync --delete --exclude='*.pkl' --exclude='.gitattributes'`. Stale files in the Space that are no longer in `app/` get deleted, but the large `.pkl` model files uploaded out-of-band are kept intact across deploys.
5. **Enable Git LFS** locally and register additional binary extensions (`*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.db`, `*.npy`, `*.npz`, `*.ipynb`). Hugging Face's xet storage speaks the LFS protocol, so binary files transparently route through their content-addressable backend instead of being rejected as oversized git blobs.
6. **Commit and push** to the Space's `main` branch. Hugging Face detects the push, rebuilds the Docker image defined by [Dockerfile](Dockerfile), and restarts the container.

### Secrets and one-time setup

The `deploy` job requires two GitHub Actions secrets configured at **Settings → Secrets and variables → Actions**:

- `HF_TOKEN` — a Hugging Face access token with **Write** scope.
- `HF_NAME` — the HF username that owns the target Space.

The model `.pkl` artifacts (~512 MB total) are too large to ship through git on every deploy, so they were uploaded once via `hf upload` and live permanently on the Space. The deploy workflow never touches them — the `*.pkl` exclude rule in step 4 keeps them safe.

### Lifecycle

A typical deploy cycle:

1. Push a commit to `dev` on GitHub.
2. GitHub Actions runs the `test` job (~30 s). If anything fails, the deploy is skipped.
3. The `deploy` job stages, syncs, and pushes the slim tree to Hugging Face.
4. Hugging Face rebuilds the Docker image (≈ 5–10 min) and restarts the container.
5. The change is live at the Space URL above.

---

## Computing EDA Cache

The EDA viewer requires a precomputed cache file. Generate it with:

```bash
uv run python app/services/compute_eda_cache.py
```

This creates `analysis_and_inference/EDA/eda_cache.json`. The app will warn if it's missing, but the feature still works with slower on-demand rendering.

---

## Using the Wikipedia Integration

To fetch and classify comments from a Wikipedia article:

1. Navigate to the article management interface
2. Paste a Wikipedia URL (e.g., `https://en.wikipedia.org/wiki/Climate_change`)
3. The app fetches the talk page, extracts all comments, classifies each one
4. Browse results with toxicity heatmaps and detailed SHAP explanations

---

## Repository Conventions

- All branches follow the team's Git collaboration guidelines (`admin/documentations/`)
- Pull requests require review by the designated reviewer before merging
- Tasks are tracked in the team backlog; priority ranges from P0 (most urgent) to P2
- Sprint schedule: Sprint 1 (3–10 April), Sprint 2 (14–21 April), Sprint 3 (28 April – 4 May)
- Project deadline: 9 May 2026

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. You are free to share and adapt the work for non-commercial purposes with proper attribution. Commercial use is not permitted without explicit permission.

See `LICENSE` for full details.

---

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style and modular design
- Tests pass: `uv run pytest`
- New features include integration tests in `app/tests/`
- Service logic is isolated from route handlers
- Database access goes through the repository layer

---

## Troubleshooting

### Models not found
Ensure `run_all.py` has completed successfully:
```bash
uv run python analysis_and_inference/models/run_all.py
```

### EDA cache missing
Generate it with:
```bash
uv run python app/services/compute_eda_cache.py
```

### Database connection error
Check `SQL_URI` in `app/config.py` and ensure the database file/server is accessible.

### Wikipedia fetching fails
Verify your internet connection and that the Wikipedia URL is valid.


### Disclaimer regarding the use of AI in the making of this project
AI assistants were used throughout the making of this project. In particular, AI agents were useful to generate code, ensure code quality and consistency across all group members. 