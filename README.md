---
title: Wikipedia Toxic Comment Classifier
emoji: 🐷
colorFrom: pink
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# project-bogosort - Toxic Comment Classifier

## Overview

This project is a machine learning web application that automatically detects toxic comments in text. It was developed in Spring 2026 by Team Bogosort for the *Data Structures and Algorithms* course at the Hertie School in Berlin, in partial fulfilment of the Master of Data Science for Public Policy (MDS) programme.

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
├── analysis_and_inference/
│   ├── EDA/
│   │   └── eda_v1_3.ipynb                  # Exploratory data analysis
│   ├── features/
│   │   ├── build_features.py               # DenseFeatureTransformer (~32 features)
│   │   └── tests/
│   ├── models/
│   │   ├── _common.py                      # Shared training, tuning, evaluation utilities
│   │   ├── run_all.py                      # End-to-end training orchestrator (8 steps)
│   │   ├── inference.py                    # Single-comment prediction API for Flask
│   │   ├── split_and_features/
│   │   │   └── prepare_split.py            # Stratified 80/20 train/test split
│   │   ├── baseline/baseline.py            # Dummy stratified classifier
│   │   ├── lasso_log_reg/
│   │   │   ├── lasso.py                    # Training entry point
│   │   │   └── core_logistic_regression_lasso.py  # Custom L1 implementation from scratch
│   │   ├── ridge_log_reg/ridge.py          # sklearn L2 logistic regression
│   │   ├── random_forest/
│   │   │   ├── random_forest.py            # RandomForestClassifier
│   │   │   └── feature_selection.py        # Post-hoc top-5 / top-10 feature ablation
│   │   ├── svm/svm.py                      # LinearSVC
│   │   └── ensemble/ensemble.py            # Soft-vote ensemble (final classifier)
│   └── evaluation_code/
│       ├── evaluator.py                    # Metrics, confusion matrix, ROC/PR/calibration plots
│       ├── error_analysis.py               # FP/FN inspection, error patterns, confidence distribution
│       └── feature_evaluation.py           # SHAP and permutation importance (heavy analysis)
├── app/
│   ├── routes/                             # Flask route handlers
│   ├── templates/                          # HTML templates
│   └── static/                             # CSS / JS assets
├── data/
│   ├── raw/jigsaw-dataset/                 # Raw zipped CSVs from Kaggle
│   └── processed/                          # Cached splits, TF-IDF matrices, BERT embeddings
├── visuals/                                # EDA plots generated during feature exploration
├── admin/
│   ├── meeting_minutes/                    # Team meeting notes (Markdown)
│   └── documentations/                     # Git and collaboration guidelines
└── pyproject.toml
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
| **Identity mentions** | `identity_mention_count` + binary flags for race, gender, sexuality, religion, disability, nationality |

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

## Web Application

The Flask application (under `app/`) wraps the inference layer into an interactive interface. Users submit a comment and receive:
- a binary toxicity verdict (toxic / non-toxic),
- a confidence probability,
- a ranked list of the features that most influenced the decision, with their values and SHAP contributions.

---

## Requirements

- Python >= 3.10
- **Core:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `vaderSentiment`, `matplotlib`
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

## Running the Tests

Each model sub-package has its own test suite:

```bash
uv run pytest
```

---

## Repository Conventions

- All branches follow the team's Git collaboration guidelines (`admin/documentations/`).
- Pull requests require review by the designated reviewer before merging.
- Tasks are tracked in the team backlog; priority ranges from P0 (most urgent) to P2.
- Sprint schedule: Sprint 1 (3–10 April), Sprint 2 (14–21 April), Sprint 3 (28 April – 4 May). Project deadline: 9 May 2026.
