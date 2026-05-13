# Application Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Running the Application](#3-running-the-application)
4. [Configuration](#4-configuration)
5. [Application Factory](#5-application-factory)
6. [Routes Layer](#6-routes-layer)
7. [Service Layer](#7-service-layer)
8. [Database Layer](#8-database-layer)
9. [Dependencies](#9-dependencies)
10. [Testing](#10-testing)

---

## 1. Overview

This is a Flask web application for **Wikipedia comment toxicity moderation**. Given a Wikipedia article URL, it fetches comments from the article's talk page, scores each comment with a pre-trained toxicity model, assigns moderation decisions, and stores the results. A browser UI lets moderators review, filter, and override those decisions.

A secondary feature is a visual **Bogosort vs MergeSort demo** that animates algorithm behaviour on a dataset of toxic words.

---

## 2. Architecture

```
browser / API client
        │
        ▼
┌───────────────────────────────────┐
│           Flask Routes            │
│  main · api · dashboard ·         │
│  bogosort_demo · eda              │
└──────────────┬────────────────────┘
               │ calls
               ▼
┌───────────────────────────────────┐
│          Service Layer            │
│  article_service                  │
│  toxicity_service                 │
│  wiki_client / TalkFetcher        │
│  evaluation_service               │
│  eda_service                      │
│  sorting_service                  │
│  session_manager                  │
└──────┬──────────┬─────────────────┘
       │          │ external I/O
       ▼          ▼
┌──────────┐  ┌─────────────────────┐
│  DB      │  │  Wikipedia API      │
│  layer   │  │  Model inference    │
│ (SQLite) │  │  Filesystem (plots, │
└──────────┘  │  GIFs, artifacts)   │
              └─────────────────────┘
```

**Request flow for article ingestion:**

1. `POST /api/articles/ingest` → `api.py`
2. `api.py` → `article_service.ingest_article()`
3. `article_service` → `wiki_client.fetch_wikipedia_metadata()` (MediaWiki API)
4. `article_service` → `wiki_client.fetch_talk_page_comments()` → `WikipediaTalkFetcher`
5. Per comment → `toxicity_service.score_comment()` → ML model inference
6. `article_service` → `article_repository.upsert_article()` → SQLite

---

## 3. Running the Application

**Development**

```bash
python run.py
# or
flask --app run run --debug
```

**Production (WSGI)**

```bash
gunicorn wsgi:app
```

`wsgi.py` exposes the `app` object created by the application factory.

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | Selects config class: `development`, `testing`, `production` |
| `SQL_URI` | `sqlite:///app/db/articles.db` | Database URI (SQLite only) |
| `SECRET_KEY` | `jigsaw_secret_key` | Flask session signing key (**must be overridden in production**) |
| `TEST_SQL_URI` | `sqlite:///app/db/test_articles.db` | Database URI used in testing |

**EDA cache**: must be pre-generated before the `/eda` routes work:

```bash
python app/services/compute_eda_cache.py
# writes to analysis_and_inference/EDA/eda_cache.json
```

---

## 4. Configuration

**File:** [`app/config.py`](../app/config.py)

Three config classes, all extending `Config`:

| Class | `FLASK_ENV` value | `DEBUG` | `TESTING` | Notes |
|---|---|---|---|---|
| `DevelopmentConfig` | `development` | `True` | `False` | Default |
| `TestingConfig` | `testing` | `False` | `True` | Uses `TEST_SQL_URI` |
| `ProductionConfig` | `production` | `False` | `False` | `SECRET_KEY` must come from env |

---

## 5. Application Factory

**File:** [`app/__init__.py`](../app/__init__.py)

`create_app(config_name=None)` is the Flask application factory. It:

1. Loads the appropriate config class from `FLASK_ENV`
2. Initialises the database via `init_db(SQL_URI)` (creates schema if absent)
3. Attempts to load the EDA cache from `analysis_and_inference/EDA/eda_cache.json`; logs a warning if missing but does not abort startup
4. Registers all five blueprints (see §6)
5. Registers `404` and `500` error handlers: JSON responses for `/api/*` paths, HTML error page otherwise

---

## 6. Routes Layer

All blueprints live in `app/routes/`. They are intentionally thin: they parse and validate HTTP inputs, call a service function, and return the result. No business logic lives here.

### 6.1 `main` (prefix `/`)

**File:** [`app/routes/main.py`](../app/routes/main.py)

Page-only blueprint. All routes return rendered HTML. Data is fetched client-side by the API layer.

| Path | Template |
|---|---|
| `/` | `landing.html` |
| `/analyze/` | `index.html` |
| `/about/` | `about.html` (injects `team_bio.json`) |
| `/articles/<article_id>/` | `article.html` |
| `/articles/<article_id>/comments/<comment_id>/` | `comment.html` |
| `/demo/` | `demo.html` |

### 6.2 `api` (prefix `/api`)

**File:** [`app/routes/api.py`](../app/routes/api.py)

All JSON. See [API documentation](api.md) for full endpoint details.

**Input parsing helpers** (private):

| Helper | Purpose |
|---|---|
| `_parse_int(value, default, min, max, field)` | Parses and range-checks integer query/body params |
| `_parse_float(value, default, min, max, field)` | Parses and range-checks float params |
| `_parse_sort(value)` | Validates against `VALID_SORTS` set |
| `_parse_decision(value)` | Validates against `VALID_DECISIONS` set |
| `_attach_artifacts(evaluation, model_id)` | Merges artifact image URLs into an evaluation payload |

**Constants:**

```python
MIN_LIMIT = 1
MAX_LIMIT = 200
DEFAULT_LIMIT = 30          # article ingest
DEFAULT_COMMENT_LIMIT = 50  # comment listing
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0
DEFAULT_AUTO_THRESHOLD = 0.75
DEFAULT_MANUAL_THRESHOLD = 0.55
MAX_OFFSET = 1_000_000
VALID_DECISIONS = {"auto-ban", "manual-ban", "manual-review", "none", "flagged"}
VALID_SORTS = {"toxicity_desc", "toxicity_asc", "timestamp_desc", "timestamp_asc", "decision_asc"}
```

### 6.3 `dashboard` (prefix `/dashboard`)

**File:** [`app/routes/dashboard.py`](../app/routes/dashboard.py)

Only one active route:

| Path | Template |
|---|---|
| `/dashboard/nerdy/` | `dashboard_nerdy.html` |

> `/dashboard/` and its `dashboard_page` handler are commented out. The `dashboard.html` template exists but is not currently served.

### 6.4 `bogosort_demo` (prefix `/sort-demo`)

**File:** [`app/routes/bogosort.py`](../app/routes/bogosort.py)

Interactive sorting demo. Uses Flask's `session` (cookie-based) to hold a `bogosort_session_id` per browser. The actual sort runs in a background `threading.Thread` so the web server thread stays responsive.

| Path | Method | Behaviour |
|---|---|---|
| `/sort-demo/` | GET | Renders current sort state (`null` / `running` / `done` / `error`) |
| `/sort-demo/` | POST | Starts a new sort (`algorithm`, `seed` from form data); redirects to GET |
| `/sort-demo/stop` | POST | Sets `stop_flag` on the session; background thread checks this flag |
| `/sort-demo/reset` | GET | Deletes the session state; redirects to GET |

**Session state fields:**

| Field | Type | Meaning |
|---|---|---|
| `state` | `None \| "running" \| "done" \| "error"` | Current sort phase |
| `algorithm` | `"bogosort" \| "mergesort"` | Algorithm in use |
| `seed` | string | Seed string used for reproducibility |
| `final_iteration` | int | Number of iterations completed |
| `sorted` | bool | Whether the array is actually sorted at completion |
| `error` | string | Error message if `state == "error"` |
| `stop_flag` | bool | Cooperative stop signal |

### 6.5 `eda` (prefix `/eda`)

**File:** [`app/routes/eda.py`](../app/routes/eda.py)

Serves pre-computed EDA statistics. The cache is loaded lazily on first request if not already loaded at startup. All endpoints return `503` when the cache is unavailable.

| Path | Returns |
|---|---|
| `/eda/` | HTML dashboard (`eda.html`) |
| `/eda/api/data` | Full EDA JSON object |
| `/eda/api/overview` | `overview` section only |
| `/eda/api/top-features` | `top_features` array only |

---

## 7. Service Layer

All services live in `app/services/`. They contain all business logic and are independent of Flask (no `request`/`session` imports).

### 7.1 `article_service`

**File:** [`app/services/article_service.py`](../app/services/article_service.py)

Orchestrates the full article ingestion pipeline and wraps all repository reads/writes.

**Key functions:**

| Function | Description |
|---|---|
| `ingest_article(url, limit, auto_threshold, manual_threshold, model_name)` | Full pipeline: validate URL → fetch metadata → check model → fetch talk comments → score each → build trend/stats → upsert to DB |
| `list_articles()` | Thin wrapper over repository |
| `get_article(article_id, ...)` | Thin wrapper; supports pagination, filtering, sorting |
| `list_comments(article_id, ...)` | Wraps repository, returns `{comments, total, limit, offset}` |
| `update_thresholds(article_id, auto, manual)` | Delegates to repository |
| `update_comment_decision(article_id, comment_id, decision)` | Delegates to repository |
| `get_comment_detail(article_id, comment_id)` | Fetches comment; triggers on-demand explanation generation if `top_features` is empty or `explain_version` is stale |

**Article ID generation:** `slugify_title(title)`, which produces a lowercase alphanumeric slug joined with `-`.

**Comment ID generation:** MD5 of `"{timestamp}-{author}-{text}"`.

**Trend data structure** (stored as `trend_json`):

```python
{
  "dates": [date_key, ...],          # ISO date from comment timestamp
  "scores": [cumulative_toxicity, ...],  # running sum, not per-comment
  "threshold": manual_threshold
}
```

**`inference_stats` structure:**

```python
{
  "count": int,       # comments with inference_ms > 0
  "total_ms": float,
  "avg_ms": float,
  "min_ms": float,
  "max_ms": float
}
```

---

### 7.2 `toxicity_service`

**File:** [`app/services/toxicity_service.py`](../app/services/toxicity_service.py)

Thin wrapper over the ML inference pipeline at `analysis_and_inference.models.inference`.

**Constants:**

| Name | Value | Meaning |
|---|---|---|
| `DEFAULT_MODEL` | `"ensemble"` | Model used when none is specified |
| `EXPLAIN_VERSION` | `"v1"` | Version tag written to `explain_version` column |

**Functions:**

| Function | Description |
|---|---|
| `check_model_available(model_name)` | Calls `_load()` to verify model files exist. Raises `RuntimeError` if not. Call before a batch scoring loop. |
| `score_comment(text, model_name, explain)` | Times inference with `perf_counter`. Returns `{label, probability, top_features, explain_version, inference_ms}`. On error returns all-zero safe defaults. |

**`score_comment` return shape:**

```python
{
  "label": int,           # 0 = non-toxic, 1 = toxic
  "probability": float,   # toxicity score in [0.0, 1.0]
  "top_features": list,   # SHAP/LIME features if explain=True, else []
  "explain_version": str, # "v1" if explain=True, else ""
  "inference_ms": float
}
```

---

### 7.3 `wiki_client`

**File:** [`app/services/wiki_client.py`](../app/services/wiki_client.py)

Wikipedia API integration.

**Functions:**

| Function | Description |
|---|---|
| `is_allowed_wikipedia_url(url)` | Returns `True` only for `http(s)://*.wikipedia.org/wiki/…` URLs |
| `parse_wiki_title_from_url(url)` | Extracts and URL-decodes the article title from the path |
| `fetch_wikipedia_metadata(title)` | Calls MediaWiki `query` API (props: `extracts`, `info`). Returns `{title, summary, url}`. |
| `fetch_talk_page_comments(title, limit)` | Instantiates `WikipediaTalkFetcher`, calls `get_all_comments()`, maps `WikiComment` objects to plain dicts. Applies `limit` slice. |

---

### 7.4 `WikipediaTalkFetcher`

**File:** [`app/services/wikipedia_talk_fetcher.py`](../app/services/wikipedia_talk_fetcher.py)

Parses raw wikitext from Wikipedia talk pages into structured `WikiComment` objects.

**`WikiComment` dataclass fields:**

| Field | Type | Description |
|---|---|---|
| `author` | str | Wikipedia username, or `""` for unsigned comments |
| `timestamp` | `Optional[datetime]` | Parsed from `HH:MM, DD Month YYYY (UTC)` format |
| `text` | str | Comment body (before the signature) |
| `section` | str | Talk page section heading |
| `level` | int | Indentation depth (`:`, `::`, `*` prefixes; min 1) |
| `raw_wikitext` | str | Original unparsed line |

**Key methods:**

| Method | Description |
|---|---|
| `get_all_comments(page_title, parse_method)` | Top-level entry point. `parse_method` is `'wikitext'` (default) or `'html'`. |
| `get_talk_page_wikitext(page_title)` | Fetches raw wikitext via MediaWiki `revisions` API. Handles both old (`*`) and new (`slots.main.*`) response formats. |
| `parse_wikitext_comments(wikitext, section_title)` | Line-by-line parser. Identifies signed comments by `[[User:…]]` links and timestamp patterns. Buffers unsigned lines. |
| `_get_with_backoff(url, params, max_retries)` | HTTP GET with exponential backoff + jitter on `429` or network errors. Default 5 retries. |
| `_parse_timestamp(timestamp_str)` | Parses Wikipedia's `HH:MM, DD Month YYYY (UTC)` into `datetime`. |
| `get_recent_changes(page_title, limit)` | Fetches revision history for the talk page (not used in the main pipeline). |

**Module-level helpers** (standalone use):

- `fetch_comments_simple(page_title, language)`: one-liner returning list of dicts
- `fetch_and_export_comments(page_title, output_format, output_file, language)`: fetch and write to JSON, CSV, or TXT

---

### 7.5 `evaluation_service`

**File:** [`app/services/evaluation_service.py`](../app/services/evaluation_service.py)

Loads model evaluation metadata from `app/data/model_evaluations.json` and resolves paths to evaluation artifact files. Path traversal is prevented by allowlisting filenames and validating that resolved paths stay within `analysis_and_inference/models/`.

`app/data/model_evaluations.json` is generated by `analysis_and_inference/models/generate_evaluations_json.py`, which is called automatically at the end of `run_all.py` or can be run standalone:

```bash
uv run python analysis_and_inference/models/generate_evaluations_json.py
```

**Supported model IDs** (`MODEL_DIR_ALIASES`):

`lasso_log_reg`, `random_forest`, `ridge_log_reg`, `svm`, `ensemble`

**Artifact files** (`ALLOWED_ARTIFACT_FILES`):

`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `calibration.png`, `feature_importance.png`, `error_confidence_distribution.png`, `error_patterns_by_feature.png`

**CSV sample files** (up to 5 rows each):

`false_positives.csv`, `false_negatives.csv`, `error_patterns_by_feature.csv`

**Artifact directory:** `analysis_and_inference/models/<model_id>/outputs/evaluation/`

**Functions:**

| Function | Description |
|---|---|
| `load_all_evaluations()` | Reads `data/model_evaluations.json`. Returns `{"models": []}` on any error. |
| `get_model_evaluation(model_id)` | Returns matching model dict, or first model if `model_id=None`, or `{}` if not found. |
| `get_model_version(model_id)` | Returns `evaluation.get("version", "")`. |
| `get_model_artifacts(model_id)` | Returns `{"images": {key: url}, "samples": {key: rows}}` for all present files. |
| `resolve_artifact_dir(model_id)` | Returns `Path` to evaluation output directory, or `None` if unsafe/missing. |
| `is_safe_model_id(model_id)` | Validates against `^[A-Za-z0-9_-]+$` to prevent path injection. |

---

### 7.6 `eda_service`

**File:** [`app/services/eda_service.py`](../app/services/eda_service.py)

Module-level in-memory cache (`_EDA_CACHE`) for pre-computed EDA data.

**Expected cache keys** (warns if missing, does not error):

`missing_values`, `duplicate_rows`, `dtype_distribution`, `target_distribution`, `imbalance_ratio`, `split_report`, `feature_occurrence`, `feature_target_correlation`, `feature_means_by_class`, `modeling_readiness`

**Functions:**

| Function | Description |
|---|---|
| `load_eda_cache(cache_path)` | Reads JSON, validates structure, stores in `_EDA_CACHE`. Raises on file/parse errors. |
| `get_eda_data()` | Returns full cache dict. Raises `RuntimeError` if not loaded. |
| `get_eda_section(section_name)` | Returns one named section or `None`. |
| `is_eda_cache_loaded()` | Returns `bool`. |
| `clear_eda_cache()` | Sets `_EDA_CACHE = None`. Used in tests. |

---

### 7.7 `sorting_service`

**File:** [`app/services/sorting_service.py`](../app/services/sorting_service.py)

All sorting demo logic. Stateless; all methods are `@staticmethod`.

**Data source:** `app/data/top_toxic_words.npy`, a NumPy array of `(word, count)` tuples. Top 20 by count are used; order is shuffled with the provided seed.

| Method | Description |
|---|---|
| `load_shuffled_toxic_words(filename, seed, top_n)` | Loads `.npy`, takes top 20 by count, shuffles with `random.Random(seed)`. Returns `(words, counts)`. |
| `is_sorted(counts)` | Returns `True` if counts are non-increasing (descending sort). |
| `save_distribution_plot(words, counts, filename)` | Saves a bar chart PNG. |
| `bogosort_snapshots(words, counts, max_iterations, seed, stop_flag)` | Runs bogosort up to `max_iterations` (route passes 1000). Captures `(state, iteration)` snapshots. Checks `stop_flag["stop"]` every iteration. |
| `mergesort_snapshots(words, counts, seed, stop_flag)` | Recursive merge sort, capturing a snapshot after each merge step. |
| `save_sort_animation(snapshots, filename, title)` | Renders each snapshot as a matplotlib bar chart frame and saves as an animated GIF via `imageio`. |

---

### 7.8 `session_manager`

**File:** [`app/services/session_manager.py`](../app/services/session_manager.py)

In-memory store for bogosort sort sessions, keyed by a UUID stored in the Flask cookie session. Sessions expire after inactivity.

```python
SessionManager(timeout_minutes=30)  # instantiated once at module level in bogosort.py
```

**Session dict fields:**

| Field | Default | Description |
|---|---|---|
| `state` | `None` | `None / "running" / "done" / "error"` |
| `final_iteration` | `0` | Iterations at completion |
| `sorted` | `False` | Whether result is sorted |
| `error` | `None` | Error string |
| `stop_flag` | `False` | Cooperative stop signal for background thread |
| `algorithm` | `None` | `"bogosort"` or `"mergesort"` |
| `last_access` | `datetime.now()` | Updated on every access |

**Methods:**

| Method | Description |
|---|---|
| `get_session(session_id)` | Returns session if not expired, else deletes and returns `None` |
| `create_session(session_id)` | Creates session with default state |
| `get_or_create_session(session_id)` | Convenience wrapper |
| `update_session(session_id, updates)` | Merges dict and bumps `last_access` |
| `cleanup_expired_sessions()` | Deletes all sessions past timeout (must be called manually; not automatic) |

---

## 8. Database Layer

**Files:** [`app/db/article_repository.py`](../app/db/article_repository.py), [`app/db/queries.py`](../app/db/queries.py), [`app/db/__init__.py`](../app/db/__init__.py)

See [DB Schema documentation](db_schema.md) for the full schema reference.

**Layer structure:**

| File | Role |
|---|---|
| `article_repository.py` | All SQL, serialisation, schema init |
| `queries.py` | Re-exports the repository's public functions as the `db` package API |
| `__init__.py` | `init_db(sql_uri)`: called by the app factory; calls `setup_database()` then `initialize_schema()` |

**Connection:** `sqlite3.connect()` with `row_factory = sqlite3.Row`. Each function opens and closes its own connection via context manager. No connection pool.

**Schema management:** `initialize_schema()` is called at startup and defensively at the start of each write function. `_ensure_columns()` adds missing columns via `ALTER TABLE` for backward compatibility.

---

## 9. Dependencies

Declared in [`pyproject.toml`](../pyproject.toml). Requires Python ≥ 3.10.

| Package | Use |
|---|---|
| `flask` | Web framework |
| `numpy` | Toxic words array, sorting data |
| `pandas` | EDA preprocessing |
| `scipy` | EDA statistics |
| `scikit-learn` | ML model training / inference pipeline |
| `vaderSentiment` | Sentiment features for toxicity model |
| `matplotlib` | Distribution plots, GIF frame rendering |
| `imageio` | Animated GIF generation |
| `shap` | Explainability features (`top_features`) |
| `requests` | MediaWiki API calls |
| `pytest` | Testing (dev dependency) |

> BERT embeddings (`torch`, `transformers`) are optional and not installed by default (~2 GB). Install manually if `BertTransformer` is needed.

---

## 10. Testing

Tests live in `app/tests/`. Run with:

```bash
pytest
```

**Test layout:**

| Path | Covers |
|---|---|
| `tests/test_app.py` | App factory smoke tests |
| `tests/test_config.py` | Config class values |
| `tests/test_queries.py` | DB query functions |
| `tests/test_routes/test_api.py` | API endpoint integration tests |
| `tests/test_routes/test_main.py` | Main page routes |
| `tests/test_routes/test_dashboard.py` | Dashboard routes |
| `tests/test_services/test_article_service.py` | Ingestion pipeline |
| `tests/test_services/test_toxicity_service.py` | Scoring wrapper |
| `tests/test_services/test_wiki_client.py` | URL validation, metadata fetch |
| `tests/test_services/test_evaluation_service.py` | Artifact resolution, model lookup |
| `tests/test_services/test_sorting_service.py` | Sort algorithms, animation |
| `tests/test_services/test_session_manager.py` | Session expiry, CRUD |

The test config (`TestingConfig`) points to a separate `test_articles.db` so tests never touch the development database.
