# Database Schema

## Overview

The application uses **SQLite** with a custom repository pattern (no ORM). The database is initialised on application startup via `initialize_schema()` in [`app/db/article_repository.py`](../app/db/article_repository.py).

| Property | Value |
|---|---|
| Engine | SQLite |
| Default path (dev) | `app/db/articles.db` |
| Default path (test) | `app/db/test_articles.db` |
| Configuration | `SQL_URI` environment variable (see [`app/config.py`](../app/config.py)) |
| Schema migrations | Additive `ALTER TABLE` via `_ensure_columns()` — no versioning system |

---

## Entity-Relationship Diagram

```
articles
────────────────────────────────────────────
 id (PK, TEXT)
 title
 url
 summary
 created_at
 model_name
 auto_threshold
 manual_threshold
 flagged_count
 trend_json          ──► JSON blob
 inference_stats_json ─► JSON blob
        │
        │ 1 : N
        ▼
comments
────────────────────────────────────────────
 id (PK, TEXT)
 article_id (FK → articles.id)
 author
 timestamp
 text
 toxicity
 decision
 is_flagged
 top_features_json  ──► JSON blob
 model_version
 explain_version
 inference_ms
```

---

## Tables

### `articles`

Stores one row per Wikipedia article that has been analysed for comment toxicity.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | TEXT | PRIMARY KEY | Wikipedia article identifier (e.g. page slug or hash) |
| `title` | TEXT | NOT NULL | Article title |
| `url` | TEXT | NOT NULL | Full Wikipedia URL |
| `summary` | TEXT | NOT NULL | Short article summary/extract |
| `created_at` | TEXT | NOT NULL | ISO-8601 timestamp of when the article was ingested |
| `model_name` | TEXT | NOT NULL | Name of the toxicity model used when the article was processed |
| `auto_threshold` | REAL | NOT NULL | Toxicity score at or above which a comment is automatically banned |
| `manual_threshold` | REAL | NOT NULL | Toxicity score at or above which a comment is flagged for manual review |
| `flagged_count` | INTEGER | NOT NULL | Cached count of comments with `is_flagged = 1` |
| `trend_json` | TEXT | NOT NULL | JSON object — time-series toxicity trend data |
| `inference_stats_json` | TEXT | NOT NULL DEFAULT `'{}'` | JSON object — aggregate model inference statistics for the article |

**`trend_json` structure** (example):
```json
{
  "dates": ["2024-01-01", "2024-01-02"],
  "avg_toxicity": [0.42, 0.61]
}
```

**`inference_stats_json` structure** (example):
```json
{
  "total_ms": 1234.5,
  "count": 42,
  "avg_ms": 29.4
}
```

---

### `comments`

Stores one row per comment associated with an article, including its toxicity score and moderation decision.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | TEXT | PRIMARY KEY | Unique comment identifier |
| `article_id` | TEXT | NOT NULL, FK → `articles.id` | Parent article |
| `author` | TEXT | NOT NULL DEFAULT `'unsigned'` | Comment author name; defaults to `'unsigned'` when anonymous |
| `timestamp` | TEXT | NOT NULL | ISO-8601 timestamp of the original comment |
| `text` | TEXT | NOT NULL | Raw comment body |
| `toxicity` | REAL | NOT NULL | Toxicity score in `[0.0, 1.0]` |
| `decision` | TEXT | NOT NULL | Moderation outcome — see values below |
| `is_flagged` | INTEGER | NOT NULL DEFAULT `0` | Boolean (`0`/`1`): `1` when `toxicity >= manual_threshold` |
| `top_features_json` | TEXT | NOT NULL | JSON array — explainability features driving the toxicity score |
| `model_version` | TEXT | NOT NULL DEFAULT `''` | Version string of the toxicity model that scored this comment |
| `explain_version` | TEXT | NOT NULL DEFAULT `''` | Version string of the explanation generator |
| `inference_ms` | REAL | NOT NULL DEFAULT `0` | Time taken (milliseconds) for the model inference |

#### `decision` values

| Value | Meaning |
|---|---|
| `'auto-ban'` | `toxicity >= auto_threshold` — system automatically bans the comment |
| `'manual-review'` | `manual_threshold <= toxicity < auto_threshold` — queued for human review |
| `'manual-ban'` | Moderator has explicitly banned the comment; preserved across threshold recalculations |
| `'none'` | `toxicity < manual_threshold` — comment passes moderation |

**`top_features_json` structure** (example):
```json
[
  {"token": "idiot", "weight": 0.38},
  {"token": "hate", "weight": 0.21}
]
```

---

## Indexes

| Index name | Table | Columns | Purpose |
|---|---|---|---|
| `idx_comments_article` | `comments` | `(article_id)` | Efficient lookup of all comments for an article |
| `idx_comments_article_decision_toxicity` | `comments` | `(article_id, decision, toxicity)` | Filtered + sorted comment listings by decision and toxicity |
| `idx_comments_article_flagged` | `comments` | `(article_id, is_flagged)` | Fast count/filter of flagged comments per article |

---

## Key Operations

| Operation | Function | Notes |
|---|---|---|
| Upsert article + comments | `upsert_article(article, comments)` | Replaces the article row and **deletes + re-inserts** all its comments atomically |
| List articles | `list_articles()` | Returns all articles ordered by `created_at DESC` |
| Get article (with comments) | `get_article(article_id, ...)` | Supports pagination (`limit`/`offset`), filter by `decision`, and sort |
| Recalculate thresholds | `update_thresholds(article_id, auto, manual)` | Re-assigns `decision`/`is_flagged` for all system-decided comments; preserves `manual-ban` |
| Override comment decision | `update_comment_decision(comment_id, decision)` | Moderator action — sets decision directly |
| Update explainability | `update_comment_explanation(comment_id, features, version)` | Stores LIME/SHAP features after async explanation run |

---

## Schema Evolution

New columns are added without a migration framework using `_ensure_columns()`, which issues `ALTER TABLE … ADD COLUMN` only when the column is absent. Columns added this way:

| Table | Column | Added reason |
|---|---|---|
| `articles` | `inference_stats_json` | Added to track per-article aggregate inference timing |
| `comments` | `is_flagged` | Added to cache flagged state for index-accelerated queries |
| `comments` | `model_version` | Added to track which model version produced the score |
| `comments` | `explain_version` | Added to track which explainer version produced features |
| `comments` | `inference_ms` | Added to record per-comment inference latency |
