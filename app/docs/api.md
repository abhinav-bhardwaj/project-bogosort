# API Documentation

## Overview

The application is a **Flask** web service. Routes are organised into five blueprints:

| Blueprint | URL Prefix | Purpose |
|---|---|---|
| `main` | `/` | HTML page views |
| `api` | `/api` | JSON REST API |
| `dashboard` | `/dashboard` | Dashboard page views |
| `bogosort_demo` | `/sort-demo` | Bogosort demo (HTML + actions) |
| `eda` | `/eda` | Exploratory data analysis views and data API |

All JSON API endpoints are under `/api`. Request bodies must be `Content-Type: application/json`. Responses are JSON unless noted otherwise.

---

## Common Conventions

### Toxicity thresholds

| Parameter | Default | Range | Rule |
|---|---|---|---|
| `auto_threshold` | `0.75` | `0.0 – 1.0` | Comments at or above this score are automatically banned |
| `manual_threshold` | `0.55` | `0.0 – 1.0` | Comments at or above this score are queued for manual review |

`manual_threshold` must always be ≤ `auto_threshold`.

### Decision values

| Value | Meaning |
|---|---|
| `"auto-ban"` | System-assigned; toxicity ≥ `auto_threshold` |
| `"manual-review"` | System-assigned; `manual_threshold` ≤ toxicity < `auto_threshold` |
| `"manual-ban"` | Moderator-assigned; preserved across threshold recalculations |
| `"none"` | Toxicity < `manual_threshold`; comment passes |

The filter value `"flagged"` (query param only) selects all comments where `is_flagged = true` (i.e. `decision` is `"auto-ban"` or `"manual-review"`).

### Pagination

| Parameter | Type | Default | Max |
|---|---|---|---|
| `limit` | integer | `50` | `200` |
| `offset` | integer | `0` | `1 000 000` |

### Sort options (comments)

| Value | Order |
|---|---|
| `toxicity_desc` *(default)* | Highest toxicity first |
| `toxicity_asc` | Lowest toxicity first |
| `timestamp_desc` | Newest first |
| `timestamp_asc` | Oldest first |
| `decision_asc` | auto-ban → manual-ban → manual-review → none |

### Error responses

All error responses follow this shape:

```json
{ "error": "<human-readable message>" }
```

---

## Endpoints

### Demo inference

#### `POST /api/demo/infer`

Score a single piece of text for toxicity without persisting anything.

**Request body**

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | string | yes | — | Max 10 000 characters |
| `model_name` | string | no | `DEFAULT_MODEL` | Must be a loaded model ID |
| `auto_threshold` | float | no | `0.75` | `0.0 – 1.0` |
| `manual_threshold` | float | no | `0.55` | `0.0 – 1.0`, ≤ `auto_threshold` |

**Response `200`**

```json
{
  "text": "string",
  "model_name": "string",
  "toxicity": 0.0,
  "label": "string",
  "decision": "auto-ban | manual-review | none",
  "auto_threshold": 0.75,
  "manual_threshold": 0.55,
  "inference_ms": 0.0,
  "top_features": [
    { "token": "string", "weight": 0.0 }
  ]
}
```

**Errors**

| Status | Condition |
|---|---|
| `400` | `text` missing or empty |
| `400` | `text` exceeds 10 000 characters |
| `400` | Threshold out of range or `manual_threshold > auto_threshold` |
| `500` | Inference failed |

---

### Models

#### `GET /api/models`

List all available toxicity models.

**Response `200`**

```json
{
  "models": [
    {
      "model_id": "string",
      "model_name": "string",
      "version": "string",
      "metrics": {}
    }
  ]
}
```

**Errors**

| Status | Condition |
|---|---|
| `400` | No models found |
| `500` | General error |

---

#### `GET /api/models/<model_id>/evaluation`

Return evaluation report for a specific model.

**URL parameters**

| Parameter | Description |
|---|---|
| `model_id` | Model identifier |

**Response `200`** — evaluation object with attached image artifacts and sample data.

**Errors**

| Status | Condition |
|---|---|
| `500` | Failed to fetch evaluation |

---

#### `GET /api/evaluation`

Return the evaluation report for one model, defaulting to the first available.

**Query parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | no | Model to fetch; omit to get the first model |

**Response `200`** — same shape as `GET /api/models/<model_id>/evaluation`.

**Errors**

| Status | Condition |
|---|---|
| `500` | Failed to fetch evaluation |

---

#### `GET /api/models/<model_id>/artifacts/<filename>`

Serve a model evaluation image artifact.

**URL parameters**

| Parameter | Description |
|---|---|
| `model_id` | Model identifier |
| `filename` | One of the allowed filenames below |

**Allowed filenames**

- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `calibration.png`
- `feature_importance.png`
- `error_confidence_distribution.png`
- `error_patterns_by_feature.png`

**Response `200`** — binary PNG image (`image/png`).

**Errors**

| Status | Condition |
|---|---|
| `404` | Model not found, or `filename` not in allowlist |
| `500` | File access error |

---

### Articles

#### `GET /api/articles`

List all ingested articles ordered by `created_at` descending.

**Response `200`**

```json
{
  "articles": [
    {
      "id": "string",
      "title": "string",
      "url": "string",
      "summary": "string",
      "model_name": "string",
      "auto_threshold": 0.75,
      "manual_threshold": 0.55,
      "flagged_count": 0,
      "trend": {
        "dates": ["2024-01-01"],
        "avg_toxicity": [0.42]
      },
      "inference_stats": {
        "count": 42,
        "total_ms": 1234.5,
        "avg_ms": 29.4,
        "min_ms": 12.1,
        "max_ms": 80.3
      }
    }
  ]
}
```

**Errors**

| Status | Condition |
|---|---|
| `500` | Failed to retrieve articles |

---

#### `POST /api/articles/ingest`

Fetch a Wikipedia article and its comments, score them for toxicity, and store the result.

**Request body**

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `url` | string | yes | — | Must be a valid Wikipedia article URL |
| `limit` | integer | no | `30` | `1 – 200`; number of comments to fetch |
| `auto_threshold` | float | no | `0.75` | `0.0 – 1.0` |
| `manual_threshold` | float | no | `0.55` | `0.0 – 1.0`, ≤ `auto_threshold` |
| `model_name` | string | no | `DEFAULT_MODEL` | Must be a loaded model ID |

**Response `200`** — article summary object (same shape as one element of `GET /api/articles`).

**Errors**

| Status | Condition |
|---|---|
| `400` | `url` missing, not a Wikipedia URL, or threshold constraint violated |
| `503` | Requested model is unavailable |
| `500` | Unexpected ingestion error |

---

#### `GET /api/articles/<article_id>`

Get a single article, optionally with its comments.

**URL parameters**

| Parameter | Description |
|---|---|
| `article_id` | Article identifier |

**Query parameters**

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `include_comments` | boolean | `true` | Set to `false` to omit the `comments` array |
| `decision` | string | — | Filter comments; see [decision values](#decision-values) + `"flagged"` |
| `sort` | string | `toxicity_desc` | See [sort options](#sort-options-comments) |
| `limit` | integer | `50` | `1 – 200` |
| `offset` | integer | `0` | `0 – 1 000 000` |

**Response `200`**

```json
{
  "id": "string",
  "title": "string",
  "url": "string",
  "summary": "string",
  "model_name": "string",
  "auto_threshold": 0.75,
  "manual_threshold": 0.55,
  "flagged_count": 0,
  "trend": {},
  "inference_stats": {},
  "comments": [
    {
      "id": "string",
      "author": "string",
      "timestamp": "2024-01-01T00:00:00",
      "text": "string",
      "toxicity": 0.0,
      "decision": "string",
      "is_flagged": false,
      "top_features": [{ "token": "string", "weight": 0.0 }],
      "model_version": "string",
      "explain_version": "string",
      "inference_ms": 0.0
    }
  ],
  "comments_total": 0,
  "comments_limit": 50,
  "comments_offset": 0
}
```

**Errors**

| Status | Condition |
|---|---|
| `400` | Invalid query parameters |
| `404` | Article not found |
| `500` | Failed to retrieve article |

---

#### `PUT /api/articles/<article_id>/thresholds`

Update the toxicity thresholds for an article. Re-assigns `decision` and `is_flagged` for all system-scored comments; moderator `"manual-ban"` decisions are preserved.

**URL parameters**

| Parameter | Description |
|---|---|
| `article_id` | Article identifier |

**Request body**

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `auto_threshold` | float | no | `0.75` | `0.0 – 1.0` |
| `manual_threshold` | float | no | `0.55` | `0.0 – 1.0`, ≤ `auto_threshold` |

**Response `200`**

```json
{ "status": "ok" }
```

**Errors**

| Status | Condition |
|---|---|
| `400` | Threshold out of range or `manual_threshold > auto_threshold` |
| `500` | Failed to update thresholds |

---

### Comments

#### `GET /api/articles/<article_id>/comments`

List comments for an article with filtering, sorting, and pagination.

**URL parameters**

| Parameter | Description |
|---|---|
| `article_id` | Article identifier |

**Query parameters**

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `decision` | string | — | Filter; see [decision values](#decision-values) + `"flagged"` |
| `sort` | string | `toxicity_desc` | See [sort options](#sort-options-comments) |
| `limit` | integer | `50` | `1 – 200` |
| `offset` | integer | `0` | `0 – 1 000 000` |

**Response `200`**

```json
{
  "comments": [ /* comment objects */ ],
  "total": 0,
  "limit": 50,
  "offset": 0
}
```

**Errors**

| Status | Condition |
|---|---|
| `400` | Invalid query parameters |
| `500` | Failed to retrieve comments |

---

#### `GET /api/articles/<article_id>/comments/<comment_id>`

Get a single comment together with its parent article metadata.

**URL parameters**

| Parameter | Description |
|---|---|
| `article_id` | Article identifier |
| `comment_id` | Comment identifier (MD5 hash) |

**Response `200`**

```json
{
  "article": {
    "id": "string",
    "title": "string",
    "url": "string",
    "summary": "string",
    "auto_threshold": 0.75,
    "manual_threshold": 0.55,
    "model_name": "string"
  },
  "comment": { /* comment object */ }
}
```

**Errors**

| Status | Condition |
|---|---|
| `404` | Article or comment not found |
| `500` | Failed to retrieve comment |

---

#### `PATCH /api/articles/<article_id>/comments/<comment_id>`

Override the moderation decision for a comment (moderator action).

**URL parameters**

| Parameter | Description |
|---|---|
| `article_id` | Article identifier |
| `comment_id` | Comment identifier |

**Request body**

| Field | Type | Required | Notes |
|---|---|---|---|
| `decision` | string | yes | One of `"auto-ban"`, `"manual-ban"`, `"manual-review"`, `"none"` |

**Response `200`**

```json
{
  "status": "ok",
  "decision": "string"
}
```

**Errors**

| Status | Condition |
|---|---|
| `400` | `decision` missing or not a valid value |
| `500` | Failed to update decision |

---

### EDA (Exploratory Data Analysis)

#### `GET /eda/api/data`

Return the full precomputed EDA dataset.

**Response `200`**

```json
{
  "overview": {},
  "top_features": []
}
```

**Errors**

| Status | Condition |
|---|---|
| `503` | EDA cache not loaded |

---

#### `GET /eda/api/overview`

Return only the overview statistics section of the EDA data.

**Response `200`** — overview statistics object.

**Errors**

| Status | Condition |
|---|---|
| `503` | EDA cache not loaded |

---

#### `GET /eda/api/top-features`

Return only the top-features section of the EDA data.

**Response `200`** — array of feature objects.

**Errors**

| Status | Condition |
|---|---|
| `503` | EDA cache not loaded |

---

## Page Routes (HTML)

These routes return rendered HTML and are not intended for programmatic use.

| Method | Path | Description |
|---|---|---|
| GET | `/` | Landing page |
| GET | `/analyze/` | Article ingestion UI |
| GET | `/about/` | About / team page |
| GET | `/articles/<article_id>/` | Article detail page |
| GET | `/articles/<article_id>/comments/<comment_id>/` | Comment detail page |
| GET | `/demo/` | Toxicity demo UI |
| GET | `/dashboard/` | Moderation dashboard |
| GET | `/dashboard/nerdy/` | Extended statistics dashboard |
| GET | `/eda/` | EDA dashboard |
| GET / POST | `/sort-demo/` | Bogosort demo — GET shows current state, POST starts a new sort |
| POST | `/sort-demo/stop` | Stop an in-progress sort |
| GET | `/sort-demo/reset` | Reset the sort session |
