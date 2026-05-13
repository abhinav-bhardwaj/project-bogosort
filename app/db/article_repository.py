"""
article_repository.py - article database and database access layer for storing and retrieving articles and comments

This module implements the application's persistence layer for storing,
retrieving, updating, and serializing Wikipedia articles and their associated
toxicity-scored comments using SQLite.

The repository acts as the primary interface between the service layer and
the database, encapsulating all SQL operations, schema management, and
serialization logic.

Used by:
- article_service.py
- moderation workflows
- API route handlers
- dashboard views
- toxicity explanation services
"""

import json
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = None

def setup_database(sql_uri):
    """Initialize the database path from SQL_URI config."""
    global _DB_PATH
    if sql_uri.startswith("sqlite:///"):
        _DB_PATH = Path(sql_uri.replace("sqlite:///", ""))
    else:
        raise ValueError(f"Unsupported database URI: {sql_uri}")

def get_db_path():
    """Get the configured database path."""
    if _DB_PATH is None:
        return Path("app/db/articles.db")
    return _DB_PATH

def get_connection():
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_columns(conn, table, columns):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

def initialize_schema():
    try:
        with get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    auto_threshold REAL NOT NULL,
                    manual_threshold REAL NOT NULL,
                    flagged_count INTEGER NOT NULL,
                    trend_json TEXT NOT NULL,
                    inference_stats_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comments (
                    id TEXT PRIMARY KEY,
                    article_id TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT 'unsigned',
                    timestamp TEXT NOT NULL,
                    text TEXT NOT NULL,
                    toxicity REAL NOT NULL,
                    decision TEXT NOT NULL,
                    is_flagged INTEGER NOT NULL DEFAULT 0,
                    top_features_json TEXT NOT NULL,
                    model_version TEXT NOT NULL DEFAULT '',
                    explain_version TEXT NOT NULL DEFAULT '',
                    inference_ms REAL NOT NULL DEFAULT 0,
                    FOREIGN KEY(article_id) REFERENCES articles(id)
                )
                """
            )
            _ensure_columns(
                conn,
                "articles",
                [
                    ("inference_stats_json", "TEXT NOT NULL DEFAULT '{}'"),
                ],
            )
            _ensure_columns(
                conn,
                "comments",
                [
                    ("is_flagged", "INTEGER NOT NULL DEFAULT 0"),
                    ("model_version", "TEXT NOT NULL DEFAULT ''"),
                    ("explain_version", "TEXT NOT NULL DEFAULT ''"),
                    ("inference_ms", "REAL NOT NULL DEFAULT 0"),
                ],
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_comments_article ON comments(article_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_comments_article_decision_toxicity ON comments(article_id, decision, toxicity)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_comments_article_flagged ON comments(article_id, is_flagged)"
            )
            conn.commit()
    except Exception as exc:
        logger.error(f"Failed to initialize database schema: {exc}", exc_info=True)
        raise RuntimeError(f"Database initialization failed: {exc}") from exc

def serialize_article_summary(row):
    try:
        trend = json.loads(row["trend_json"]) if row["trend_json"] else {}
    except json.JSONDecodeError as exc:
        logger.error(f"Corrupted trend JSON for article {row['id']}: {exc}")
        trend = {}

    try:
        inference_stats = json.loads(row["inference_stats_json"]) if row["inference_stats_json"] else {}
    except (json.JSONDecodeError, IndexError):
        inference_stats = {}

    return {
        "id": row["id"],
        "title": row["title"],
        "url": row["url"],
        "summary": row["summary"],
        "model_name": row["model_name"],
        "auto_threshold": row["auto_threshold"],
        "manual_threshold": row["manual_threshold"],
        "flagged_count": row["flagged_count"],
        "trend": trend,
        "inference_stats": inference_stats,
    }

def serialize_comment(row):
    try:
        top_features = json.loads(row["top_features_json"]) if row["top_features_json"] else []
    except json.JSONDecodeError as exc:
        logger.error(f"Corrupted top_features JSON for comment {row['id']}: {exc}")
        top_features = []

    return {
        "id": row["id"],
        "author": row["author"] or "unsigned",
        "timestamp": row["timestamp"],
        "text": row["text"],
        "toxicity": row["toxicity"],
        "decision": row["decision"],
        "is_flagged": bool(row["is_flagged"]),
        "top_features": top_features,
        "model_version": row["model_version"],
        "explain_version": row["explain_version"],
        "inference_ms": row["inference_ms"] if row["inference_ms"] else 0.0,
    }

def upsert_article(article, comments):
    initialize_schema()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO articles
            (id, title, url, summary, created_at, model_name, auto_threshold, manual_threshold, flagged_count, trend_json, inference_stats_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article["id"],
                article["title"],
                article["url"],
                article["summary"],
                article["created_at"],
                article["model_name"],
                article["auto_threshold"],
                article["manual_threshold"],
                article["flagged_count"],
                json.dumps(article["trend"]),
                json.dumps(article.get("inference_stats", {})),
            ),
        )
        conn.execute("DELETE FROM comments WHERE article_id = ?", (article["id"],))
        for c in comments:
            conn.execute(
                """
                INSERT INTO comments
                (id, article_id, author, timestamp, text, toxicity, decision, is_flagged, top_features_json, model_version, explain_version, inference_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c["id"],
                    article["id"],
                    c["author"] or "unsigned",
                    c["timestamp"],
                    c["text"],
                    c["toxicity"],
                    c["decision"],
                    int(c["is_flagged"]),
                    json.dumps(c["top_features"]),
                    c["model_version"],
                    c["explain_version"],
                    c.get("inference_ms", 0.0),
                ),
            )
        conn.commit()

def list_articles():
    initialize_schema()
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM articles ORDER BY created_at DESC").fetchall()
    return [serialize_article_summary(r) for r in rows]


def get_article_summary(article_id):
    initialize_schema()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
    return serialize_article_summary(row) if row else {}


def _comment_base(article_id, decision):
    base = "FROM comments WHERE article_id = ?"
    params = [article_id]
    if decision == "flagged":
        base += " AND is_flagged = 1"
    elif decision:
        base += " AND decision = ?"
        params.append(decision)
    return base, params


def _comment_order_clause(sort):
    sort_map = {
        "toxicity_desc": "ORDER BY toxicity DESC",
        "toxicity_asc": "ORDER BY toxicity ASC",
        "timestamp_desc": "ORDER BY timestamp DESC",
        "timestamp_asc": "ORDER BY timestamp ASC",
        "decision_asc": (
            "ORDER BY CASE decision "
            "WHEN 'auto-ban' THEN 1 "
            "WHEN 'manual-ban' THEN 2 "
            "WHEN 'manual-review' THEN 3 "
            "ELSE 4 END"
        ),
    }
    return sort_map.get(sort, "ORDER BY toxicity DESC")


def list_comments(article_id, limit=50, offset=0, decision=None, sort="toxicity_desc"):
    initialize_schema()
    base_sql, params = _comment_base(article_id, decision)
    order_clause = _comment_order_clause(sort)
    with get_connection() as conn:
        total = conn.execute(f"SELECT COUNT(*) {base_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * {base_sql} {order_clause} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    return [serialize_comment(r) for r in rows], total


def get_article(article_id, include_comments=True, limit=50, offset=0, decision=None, sort="toxicity_desc"):
    initialize_schema()
    with get_connection() as conn:
        article = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()

    if not article:
        return {}

    payload = serialize_article_summary(article)
    if include_comments:
        comments, total = list_comments(article_id, limit=limit, offset=offset, decision=decision, sort=sort)
        payload.update(
            {
                "comments": comments,
                "comments_total": total,
                "comments_limit": limit,
                "comments_offset": offset,
            }
        )
    return payload


def update_thresholds(article_id, auto_threshold, manual_threshold):
    initialize_schema()
    with get_connection() as conn:
        comments = conn.execute(
            "SELECT id, toxicity, decision FROM comments WHERE article_id = ?",
            (article_id,),
        ).fetchall()

        flagged_count = 0
        for row in comments:
            toxicity = row["toxicity"]
            # Preserve moderator decisions — only recalculate system-assigned ones
            if row["decision"] == "manual-ban":
                flagged_count += 1
                continue
            decision = "none"
            if toxicity >= auto_threshold:
                decision = "auto-ban"
            elif toxicity >= manual_threshold:
                decision = "manual-review"
            is_flagged = toxicity >= manual_threshold
            if is_flagged:
                flagged_count += 1
            conn.execute(
                "UPDATE comments SET decision = ?, is_flagged = ? WHERE id = ?",
                (decision, int(is_flagged), row["id"]),
            )

        conn.execute(
            """
            UPDATE articles
            SET auto_threshold = ?, manual_threshold = ?, flagged_count = ?
            WHERE id = ?
            """,
            (auto_threshold, manual_threshold, flagged_count, article_id),
        )
        conn.commit()


def get_comment(article_id, comment_id):
    initialize_schema()
    with get_connection() as conn:
        article = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        comment = conn.execute(
            "SELECT * FROM comments WHERE id = ? AND article_id = ?",
            (comment_id, article_id),
        ).fetchone()

    if not article or not comment:
        return {}

    return {
        "article": {
            "id": article["id"],
            "title": article["title"],
            "url": article["url"],
            "summary": article["summary"],
            "auto_threshold": article["auto_threshold"],
            "manual_threshold": article["manual_threshold"],
            "model_name": article["model_name"],
        },
        "comment": serialize_comment(comment),
    }


def update_comment_decision(comment_id, decision):
    initialize_schema()
    with get_connection() as conn:
        conn.execute(
            "UPDATE comments SET decision = ? WHERE id = ?",
            (decision, comment_id),
        )
        conn.commit()


def update_comment_explanation(comment_id, top_features, explain_version):
    initialize_schema()
    with get_connection() as conn:
        conn.execute(
            "UPDATE comments SET top_features_json = ?, explain_version = ? WHERE id = ?",
            (json.dumps(top_features), explain_version, comment_id),
        )
        conn.commit()