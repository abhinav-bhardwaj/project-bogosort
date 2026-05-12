import hashlib
import logging
from datetime import datetime

from app.db import article_repository
from app.services import evaluation_service, toxicity_service, wiki_client

logger = logging.getLogger(__name__)

DEFAULT_MODEL = toxicity_service.DEFAULT_MODEL


def slugify_title(title):
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in title).split())


def _decide_action(toxicity, auto_threshold, manual_threshold):
    if toxicity >= auto_threshold:
        return "auto-ban"
    if toxicity >= manual_threshold:
        return "manual-review"
    return "none"

def ingest_article(url, limit=30, auto_threshold=0.75, manual_threshold=0.55, model_name=DEFAULT_MODEL):
    logger.info(f"Starting article ingestion: {url}")

    if manual_threshold > auto_threshold:
        raise ValueError("manual_threshold must be <= auto_threshold")

    article_repository.initialize_schema()
    if not wiki_client.is_allowed_wikipedia_url(url):
        raise ValueError("Invalid Wikipedia URL")
    title = wiki_client.parse_wiki_title_from_url(url)
    if not title:
        raise ValueError("Invalid Wikipedia URL")

    meta = wiki_client.fetch_wikipedia_metadata(title)
    if not meta:
        raise ValueError(f"Failed to fetch article metadata")
    logger.info(f"Fetched metadata for: {meta.get('title', 'Unknown')}")

    # Fetch talk page comments using WikipediaTalkFetcher
    talk_comments = wiki_client.fetch_talk_page_comments(title, limit=limit)
    logger.info(f"Fetched {len(talk_comments)} comments from talk page")

    article_id = slugify_title(meta["title"])
    created_at = datetime.utcnow().isoformat()
    model_version = evaluation_service.get_model_version(model_name) or "unknown"

    comments = []
    trend_dates = []
    trend_scores = []
    running = 0.0

    for c in talk_comments:
        comment_text = c["text"]
        result = toxicity_service.score_comment(comment_text, model_name=model_name, explain=False)
        try:
            prob = result.get("probability")
            toxicity = float(prob) if prob is not None else 0.0
        except (TypeError, ValueError) as exc:
            logger.error(f"Invalid toxicity probability: {prob}", exc_info=True)
            toxicity = 0.0
        inference_ms = result.get("inference_ms", 0.0)
        running += toxicity

        timestamp = c.get("timestamp", "")
        date_key = timestamp.split("T")[0] if timestamp else ""

        trend_dates.append(date_key)
        trend_scores.append(round(running, 3))

        decision = _decide_action(toxicity, auto_threshold, manual_threshold)
        is_flagged = toxicity >= manual_threshold

        comment_id = hashlib.md5(
            f"{timestamp}-{c.get('author','')}-{comment_text}".encode("utf-8")
        ).hexdigest()

        comments.append(
            {
                "id": comment_id,
                "author": c.get("author") or "unsigned",
                "timestamp": timestamp,
                "text": comment_text,
                "toxicity": toxicity,
                "decision": decision,
                "is_flagged": is_flagged,
                "top_features": result.get("top_features", []),
                "model_version": model_version,
                "explain_version": result.get("explain_version", ""),
                "inference_ms": inference_ms,
            }
        )

    trend = {"dates": trend_dates, "scores": trend_scores, "threshold": manual_threshold}
    flagged_count = sum(1 for c in comments if c["is_flagged"])

    inference_times = [c["inference_ms"] for c in comments if c["inference_ms"] > 0]
    inference_stats = {
        "count": len(inference_times),
        "total_ms": round(sum(inference_times), 1),
        "avg_ms": round(sum(inference_times) / len(inference_times), 1) if inference_times else 0.0,
        "min_ms": round(min(inference_times), 1) if inference_times else 0.0,
        "max_ms": round(max(inference_times), 1) if inference_times else 0.0,
    }
    logger.info(f"Scored {len(comments)} comments, flagged {flagged_count}, avg inference {inference_stats['avg_ms']}ms")

    article_repository.upsert_article(
        {
            "id": article_id,
            "title": meta["title"],
            "url": meta["url"] or url,
            "summary": meta["summary"],
            "created_at": created_at,
            "model_name": model_name,
            "auto_threshold": auto_threshold,
            "manual_threshold": manual_threshold,
            "flagged_count": flagged_count,
            "trend": trend,
            "inference_stats": inference_stats,
        },
        comments,
    )

    logger.info(f"Successfully ingested article: {article_id}")
    return article_repository.get_article_summary(article_id)

def list_articles():
    return article_repository.list_articles()

def get_article(article_id, include_comments=True, limit=50, offset=0, decision=None, sort="toxicity_desc"):
    return article_repository.get_article(
        article_id, include_comments=include_comments, limit=limit, offset=offset, decision=decision, sort=sort
    )


def list_comments(article_id, limit=50, offset=0, decision=None, sort="toxicity_desc"):
    comments, total = article_repository.list_comments(
        article_id, limit=limit, offset=offset, decision=decision, sort=sort
    )
    return {"comments": comments, "total": total, "limit": limit, "offset": offset}

def update_thresholds(article_id, auto_threshold, manual_threshold):
    article_repository.update_thresholds(article_id, auto_threshold, manual_threshold)


def update_comment_decision(article_id, comment_id, decision):
    article_repository.update_comment_decision(comment_id, decision)


def get_comment_detail(article_id, comment_id):
    payload = article_repository.get_comment(article_id, comment_id)
    if not payload:
        return {}

    comment = payload["comment"]
    needs_explain = not comment["top_features"] or (
        comment.get("explain_version") != toxicity_service.EXPLAIN_VERSION
    )
    if needs_explain:
        result = toxicity_service.score_comment(
            comment["text"], model_name=payload["article"]["model_name"], explain=True
        )
        if result.get("top_features"):
            article_repository.update_comment_explanation(
                comment_id, result["top_features"], result["explain_version"]
            )
            comment["top_features"] = result["top_features"]
            comment["explain_version"] = result["explain_version"]

    payload["comment"] = comment
    return payload