from urllib.parse import urlparse, unquote
import re
import html
import json
import logging
import requests
from app.services.wikipedia_talk_fetcher import WikipediaTalkFetcher

logger = logging.getLogger(__name__)

WIKI_HEADERS = {
    "User-Agent": "bogosort-dashboard/1.0 (contact: ab.dubey@students.hertie-school.org)",
    "From": "ab.dubey@students.hertie-school.org",
    "Accept": "application/json",
}
HTML_TAG_RE = re.compile(r"<[^>]+>")

def is_allowed_wikipedia_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host != "wikipedia.org" and not host.endswith(".wikipedia.org"):
        return False
    return "/wiki/" in parsed.path


def parse_wiki_title_from_url(url):
    if not is_allowed_wikipedia_url(url):
        return None
    path = urlparse(url).path
    parts = path.split("/wiki/", 1)
    if len(parts) < 2 or not parts[1]:
        return None
    return unquote(parts[1])


def fetch_wikipedia_metadata(title):
    logger.debug(f"Fetching Wikipedia metadata for: {title}")

    params = {
        "action": "query",
        "prop": "extracts|info",
        "exintro": "1",
        "explaintext": "1",
        "inprop": "url",
        "titles": title,
        "format": "json",
        "origin": "*",
    }
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            headers=WIKI_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        logger.debug(f"Wikipedia API response status: {resp.status_code}")
    except requests.RequestException as exc:
        logger.error(f"Failed to fetch Wikipedia metadata for {title}: {exc}", exc_info=True)
        raise ValueError("Failed to fetch Wikipedia metadata") from exc

    try:
        payload = resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(f"Invalid JSON from Wikipedia API: {exc}", exc_info=True)
        raise ValueError("Wikipedia API returned invalid response") from exc

    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        logger.warning(f"Wikipedia article not found: {title}")
        return {"title": title, "summary": "", "url": ""}
    page = next(iter(pages.values()))
    return {
        "title": page.get("title", title),
        "summary": page.get("extract", "") or "",
        "url": page.get("fullurl", "") or "",
    }

def fetch_talk_page_comments(title, limit=None):
    """
    Fetch comments from Wikipedia talk page using WikipediaTalkFetcher.

    Args:
        title: Wikipedia article title
        limit: Maximum number of comments to return

    Returns:
        List of comment dictionaries with author, timestamp, text
    """
    try:
        fetcher = WikipediaTalkFetcher()
        wiki_comments = fetcher.get_all_comments(title)

        results = []
        for comment in wiki_comments:
            try:
                results.append({
                    "id": "",
                    "author": comment.author,
                    "timestamp": comment.timestamp.isoformat() if comment.timestamp else "",
                    "text": comment.text,
                })
            except (AttributeError, TypeError) as exc:
                logger.warning(f"Skipping malformed comment for {title}: {exc}")
                continue

        if limit:
            return results[:limit]
        return results
    except Exception as exc:
        logger.error(f"Talk page fetch failed for {title}: {exc}", exc_info=True)
        raise ValueError(f"Failed to fetch talk page comments for {title}") from exc