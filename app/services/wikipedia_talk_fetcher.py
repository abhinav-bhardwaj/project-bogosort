"""
Wikipedia Talk Page Comment Fetcher

This module provides utilities to fetch and parse comments from Wikipedia talk pages
using the MediaWiki API. It handles both the old and new API response formats.

Key improvements:
- Supports both URL input and page titles
- Handles both old API format (revisions[0]["*"]) and new format (revisions[0]["slots"]["main"]["*"])
- Robust error handling with exponential backoff
- Improved wikitext parsing for real Wikipedia talk pages
- Supports export to JSON, CSV, and plain text
"""

import requests
import re
import time
import json
import csv
from io import StringIO
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse, unquote
from random import random


@dataclass
class WikiComment:
    """Represents a single comment on a talk page."""
    author: str
    timestamp: Optional[datetime]
    text: str
    section: str
    level: int  # Indentation level (1-based)
    raw_wikitext: str


class WikipediaTalkFetcher:
    """Fetches and parses comments from Wikipedia talk pages."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize the fetcher.
        
        Args:
            language: Wikipedia language code (e.g., 'en', 'de', 'fr')
        """
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikitalkCommentFetcher/1.0 (Research; +https://github.com)'
        })

    @staticmethod
    def _extract_title(page_or_url: str) -> str:
        """
        Extract page title from either a URL or a page title string.
        
        Supports:
        - Full URLs: https://en.wikipedia.org/wiki/Albert_Einstein
        - Page titles: Albert Einstein or Talk:Albert Einstein
        
        Args:
            page_or_url: URL or page title
            
        Returns:
            Page title (without "Talk:" prefix if present)
        """
        # Check if it's a URL
        if page_or_url.startswith("http://") or page_or_url.startswith("https://"):
            parsed = urlparse(page_or_url)
            path = parsed.path
            if "/wiki/" not in path:
                raise ValueError("URL must be a Wikipedia article URL containing /wiki/")
            title = path.split("/wiki/", 1)[1]
            title = unquote(title)
        else:
            title = page_or_url
        
        # Remove "Talk:" prefix if present
        if title.startswith("Talk:"):
            title = title[5:]
        
        return title

    def _get_with_backoff(self, url: str, params: Dict, max_retries: int = 5) -> requests.Response:
        """
        Make HTTP request with exponential backoff for rate limiting.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response object
            
        Raises:
            ValueError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=20)
                
                if resp.status_code == 429:
                    # Rate limited - exponential backoff with jitter
                    sleep_time = (2 ** attempt) + random() * 0.5
                    print(f"Rate limited. Waiting {sleep_time:.2f}s before retry...")
                    time.sleep(sleep_time)
                    continue
                
                resp.raise_for_status()
                return resp
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff for other errors
                sleep_time = (2 ** attempt) + random() * 0.5
                time.sleep(sleep_time)
        
        raise ValueError("Failed after maximum retries")

    def get_talk_page_wikitext(self, page_title: str) -> Optional[str]:
        """
        Fetch the raw wikitext of a talk page.
        
        Args:
            page_title: Title of the article (or full URL, or Talk:ArticleTitle)
            
        Returns:
            Raw wikitext content or None if page not found or empty
        """
        # Extract clean title
        page_title = self._extract_title(page_title)
        talk_page_title = f"Talk:{page_title}"
        
        print(f"Fetching wikitext for: {talk_page_title}")
        
        # Use new API format with slots for better structure
        params = {
            'action': 'query',
            'prop': 'revisions',
            'titles': talk_page_title,
            'rvprop': 'content',
            'rvslots': 'main',
            'format': 'json',
            'origin': '*'
        }
        
        try:
            response = self._get_with_backoff(self.base_url, params=params)
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            if not pages:
                print(f"No pages found for {talk_page_title}")
                return None
            
            page_id = list(pages.keys())[0]
            page = pages[page_id]
            
            # Handle case where page doesn't exist (negative page ID)
            if page_id.startswith('-'):
                print(f"Talk page does not exist: {talk_page_title}")
                return None
            
            revisions = page.get('revisions', [])
            if not revisions:
                print(f"No revisions found for {talk_page_title}")
                return None
            
            revision = revisions[0]
            
            # Handle both old and new API formats
            # New format: slots -> main -> * 
            if 'slots' in revision:
                content = revision.get('slots', {}).get('main', {}).get('*')
            # Old format: *
            else:
                content = revision.get('*')
            
            if not content:
                print(f"Empty content for {talk_page_title}")
                return None
            
            print(f"Successfully fetched {len(content)} characters")
            return content
            
        except Exception as e:
            print(f"Error fetching talk page: {e}")
            return None

    def get_talk_page_html(self, page_title: str) -> Optional[str]:
        """
        Fetch the HTML-rendered talk page.
        
        Args:
            page_title: Title of the article (or URL)
            
        Returns:
            HTML content or None if page not found
        """
        page_title = self._extract_title(page_title)
        talk_page_title = f"Talk:{page_title}"
        
        params = {
            'action': 'query',
            'titles': talk_page_title,
            'prop': 'extracts',
            'explaintext': False,
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            page_id = list(pages.keys())[0]
            
            if 'extract' not in pages[page_id]:
                return None
            
            return pages[page_id]['extract']
            
        except Exception as e:
            print(f"Error fetching talk page HTML: {e}")
            return None

    def parse_wikitext_comments(self, wikitext: str, section_title: str = "") -> List[WikiComment]:
        """
        Parse comments from raw wikitext.
        
        Captures:
        1. Signed comments: [[User:... with timestamp HH:MM, DD Month YYYY (UTC)
        2. Unsigned with user: [[User:... but NO timestamp (e.g., {{Unsigned}} templates)
        3. Unsigned plain: text without any user link
        
        Multi-line comments are buffered until a signature or section is found.
        
        Args:
            wikitext: Raw wikitext content
            section_title: Current section being parsed
            
        Returns:
            List of WikiComment objects
        """
        comments: List[WikiComment] = []
        lines = wikitext.split("\n")
        
        current_section = section_title
        buffer: List[str] = []
        
        def should_skip_line(line: str) -> bool:
            """Check if line should be skipped (template, metadata, HTML)."""
            stripped = line.strip()
            if not stripped:
                return True
            # Skip closing template braces, template parameters, HTML
            if stripped.startswith("{{") or stripped.startswith("|") or stripped.startswith("<"):
                return True
            # Skip closing template braces
            if stripped.startswith("}}") or stripped == "}}":
                return True
            # Skip lines that are just reference markers or wiki syntax
            if stripped in (":", "::") or stripped.startswith("<!--"):
                return True
            return False
        
        def flush_signed_or_unsigned_comment(line: str):
            """Process a line with a user signature (signed or unsigned)."""
            nonlocal buffer, current_section
            
            # Check for user link
            if "[[User:" not in line:
                return False
            
            # Check for timestamp
            has_timestamp = re.search(r"\d{2}:\d{2},\s+\d{1,2}\s+\w+\s+\d{4}\s+\(UTC\)", line)
            
            # Extract user
            user_match = re.search(r"\[\[User:([^\]|#]+)", line)
            if not user_match:
                return False
            
            author = user_match.group(1).strip()
            timestamp_str = None
            timestamp = None
            
            if has_timestamp:
                timestamp_str = has_timestamp.group(0)
                try:
                    timestamp = self._parse_timestamp(timestamp_str)
                except Exception:
                    timestamp = None
            
            # Get comment text - everything BEFORE the first [[User: link
            user_pos = line.find("[[User:")
            comment_text = line[:user_pos].strip()
            comment_text = comment_text.lstrip(':#* ')
            
            # Prepend buffered lines
            if buffer:
                buffered = "\n".join(buffer).strip()
                if comment_text:
                    comment_text = f"{buffered}\n{comment_text}"
                else:
                    comment_text = buffered
                buffer = []
            
            if not comment_text:
                return True
            
            # Calculate indentation level
            level = 0
            for char in line:
                if char in ':#*':
                    level += 1
                elif char != ' ' and char != '\t':
                    break
            if level == 0:
                level = 1
            
            # Create comment
            comments.append(WikiComment(
                author=author,
                timestamp=timestamp,
                text=comment_text,
                section=current_section,
                level=level,
                raw_wikitext=line
            ))
            
            return True
        
        # Main parsing loop
        for line in lines:
            # Check for section headers
            stripped = line.strip()
            if stripped.startswith("==") and stripped.endswith("=="):
                # Flush any buffered content before switching sections
                if buffer and any(b.strip() for b in buffer):
                    comment_text = "\n".join(b.strip() for b in buffer if b.strip())
                    if comment_text:
                        comments.append(WikiComment(
                            author="",
                            timestamp=None,
                            text=comment_text,
                            section=current_section,
                            level=1,
                            raw_wikitext=""
                        ))
                    buffer = []
                
                current_section = stripped.strip("= ")
                continue
            
            # Skip lines that should be ignored
            if should_skip_line(line):
                continue
            
            # Try to process as signed/unsigned comment
            if flush_signed_or_unsigned_comment(line):
                continue
            
            # Otherwise, buffer as potential unsigned comment content
            if stripped:
                buffer.append(stripped)
        
        # Flush any remaining buffered content at end of file
        if buffer and any(b.strip() for b in buffer):
            comment_text = "\n".join(b.strip() for b in buffer if b.strip())
            if comment_text:
                comments.append(WikiComment(
                    author="",
                    timestamp=None,
                    text=comment_text,
                    section=current_section,
                    level=1,
                    raw_wikitext=""
                ))
        
        return comments

    def get_all_comments(self, page_title: str, parse_method: str = 'wikitext') -> List[WikiComment]:
        """
        Fetch and parse all comments from a talk page.
        
        Args:
            page_title: Article title, URL, or Talk:PageName
            parse_method: 'wikitext' (recommended) or 'html'
            
        Returns:
            List of WikiComment objects
        """
        if parse_method == 'wikitext':
            wikitext = self.get_talk_page_wikitext(page_title)
            if not wikitext:
                return []
            return self.parse_wikitext_comments(wikitext)
            
        elif parse_method == 'html':
            html = self.get_talk_page_html(page_title)
            if not html:
                return []
            return self.parse_html_comments(html)
        else:
            raise ValueError("parse_method must be 'wikitext' or 'html'")

    def parse_html_comments(self, html: str) -> List[WikiComment]:
        """
        Parse comments from HTML content (simpler approach).
        
        Note: This is a basic implementation. For production use,
        consider using BeautifulSoup for more robust parsing.
        
        Args:
            html: HTML content of the talk page
            
        Returns:
            List of WikiComment objects
        """
        comments = []
        lines = html.split('\n')
        current_section = ""
        
        for line in lines:
            # Find section headers
            if '<h2>' in line or '<h3>' in line:
                section_match = re.search(r'<h[23]>(.+?)</h[23]>', line)
                if section_match:
                    current_section = section_match.group(1).strip()
            
            # Find comment divs
            if '<div' in line and 'id=' in line:
                text_match = re.search(r'>(.+?)<', line)
                if text_match:
                    comments.append(WikiComment(
                        author="Unknown",
                        timestamp=None,
                        text=text_match.group(1),
                        section=current_section,
                        level=1,
                        raw_wikitext=line
                    ))
        
        return comments

    def get_recent_changes(self, page_title: str, limit: int = 50) -> List[Dict]:
        """
        Fetch recent changes (edits) to a talk page.
        
        Args:
            page_title: Article title or URL
            limit: Maximum number of changes to fetch (max 500)
            
        Returns:
            List of revision records with timestamp, user, comment, size
        """
        page_title = self._extract_title(page_title)
        talk_page_title = f"Talk:{page_title}"
        
        params = {
            'action': 'query',
            'titles': talk_page_title,
            'prop': 'revisions',
            'rvlimit': min(limit, 500),
            'rvprop': 'timestamp|user|comment|size',
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            page_id = list(pages.keys())[0]
            
            revisions = pages[page_id].get('revisions', [])
            return revisions
            
        except Exception as e:
            print(f"Error fetching recent changes: {e}")
            return []

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> datetime:
        """
        Parse Wikipedia timestamp format: HH:MM, DD Month YYYY (UTC)
        
        Example: "14:30, 15 May 2024 (UTC)"
        
        Args:
            timestamp_str: Timestamp string from Wikipedia
            
        Returns:
            datetime object
            
        Raises:
            ValueError: If timestamp format is not recognized
        """
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        match = re.search(
            r'(\d{2}):(\d{2}),\s+(\d+)\s+(\w+)\s+(\d{4})',
            timestamp_str
        )
        
        if match:
            hour, minute, day, month_str, year = match.groups()
            month = month_map.get(month_str, 1)
            return datetime(int(year), month, int(day), int(hour), int(minute))
        
        raise ValueError(f"Could not parse timestamp: {timestamp_str}")


# ============================================================================
# Helper Functions for Easy Use
# ============================================================================

def fetch_comments_simple(page_title: str, language: str = "en") -> List[Dict]:
    """
    Simple one-liner to fetch comments from a Wikipedia talk page.
    
    Args:
        page_title: Article title, URL, or Talk:PageName
        language: Language code (default: 'en')
        
    Returns:
        List of dictionaries with comment data
        
    Example:
        comments = fetch_comments_simple("Albert_Einstein")
        for c in comments:
            print(f"{c['author']}: {c['text'][:50]}")
    """
    fetcher = WikipediaTalkFetcher(language=language)
    comments = fetcher.get_all_comments(page_title, parse_method='wikitext')
    
    return [
        {
            'author': c.author,
            'timestamp': c.timestamp.isoformat() if c.timestamp else None,
            'text': c.text,
            'section': c.section,
            'level': c.level
        }
        for c in comments
    ]


def fetch_and_export_comments(
    page_title: str,
    output_format: str = 'json',
    output_file: Optional[str] = None,
    language: str = 'en'
) -> str:
    """
    Fetch comments and export to various formats.
    
    Args:
        page_title: Article title, URL, or Talk:PageName
        output_format: 'json', 'csv', or 'txt'
        output_file: Path to save output (if None, only returns string)
        language: Language code (default: 'en')
        
    Returns:
        Formatted output as string
        
    Example:
        fetch_and_export_comments(
            "https://en.wikipedia.org/wiki/Albert_Einstein",
            output_format='json',
            output_file='einstein_talk.json'
        )
    """
    fetcher = WikipediaTalkFetcher(language=language)
    page_title = fetcher._extract_title(page_title)
    comments = fetcher.get_all_comments(page_title)
    
    if output_format == 'json':
        output = json.dumps(
            [
                {
                    'author': c.author,
                    'timestamp': c.timestamp.isoformat() if c.timestamp else None,
                    'text': c.text,
                    'section': c.section,
                    'level': c.level
                }
                for c in comments
            ],
            indent=2,
            ensure_ascii=False
        )
    
    elif output_format == 'csv':
        output_io = StringIO()
        writer = csv.DictWriter(
            output_io,
            fieldnames=['author', 'timestamp', 'section', 'level', 'text']
        )
        writer.writeheader()
        for c in comments:
            writer.writerow({
                'author': c.author,
                'timestamp': c.timestamp.isoformat() if c.timestamp else '',
                'section': c.section,
                'level': c.level,
                'text': c.text[:200]  # Truncate for CSV
            })
        output = output_io.getvalue()
    
    else:  # 'txt'
        lines = []
        for c in comments:
            lines.append(f"\n{'='*70}")
            lines.append(f"Author: {c.author}")
            lines.append(f"Timestamp: {c.timestamp}")
            lines.append(f"Section: {c.section}")
            lines.append(f"Level: {c.level}")
            lines.append(f"{'─'*70}")
            lines.append(c.text)
        output = '\n'.join(lines)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Saved to {output_file}")
    
    return output


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Using a page title
    print("=" * 70)
    print("Example 1: Fetch comments using page title")
    print("=" * 70)
    
    fetcher = WikipediaTalkFetcher()
    comments = fetcher.get_all_comments("Climate change")
    
    print(f"\nFound {len(comments)} comments\n")
    
    for i, comment in enumerate(comments[:3], 1):
        print(f"Comment {i}:")
        print(f"  Author: {comment.author}")
        print(f"  Timestamp: {comment.timestamp}")
        print(f"  Section: {comment.section}")
        print(f"  Text: {comment.text[:80]}...")
        print()
    
    # Example 2: Using a URL
    print("\n" + "=" * 70)
    print("Example 2: Fetch comments using URL")
    print("=" * 70)
    
    url = "https://en.wikipedia.org/wiki/Albert_Einstein"
    comments_url = fetch_comments_simple(url)
    print(f"\nFound {len(comments_url)} comments from Einstein talk page")
    
    # Example 3: Export to JSON
    print("\n" + "=" * 70)
    print("Example 3: Export to JSON")
    print("=" * 70)
    
    json_output = fetch_and_export_comments(
        "Machine learning",
        output_format='json',
        output_file='ml_talk_comments.json'
    )
    
    # Example 4: Get recent changes
    print("\n" + "=" * 70)
    print("Example 4: Recent changes to talk page")
    print("=" * 70)
    
    changes = fetcher.get_recent_changes("Climate change", limit=5)
    print(f"\nRecent changes to Talk:Climate change:")
    for change in changes:
        print(f"  {change.get('timestamp', 'N/A')}: {change.get('user', 'N/A')} - {change.get('comment', '(no summary)')}")