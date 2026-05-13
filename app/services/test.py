"""
test.py - simple testing script for validating Wikipedia talk-page comment
retrieval and export functionality

This script fetches comments from a Wikipedia article talk page and exports 
retrieved comments to JSON format. 

Primarily used for:
- local testing
- fetcher validation
- debugging Wikipedia comment extraction
"""

from wikipedia_talk_fetcher import fetch_and_export_comments

fetch_and_export_comments(
    "https://en.wikipedia.org/wiki/Hertie_School",
    output_format='json',
    output_file='output.json'
)