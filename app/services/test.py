from wikipedia_talk_fetcher import fetch_and_export_comments

fetch_and_export_comments(
    "https://en.wikipedia.org/wiki/Hertie_School",
    output_format='json',
    output_file='output.json'
)