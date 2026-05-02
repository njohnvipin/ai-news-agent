import re
import html


def clean_text(text: str) -> str:
    """Remove HTML tags and decode HTML entities from text."""
    if not text:
        return ""

    # decode HTML entities like &nbsp; &amp; &lt; &gt;
    text = html.unescape(text)

    # remove HTML tags like <b>, <br>, <a href="...">
    text = re.sub(r'<[^>]+>', '', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text