import requests
import xml.etree.ElementTree as ET
from utils.text_utils import clean_text

def fetch_ai_articles():
    url = "https://news.google.com/rss/search?q=artificial+intelligence+when:1h&hl=en-US&gl=US&ceid=US:en"
    
    response = requests.get(url)
    root = ET.fromstring(response.content)

    articles = []
    seen_titles = set()

    for item in root.findall(".//item"):
        title = item.find("title").text

        if title in seen_titles:
            continue

        seen_titles.add(title)

        description = item.find("description").text

        articles.append({
            "title": title,
            "link": item.find("link").text,
            "published": item.find("pubDate").text,
            "description": clean_text(description)
        })

        if len(articles) == 5:
            break

    return articles