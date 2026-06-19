"""
tools/web_search.py — search the web using DuckDuckGo

DuckDuckGo has an unofficial API that doesn't need any key.
it's not as powerful as Google but it's free and gets the job done.
"""

import urllib.request
import urllib.parse
import json
from config import cfg


def web_search(query: str) -> str:
    """
    search DuckDuckGo and return the top results as text.
    we use the instant answer API — it's not perfect but doesn't require auth.
    """
    # encode the query for a URL
    encoded = urllib.parse.quote(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoAgent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())

        results = []

        # instant answer (like a calculator result or definition)
        if data.get("Answer"):
            results.append(f"Answer: {data['Answer']}")

        # abstract — usually a short summary from Wikipedia/etc
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractURL"):
                results.append(f"Source: {data['AbstractURL']}")

        # related topics — these are the actual search results
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"• {topic['Text']}")

        if not results:
            # DDG instant answers API came up empty — happens with very specific queries
            return f"No instant results found for '{query}'. Try rephrasing or use wikipedia tool."

        output = "\n".join(results)
        # trim it so we don't blow up the context window
        return output[:cfg.tools.search_max_chars]

    except Exception as e:
        return f"Search failed: {e}"
