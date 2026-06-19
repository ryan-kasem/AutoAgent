"""
tools/wikipedia.py — grab summaries from Wikipedia

Wikipedia has a legit free API that doesn't need auth.
great for factual questions where you need a reliable source.
"""

import urllib.request
import urllib.parse
import json
from config import cfg


def wikipedia_search(topic: str) -> str:
    """
    search Wikipedia and return a summary of the top result.
    uses the Wikipedia REST API — clean, fast, no key needed.
    """
    # first search for the right page title
    search_url = (
        "https://en.wikipedia.org/w/api.php?"
        f"action=search&list=search&srsearch={urllib.parse.quote(topic)}"
        "&format=json&srlimit=1"
    )

    try:
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "AutoAgent/1.0 (educational project)"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            search_data = json.loads(r.read().decode())

        results = search_data.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia article found for '{topic}'."

        # grab the top result
        page_title = results[0]["title"]

        # now fetch the actual summary
        summary_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(page_title)}"
        )
        req2 = urllib.request.Request(
            summary_url,
            headers={"User-Agent": "AutoAgent/1.0 (educational project)"}
        )
        with urllib.request.urlopen(req2, timeout=8) as r:
            page_data = json.loads(r.read().decode())

        title = page_data.get("title", page_title)
        extract = page_data.get("extract", "No summary available.")
        url = page_data.get("content_urls", {}).get("desktop", {}).get("page", "")

        # trim the extract — Wikipedia summaries can run long
        if len(extract) > cfg.tools.wiki_max_chars:
            extract = extract[:cfg.tools.wiki_max_chars] + "..."

        return f"**{title}**\n{extract}\nSource: {url}"

    except Exception as e:
        return f"Wikipedia lookup failed: {e}"
