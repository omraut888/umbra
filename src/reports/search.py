"""Web search backends for external recommendations.

anthropic  Claude's server-side web search tool (web_search_20250305) on Haiku.
           Uses the same ANTHROPIC_API_KEY as probe generation, so nothing
           extra to set up.
brave      Brave Search API, needs BRAVE_API_KEY.
"""

from __future__ import annotations

import os
from typing import Dict, List

import httpx

from src.probe_generation.taxonomy import CLAUDE_MODEL
from src.reports.recommend import SearchResult

SEARCH_PROMPT = """Search the web for: {query}

Then list the most relevant sources you found, quoting one or two key sentences from each."""


class AnthropicWebSearch:
    def __init__(self, model: str = CLAUDE_MODEL):
        import anthropic

        self.client = anthropic.AsyncAnthropic(max_retries=5)
        self.model = model

    async def search(self, query: str, n: int) -> List[SearchResult]:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            messages=[{"role": "user", "content": SEARCH_PROMPT.format(query=query)}],
        )
        titles: Dict[str, str] = {}
        quotes: Dict[str, List[str]] = {}
        for block in response.content:
            # result blocks carry url + title only (content is encrypted); the
            # readable text comes back as citations on the model's reply
            if block.type == "web_search_tool_result" and isinstance(block.content, list):
                for r in block.content:
                    titles.setdefault(r.url, r.title)
            elif block.type == "text" and getattr(block, "citations", None):
                for cit in block.citations:
                    if cit.type == "web_search_result_location":
                        quotes.setdefault(cit.url, []).append(cit.cited_text)
                        titles.setdefault(cit.url, cit.title or cit.url)
        ordered = sorted(titles, key=lambda u: u not in quotes)  # results with quoted text first
        return [SearchResult(title=titles[u], url=u, snippet=" ".join(quotes.get(u, []))[:600] or titles[u])
                for u in ordered[:n]]


class BraveSearch:
    URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["BRAVE_API_KEY"]

    async def search(self, query: str, n: int) -> List[SearchResult]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self.URL, params={"q": query, "count": n},
                                    headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"})
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        return [SearchResult(title=r.get("title", ""), url=r["url"], snippet=r.get("description", "")) for r in results[:n]]


def make_provider(name: str):
    if name == "anthropic":
        return AnthropicWebSearch()
    if name == "brave":
        return BraveSearch()
    raise ValueError(f"unknown web search provider {name!r}")
