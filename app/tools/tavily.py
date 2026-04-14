"""Tavily 联网搜索工具。"""

from __future__ import annotations

from typing import Literal

from tavily import TavilyClient


def build_internet_search_tool(api_key: str):
    tavily_client = TavilyClient(api_key=api_key)

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search."""
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    return internet_search
