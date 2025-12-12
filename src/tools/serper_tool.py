"""
Serper API Web Search Tool
Provides reliable web search using Google Serper API.
"""

import os
import json
import logging
import requests
from typing import Dict, List
from src.utils import exponential_backoff, safe_text

logger = logging.getLogger(__name__)


@exponential_backoff(max_retries=3, base_delay=1.0)
def run_serper(query: str, max_results: int = 5) -> Dict:
    """
    Perform web search using Serper API (Google-based results).
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Dictionary with standardized format:
        {
            "text": "Formatted search results",
            "sources": ["url1", "url2", ...],
            "meta": {"tool": "serper", "num_results": N}
        }
    
    Example:
        >>> result = run_serper("OpenAI CEO")
        >>> print(result["text"])
        [1] Sam Altman - CEO of OpenAI
            Sam Altman is the CEO and co-founder of OpenAI...
            Source: https://openai.com/about
    """
    api_key = os.getenv('SERPER_API_KEY')
    
    if not api_key:
        logger.error("SERPER_API_KEY not found in environment")
        return {
            "text": "Serper API key not configured. Please set SERPER_API_KEY in .env file.",
            "sources": [],
            "meta": {"tool": "serper", "error": "missing_api_key"}
        }
    
    try:
        logger.info(f"Serper search: {query}")
        
        url = "https://google.serper.dev/search"
        
        payload = json.dumps({
            "q": query,
            "num": max_results  # Number of results to return
        })
        
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract organic results
        organic_results = data.get('organic', [])
        
        if not organic_results:
            logger.warning(f"Serper returned no results for: {query}")
            return {
                "text": "No search results found for your query.",
                "sources": [],
                "meta": {"tool": "serper", "num_results": 0}
            }
        
        # Format results
        formatted_text = []
        sources = []
        
        for i, result in enumerate(organic_results[:max_results], 1):
            title = result.get('title', 'No title')
            snippet = result.get('snippet', '')
            link = result.get('link', '')
            
            # Clean and truncate
            title = safe_text(title, max_length=150)
            snippet = safe_text(snippet, max_length=250)
            
            # Format as numbered list
            formatted_text.append(f"[{i}] {title}")
            if snippet:
                formatted_text.append(f"    {snippet}")
            if link:
                formatted_text.append(f"    Source: {link}")
                sources.append(link)
            formatted_text.append("")  # Empty line between results
        
        result_text = "\n".join(formatted_text).strip()
        
        logger.info(f"Serper found {len(organic_results)} results")
        
        return {
            "text": result_text,
            "sources": sources,
            "meta": {
                "tool": "serper",
                "num_results": len(organic_results),
                "query": query,
                "search_time": data.get('searchParameters', {}).get('time', 'N/A')
            }
        }
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error("Serper API authentication failed - invalid API key")
            return {
                "text": "Serper API authentication failed. Please check your SERPER_API_KEY.",
                "sources": [],
                "meta": {"tool": "serper", "error": "auth_failed"}
            }
        elif e.response.status_code == 429:
            logger.error("Serper API rate limit exceeded")
            return {
                "text": "Search quota exceeded. Please try again later.",
                "sources": [],
                "meta": {"tool": "serper", "error": "rate_limit"}
            }
        else:
            logger.error(f"Serper API HTTP error: {e}")
            raise
    
    except Exception as e:
        logger.error(f"Serper search failed: {e}")
        raise


class SerperTool:
    """
    LangChain-compatible Serper search tool.
    """
    
    name = "web_search"
    description = """Useful for searching the web for current facts, news, and information using Google search.
    Input should be a search query string.
    Returns formatted search results with sources."""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize Serper tool.
        
        Args:
            max_results: Maximum search results to return
        """
        self.max_results = max_results
    
    def _run(self, query: str) -> str:
        """
        Run the tool (LangChain interface).
        
        Args:
            query: Search query
        
        Returns:
            Formatted search results as string
        """
        result = run_serper(query, max_results=self.max_results)
        
        # Format for LangChain
        output = result["text"]
        if result["sources"]:
            sources_str = ", ".join(result["sources"])
            output += f"\n\nSOURCES: {sources_str}"
        
        return output
    
    async def _arun(self, query: str) -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(query)


if __name__ == "__main__":
    # Test Serper tool
    print("Testing Serper search tool...")
    
    try:
        # Test basic search
        result = run_serper("OpenAI CEO", max_results=3)
        
        assert "text" in result
        assert "sources" in result
        assert "meta" in result
        assert result["meta"]["tool"] == "serper"
        
        print(f"✓ Search returned {result['meta']['num_results']} results")
        print(f"✓ Found {len(result['sources'])} sources")
        print("\nSample output:")
        print(result["text"][:300] + "...")
        
        # Test LangChain-compatible tool
        tool = SerperTool(max_results=2)
        output = tool._run("Python programming")
        assert len(output) > 0
        print(f"\n✓ LangChain tool interface works")
        
        print("\n✓ All Serper tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: This test requires SERPER_API_KEY in environment")
