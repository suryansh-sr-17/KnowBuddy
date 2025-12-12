"""
DuckDuckGo Web Search Tool
Provides web search functionality with retry logic and static fallback.
"""

import os
import json
import logging
from typing import Dict, List
from pathlib import Path
from duckduckgo_search import DDGS
from src.utils import exponential_backoff, safe_text

logger = logging.getLogger(__name__)


@exponential_backoff(max_retries=3, base_delay=1.0)
def run_ddg(query: str, max_results: int = 5) -> Dict:
    """
    Perform DuckDuckGo web search.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        Dictionary with standardized format:
        {
            "text": "Formatted search results",
            "sources": ["url1", "url2", ...],
            "meta": {"tool": "duckduckgo", "num_results": N}
        }
    
    Example:
        >>> result = run_ddg("OpenAI CEO")
        >>> print(result["text"])
        - Sam Altman - CEO of OpenAI
          Summary: Sam Altman is the CEO...
          Source: https://openai.com
    """
    # Check if static KB mode is enabled
    use_static = os.getenv('USE_STATIC_KB', 'false').lower() == 'true'
    if use_static:
        return _static_kb_search(query)
    
    try:
        logger.info(f"DuckDuckGo search: {query}")
        
        # Perform search using DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            logger.warning(f"DuckDuckGo returned no results for: {query}")
            logger.info("Attempting Wikipedia fallback...")
            
            # Try Wikipedia as fallback
            try:
                from src.tools.wiki_tool import run_wikipedia
                wiki_result = run_wikipedia(query, sentences=3)
                
                if wiki_result["text"] and "not found" not in wiki_result["text"].lower():
                    logger.info("Wikipedia fallback successful")
                    return {
                        "text": f"(DuckDuckGo unavailable, using Wikipedia)\n\n{wiki_result['text']}",
                        "sources": wiki_result["sources"],
                        "meta": {"tool": "duckduckgo_wikipedia_fallback", "num_results": 1}
                    }
            except Exception as wiki_error:
                logger.error(f"Wikipedia fallback also failed: {wiki_error}")
            
            return {
                "text": "No web search results found for your query. The search service may be temporarily unavailable. Please try again in a moment.",
                "sources": [],
                "meta": {" tool": "duckduckgo", "num_results": 0}
            }
        
        # Format results
        formatted_text = []
        sources = []
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', '')
            href = result.get('href', '')
            
            # Clean and truncate
            title = safe_text(title, max_length=100)
            body = safe_text(body, max_length=200)
            
            # Format as bullet point
            formatted_text.append(f"[{i}] {title}")
            if body:
                formatted_text.append(f"    {body}")
            if href:
                formatted_text.append(f"    Source: {href}")
                sources.append(href)
            formatted_text.append("")  # Empty line between results
        
        result_text = "\n".join(formatted_text).strip()
        
        logger.info(f"DuckDuckGo found {len(results)} results")
        
        return {
            "text": result_text,
            "sources": sources,
            "meta": {
                "tool": "duckduckgo",
                "num_results": len(results),
                "query": query
            }
        }
    
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        raise


def _static_kb_search(query: str) -> Dict:
    """
    Fallback to static knowledge base when USE_STATIC_KB=true.
    
    Args:
        query: Search query
    
    Returns:
        Search result from static KB or empty result
    """
    static_kb_path = os.getenv('STATIC_KB_PATH', 'data/static_kb.json')
    
    try:
        if not Path(static_kb_path).exists():
            logger.warning(f"Static KB file not found: {static_kb_path}")
            return {
                "text": "Static knowledge base not available.",
                "sources": [],
                "meta": {"tool": "duckduckgo", "mode": "static", "num_results": 0}
            }
        
        with open(static_kb_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        # Simple keyword matching
        query_lower = query.lower()
        matches = []
        
        for entry in kb_data.get('entries', []):
            if query_lower in entry.get('keywords', '').lower():
                matches.append(entry)
        
        if not matches:
            return {
                "text": f"No results found in static knowledge base for: {query}",
                "sources": [],
                "meta": {"tool": "duckduckgo", "mode": "static", "num_results": 0}
            }
        
        # Format matches
        formatted_text = []
        sources = []
        
        for i, entry in enumerate(matches[:5], 1):
            title = entry.get('title', 'No title')
            content = entry.get('content', '')
            source = entry.get('source', '')
            
            formatted_text.append(f"[{i}] {title}")
            formatted_text.append(f"    {content}")
            if source:
                formatted_text.append(f"    Source: {source}")
                sources.append(source)
            formatted_text.append("")
        
        return {
            "text": "\n".join(formatted_text).strip(),
            "sources": sources,
            "meta": {"tool": "duckduckgo", "mode": "static", "num_results": len(matches)}
        }
    
    except Exception as e:
        logger.error(f"Static KB search failed: {e}")
        return {
            "text": f"Error accessing static knowledge base: {e}",
            "sources": [],
            "meta": {"tool": "duckduckgo", "mode": "static", "error": str(e)}
        }


class DuckDuckGoTool:
    """
    LangChain-compatible DuckDuckGo search tool.
    """
    
    name = "web_search"
    description = """Useful for searching the web for current facts, news, and information.
    Input should be a search query string.
    Returns formatted search results with sources."""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize DuckDuckGo tool.
        
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
        result = run_ddg(query, max_results=self.max_results)
        
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
    # Test DuckDuckGo tool
    print("Testing DuckDuckGo search tool...")
    
    try:
        # Test basic search
        result = run_ddg("Python programming language", max_results=3)
        
        assert "text" in result
        assert "sources" in result
        assert "meta" in result
        assert result["meta"]["tool"] == "duckduckgo"
        
        print(f"✓ Search returned {result['meta']['num_results']} results")
        print(f"✓ Found {len(result['sources'])} sources")
        print("\nSample output:")
        print(result["text"][:300] + "...")
        
        # Test LangChain-compatible tool
        tool = DuckDuckGoTool(max_results=2)
        output = tool._run("OpenAI")
        assert len(output) > 0
        print(f"\n✓ LangChain tool interface works")
        
        print("\n✓ All DuckDuckGo tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: This test requires internet connection")
