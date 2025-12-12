"""
Wikipedia Search Tool
Provides encyclopedic knowledge with citations.
"""

import logging
import wikipedia
from typing import Dict
from src.utils import safe_text

logger = logging.getLogger(__name__)


def run_wiki(query: str, sentences: int = 3) -> Dict:
    """
    Search Wikipedia and return summary.
    
    Args:
        query: Search query string
        sentences: Number of sentences to include in summary
    
    Returns:
        Dictionary with standardized format:
        {
            "text": "Wikipedia summary",
            "sources": ["https://en.wikipedia.org/wiki/..."],
            "meta": {"tool": "wikipedia", "title": "Article Title"}
        }
    
    Example:
        >>> result = run_wiki("Albert Einstein", sentences=2)
        >>> print(result["text"])
        Albert Einstein was a German-born theoretical physicist...
    """
    try:
        logger.info(f"Wikipedia search: {query}")
        
        # Search for pages
        search_results = wikipedia.search(query, results=3)
        
        if not search_results:
            return {
                "text": f"No Wikipedia page found for: {query}",
                "sources": [],
                "meta": {"tool": "wikipedia", "query": query, "found": False}
            }
        
        # Get the first result
        page_title = search_results[0]
        
        try:
            # Get page summary
            summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)
            
            # Generate Wikipedia URL
            url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            
            # Clean summary
            summary = safe_text(summary, max_length=1000)
            
            logger.info(f"Wikipedia found: {page_title}")
            
            return {
                "text": summary,
                "sources": [url],
                "meta": {
                    "tool": "wikipedia",
                    "title": page_title,
                    "query": query,
                    "found": True
                }
            }
        
        except wikipedia.DisambiguationError as e:
            # Handle disambiguation pages
            options = e.options[:5]  # Get first 5 options
            options_text = ", ".join(options)
            
            return {
                "text": f"'{query}' is ambiguous. Did you mean one of these? {options_text}",
                "sources": [],
                "meta": {
                    "tool": "wikipedia",
                    "query": query,
                    "disambiguation": True,
                    "options": options
                }
            }
        
        except wikipedia.PageError:
            return {
                "text": f"Wikipedia page not found for: {query}",
                "sources": [],
                "meta": {"tool": "wikipedia", "query": query, "found": False}
            }
    
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return {
            "text": f"Wikipedia search error: {str(e)}",
            "sources": [],
            "meta": {"tool": "wikipedia", "error": str(e)}
        }


class WikipediaTool:
    """
    LangChain-compatible Wikipedia search tool.
    """
    
    name = "wikipedia"
    description = """Useful for getting encyclopedic information about people, places, concepts, and historical events.
    Input should be a search query for a Wikipedia article.
    Returns a summary with citation."""
    
    def __init__(self, sentences: int = 3):
        """
        Initialize Wikipedia tool.
        
        Args:
            sentences: Number of sentences to include in summary
        """
        self.sentences = sentences
    
    def _run(self, query: str) -> str:
        """
        Run the tool (LangChain interface).
        
        Args:
            query: Search query
        
        Returns:
            Formatted Wikipedia summary as string
        """
        result = run_wiki(query, sentences=self.sentences)
        
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
    # Test Wikipedia tool
    print("Testing Wikipedia search tool...")
    
    try:
        # Test 1: Basic search
        result = run_wiki("Python programming language", sentences=2)
        
        assert "text" in result
        assert "sources" in result
        assert "meta" in result
        assert result["meta"]["tool"] == "wikipedia"
        
        if result["meta"].get("found"):
            print(f"✓ Found Wikipedia page: {result['meta']['title']}")
            print(f"✓ Summary length: {len(result['text'])} characters")
            print(f"✓ Source: {result['sources'][0]}")
        
        # Test 2: Person search
        result2 = run_wiki("Albert Einstein", sentences=3)
        if result2["meta"].get("found"):
            print(f"✓ Person search works: {result2['meta']['title']}")
        
        # Test 3: Non-existent page
        result3 = run_wiki("XYZ123NonExistentPageABC", sentences=1)
        assert not result3["meta"].get("found")
        print("✓ Handles non-existent pages correctly")
        
        # Test 4: LangChain-compatible tool
        tool = WikipediaTool(sentences=2)
        output = tool._run("Quantum computing")
        assert len(output) > 0
        print("✓ LangChain tool interface works")
        
        print("\n✓ All Wikipedia tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: This test requires internet connection")
