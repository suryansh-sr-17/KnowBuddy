"""
Memory Query Tool
Provides access to conversation history and memory summarization.
"""

import logging
from typing import Dict, Optional
from src.memory import ConversationMemory, summarize_conversation

logger = logging.getLogger(__name__)


class MemoryTool:
    """
    Tool for querying and summarizing conversation memory.
    
    Features:
    - Summarize conversation history
    - Search memory for specific content
    - Retrieve recent context
    """
    
    name = "memory_query"
    description = """Useful for recalling previous conversation, user preferences, or context.
    Input should be a query about what was discussed earlier or what the user mentioned.
    Returns relevant information from conversation history."""
    
    def __init__(self, memory: ConversationMemory, llm_provider=None):
        """
        Initialize Memory tool.
        
        Args:
            memory: ConversationMemory instance
            llm_provider: Optional LLM provider for summarization
        """
        self.memory = memory
        self.llm_provider = llm_provider
        logger.info("Initialized Memory Query tool")
    
    def get_summary(self, limit_sentences: int = 6) -> Dict:
        """
        Get a summary of the conversation history.
        
        Args:
            limit_sentences: Maximum sentences in summary
        
        Returns:
            Dictionary with standardized format
        """
        try:
            logger.info("Generating memory summary")
            
            if not self.memory.chat_history:
                return {
                    "text": "No conversation history available yet.",
                    "sources": [],
                    "meta": {"tool": "memory_query", "type": "summary", "messages": 0}
                }
            
            # If LLM provider available, use it for summarization
            if self.llm_provider:
                summary_text = summarize_conversation(
                    self.memory,
                    self.llm_provider,
                    max_sentences=limit_sentences
                )
            else:
                # Fallback: simple concatenation of recent messages
                recent = self.memory.get_recent_context(max_turns=10)
                summary_parts = []
                
                for msg in recent:
                    role = msg["role"]
                    text = msg["text"][:100]  # Truncate for summary
                    summary_parts.append(f"{role.capitalize()}: {text}")
                
                summary_text = "\n".join(summary_parts)
            
            stats = self.memory.get_stats()
            
            return {
                "text": summary_text,
                "sources": [],
                "meta": {
                    "tool": "memory_query",
                    "type": "summary",
                    "messages": stats["total_messages"],
                    "sentences": limit_sentences
                }
            }
        
        except Exception as e:
            logger.error(f"Memory summary failed: {e}")
            return {
                "text": f"Failed to generate memory summary: {str(e)}",
                "sources": [],
                "meta": {"tool": "memory_query", "error": str(e)}
            }
    
    def query(self, query: str, top_k: int = 5) -> Dict:
        """
        Search memory for relevant messages.
        
        Args:
            query: Search query
            top_k: Maximum results to return
        
        Returns:
            Dictionary with standardized format
        """
        try:
            logger.info(f"Memory query: {query}")
            
            if not self.memory.chat_history:
                return {
                    "text": "No conversation history to search.",
                    "sources": [],
                    "meta": {"tool": "memory_query", "type": "search", "matches": 0}
                }
            
            # Search memory (simple keyword search)
            matches = self.memory.search_memory(query, limit=top_k)
            
            if not matches:
                return {
                    "text": f"No relevant information found in conversation history for: {query}",
                    "sources": [],
                    "meta": {"tool": "memory_query", "type": "search", "matches": 0}
                }
            
            # Format matches
            formatted_results = []
            for i, msg in enumerate(matches, 1):
                role = msg["role"].upper()
                text = msg["text"]
                timestamp = msg.get("timestamp", "unknown")
                
                formatted_results.append(f"[{i}] {role} ({timestamp}):")
                formatted_results.append(f"    {text}")
                formatted_results.append("")  # Empty line
            
            result_text = "\n".join(formatted_results).strip()
            
            logger.info(f"Found {len(matches)} matching messages")
            
            return {
                "text": result_text,
                "sources": [],
                "meta": {
                    "tool": "memory_query",
                    "type": "search",
                    "matches": len(matches),
                    "query": query
                }
            }
        
        except Exception as e:
            logger.error(f"Memory query failed: {e}")
            return {
                "text": f"Memory search error: {str(e)}",
                "sources": [],
                "meta": {"tool": "memory_query", "error": str(e)}
            }
    
    def _run(self, query: str) -> str:
        """
        Run the tool (LangChain interface).
        
        Automatically determines whether to summarize or search based on query.
        
        Args:
            query: Query string
        
        Returns:
            Formatted memory results as string
        """
        query_lower = query.lower()
        
        # Determine intent
        summary_keywords = ["summarize", "summary", "overview", "what have we discussed"]
        is_summary_request = any(keyword in query_lower for keyword in summary_keywords)
        
        if is_summary_request:
            result = self.get_summary()
        else:
            result = self.query(query)
        
        # Format for LangChain
        output = result["text"]
        
        return output
    
    async def _arun(self, query: str) -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(query)


# Convenience functions for standalone use
def get_memory_summary(memory: ConversationMemory, llm_provider=None, limit_sentences: int = 6) -> Dict:
    """
    Standalone function to get memory summary.
    
    Args:
        memory: ConversationMemory instance
        llm_provider: Optional LLM provider
        limit_sentences: Maximum sentences
    
    Returns:
        Summary result dictionary
    """
    tool = MemoryTool(memory, llm_provider)
    return tool.get_summary(limit_sentences)


def query_memory(memory: ConversationMemory, query: str, top_k: int = 5) -> Dict:
    """
    Standalone function to query memory.
    
    Args:
        memory: ConversationMemory instance
        query: Search query
        top_k: Maximum results
    
    Returns:
        Query result dictionary
    """
    tool = MemoryTool(memory, llm_provider=None)
    return tool.query(query, top_k)


if __name__ == "__main__":
    # Test memory tool
    print("Testing Memory Query tool...")
    
    from src.memory import ConversationMemory
    
    # Create test memory
    memory = ConversationMemory(memory_file="data/test_memory_tool.json")
    
    # Add some test messages
    memory.add_message("user", "My name is Alice")
    memory.add_message("assistant", "Nice to meet you, Alice!")
    memory.add_message("user", "I like Python programming")
    memory.add_message("assistant", "Python is a great language!")
    
    # Test summary
    tool = MemoryTool(memory)
    summary_result = tool.get_summary(limit_sentences=3)
    
    assert "text" in summary_result
    assert summary_result["meta"]["tool"] == "memory_query"
    print(f"✓ Summary generated: {len(summary_result['text'])} characters")
    
    # Test query
    query_result = tool.query("Python", top_k=3)
    assert query_result["meta"]["matches"] > 0
    print(f"✓ Query found {query_result['meta']['matches']} matches")
    
    # Test LangChain interface
    output = tool._run("What did I say about programming?")
    assert len(output) > 0
    print("✓ LangChain tool interface works")
    
    # Cleanup
    from pathlib import Path
    Path("data/test_memory_tool.json").unlink(missing_ok=True)
    
    print("\n✓ All Memory tool tests passed!")
