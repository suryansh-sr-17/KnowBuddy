"""
Utility functions for the conversational knowledge bot.
Includes retry logic, text processing, citation formatting, and logging.
"""

import time
import json
import logging
from functools import wraps
from typing import Callable, Any, List
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @exponential_backoff(max_retries=3)
        def fetch_data():
            # Code that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator


def safe_text(text: str, max_length: int = None) -> str:
    """
    Sanitize and truncate text for safe display.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum length (None for no limit)
    
    Returns:
        Sanitized text string
    """
    if not text:
        return ""
    
    # Remove null bytes and control characters
    text = text.replace('\x00', '').strip()
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length - 3] + "..."
    
    return text


def format_citations(sources: List[str]) -> str:
    """
    Format a list of sources into a citation string.
    
    Args:
        sources: List of source URLs or references
    
    Returns:
        Formatted citation string
    
    Example:
        >>> format_citations(["https://example.com", "doc.pdf#chunk-5"])
        "SOURCES: [1] https://example.com, [2] doc.pdf#chunk-5"
    """
    if not sources:
        return ""
    
    # Remove duplicates while preserving order
    unique_sources = []
    seen = set()
    for source in sources:
        if source and source not in seen:
            unique_sources.append(source)
            seen.add(source)
    
    if not unique_sources:
        return ""
    
    # Format with numbered citations
    citations = [f"[{i+1}] {source}" for i, source in enumerate(unique_sources)]
    return "SOURCES: " + ", ".join(citations)


def log_tool_trace(
    user_input: str,
    tool_calls: List[dict],
    agent_response: str,
    trace_file: str = "sample_chats/traces.jsonl"
):
    """
    Log tool execution traces to a JSONL file for debugging and analysis.
    
    Args:
        user_input: The user's query
        tool_calls: List of tool call dictionaries with tool name, query, and results
        agent_response: The final agent response
        trace_file: Path to the trace file (relative to project root)
    
    Example tool_calls format:
        [
            {
                "tool": "web_search",
                "query": "OpenAI CEO",
                "result": {"text": "...", "sources": [...]}
            }
        ]
    """
    try:
        # Create directory if it doesn't exist
        trace_path = Path(trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create trace entry
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "tool_calls": tool_calls,
            "agent_response": agent_response
        }
        
        # Append to JSONL file
        with open(trace_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace_entry, ensure_ascii=False) + '\n')
        
        logger.debug(f"Logged trace with {len(tool_calls)} tool calls")
        
    except Exception as e:
        logger.error(f"Failed to log tool trace: {e}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[dict]:
    """
    Split text into overlapping chunks for RAG processing.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens (approximated as words)
        overlap: Number of overlapping tokens between chunks
    
    Returns:
        List of chunk dictionaries with text, start_char, end_char
    
    Example:
        >>> chunks = chunk_text("Long document...", chunk_size=100, overlap=20)
        >>> chunks[0]
        {"text": "...", "start_char": 0, "end_char": 450, "chunk_index": 0}
    """
    if not text:
        return []
    
    # Approximate tokens as words (rough estimate: 1 token ≈ 0.75 words)
    # So chunk_size tokens ≈ chunk_size * 0.75 words ≈ chunk_size * 3.75 chars
    char_chunk_size = int(chunk_size * 3.75)
    char_overlap = int(overlap * 3.75)
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = min(start + char_chunk_size, len(text))
        
        # Try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence end
            sentence_end = text.rfind('. ', start, end)
            if sentence_end > start + char_chunk_size // 2:
                end = sentence_end + 1
            else:
                # Look for word boundary
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "chunk_index": chunk_index
            })
            chunk_index += 1
        
        # Move start position with overlap
        start = end - char_overlap if end < len(text) else end
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def ensure_data_dir(subdir: str = None) -> Path:
    """
    Ensure data directory exists and return path.
    
    Args:
        subdir: Optional subdirectory within data/
    
    Returns:
        Path object for the directory
    """
    base_path = Path("data")
    if subdir:
        path = base_path / subdir
    else:
        path = base_path
    
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test safe_text
    assert safe_text("Hello\x00World") == "Hello World"
    assert safe_text("Long text" * 100, max_length=20) == "Long textLong tex..."
    
    # Test format_citations
    sources = ["https://example.com", "doc.pdf#chunk-5", "https://example.com"]
    assert "SOURCES:" in format_citations(sources)
    
    # Test chunk_text
    text = "This is a test. " * 100
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0]["chunk_index"] == 0
    
    print("✓ All utility tests passed!")
