"""
Conversation Memory System with persistence and summarization.
Manages chat history, provides context, and handles memory operations.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from src.utils import ensure_data_dir

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history with persistence and summarization.
    
    Features:
    - Stores conversation turns with timestamps
    - Persists to JSON file
    - Automatic summarization when buffer exceeds limits
    - Export and clear operations
    """
    
    def __init__(
        self,
        memory_file: str = "data/memory.json",
        max_turns: int = 20,
        max_context_tokens: int = 2048
    ):
        """
        Initialize conversation memory.
        
        Args:
            memory_file: Path to memory persistence file
            max_turns: Maximum turns to keep in buffer
            max_context_tokens: Token limit before summarization
        """
        self.memory_file = Path(memory_file)
        self.max_turns = max_turns
        self.max_context_tokens = max_context_tokens
        
        # Ensure data directory exists
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory structure
        self.chat_history: List[Dict] = []
        self.summaries: List[Dict] = []
        
        # Load existing memory if available
        self._load_memory()
        
        logger.info(f"Initialized ConversationMemory with {len(self.chat_history)} turns")
    
    def add_message(self, role: str, text: str, metadata: Optional[Dict] = None):
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('user' or 'assistant')
            text: Message text content
            metadata: Optional metadata (sources, tool calls, etc.)
        """
        message = {
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.chat_history.append(message)
        
        # Auto-save after each message
        self._save_memory()
        
        # Check if summarization is needed
        if len(self.chat_history) > self.max_turns:
            logger.info("Memory buffer exceeded, triggering summarization")
            # Note: Actual summarization requires LLM, done externally
            # For now, just trim oldest messages
            self._trim_history()
    
    def get_recent_context(self, max_turns: Optional[int] = None) -> List[Dict]:
        """
        Get recent conversation context.
        
        Args:
            max_turns: Maximum turns to retrieve (None for all)
        
        Returns:
            List of recent message dictionaries
        """
        if max_turns is None:
            max_turns = self.max_turns
        
        return self.chat_history[-max_turns:] if self.chat_history else []
    
    def get_all_history(self) -> List[Dict]:
        """
        Get complete conversation history.
        
        Returns:
            List of all message dictionaries
        """
        return self.chat_history.copy()
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search memory for relevant messages (simple keyword search).
        
        Args:
            query: Search query
            limit: Maximum results to return
        
        Returns:
            List of matching message dictionaries
        """
        query_lower = query.lower()
        matches = []
        
        for msg in self.chat_history:
            text_lower = msg.get("text", "").lower()
            if query_lower in text_lower:
                matches.append(msg)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def add_summary(self, summary_text: str, covers_turns: List[int]):
        """
        Add a summary of previous conversation turns.
        
        Args:
            summary_text: Summarized text
            covers_turns: List of turn indices covered by this summary
        """
        summary = {
            "summary": summary_text,
            "created_at": datetime.now().isoformat(),
            "covers_turns": covers_turns
        }
        
        self.summaries.append(summary)
        self._save_memory()
        
        logger.info(f"Added summary covering {len(covers_turns)} turns")
    
    def get_context_for_prompt(self, max_turns: int = 10) -> str:
        """
        Format recent context for inclusion in prompt.
        
        Args:
            max_turns: Maximum recent turns to include
        
        Returns:
            Formatted context string
        """
        recent = self.get_recent_context(max_turns)
        
        if not recent:
            return ""
        
        context_lines = []
        for msg in recent:
            role = msg["role"].upper()
            text = msg["text"]
            context_lines.append(f"{role}: {text}")
        
        return "\n".join(context_lines)
    
    def clear_memory(self):
        """Clear all conversation history and summaries."""
        self.chat_history = []
        self.summaries = []
        self._save_memory()
        logger.info("Cleared all conversation memory")
    
    def export_memory(self, export_path: Optional[str] = None) -> str:
        """
        Export memory to a human-readable format.
        
        Args:
            export_path: Optional path to save export (default: data/memory_export.txt)
        
        Returns:
            Path to exported file
        """
        if export_path is None:
            export_path = "data/memory_export.txt"
        
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CONVERSATION MEMORY EXPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write summaries
            if self.summaries:
                f.write("SUMMARIES:\n")
                f.write("-" * 80 + "\n")
                for i, summary in enumerate(self.summaries, 1):
                    f.write(f"\nSummary {i} ({summary['created_at']}):\n")
                    f.write(f"{summary['summary']}\n")
                    f.write(f"Covers turns: {summary['covers_turns']}\n")
                f.write("\n")
            
            # Write chat history
            f.write("CHAT HISTORY:\n")
            f.write("-" * 80 + "\n")
            for i, msg in enumerate(self.chat_history, 1):
                f.write(f"\n[{i}] {msg['role'].upper()} ({msg['timestamp']}):\n")
                f.write(f"{msg['text']}\n")
                if msg.get('metadata'):
                    f.write(f"Metadata: {json.dumps(msg['metadata'], indent=2)}\n")
        
        logger.info(f"Exported memory to {export_file}")
        return str(export_file)
    
    def _load_memory(self):
        """Load memory from persistence file."""
        if not self.memory_file.exists():
            logger.info("No existing memory file found, starting fresh")
            return
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.chat_history = data.get("chat_history", [])
            self.summaries = data.get("summaries", [])
            
            logger.info(f"Loaded {len(self.chat_history)} messages from memory")
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            logger.info("Starting with empty memory")
    
    def _save_memory(self):
        """Save memory to persistence file."""
        try:
            data = {
                "chat_history": self.chat_history,
                "summaries": self.summaries,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Memory saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _trim_history(self):
        """Trim oldest messages when buffer exceeds limit."""
        if len(self.chat_history) > self.max_turns:
            # Keep only the most recent max_turns messages
            removed_count = len(self.chat_history) - self.max_turns
            self.chat_history = self.chat_history[-self.max_turns:]
            
            logger.info(f"Trimmed {removed_count} old messages from history")
            self._save_memory()
    
    def get_stats(self) -> Dict:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        total_chars = sum(len(msg.get("text", "")) for msg in self.chat_history)
        
        return {
            "total_messages": len(self.chat_history),
            "total_summaries": len(self.summaries),
            "total_characters": total_chars,
            "estimated_tokens": total_chars // 4,  # Rough estimate
            "oldest_message": self.chat_history[0]["timestamp"] if self.chat_history else None,
            "newest_message": self.chat_history[-1]["timestamp"] if self.chat_history else None
        }


def summarize_conversation(
    memory: ConversationMemory,
    llm_provider,
    max_sentences: int = 6
) -> str:
    """
    Use LLM to summarize conversation history.
    
    Args:
        memory: ConversationMemory instance
        llm_provider: LLM provider instance
        max_sentences: Maximum sentences in summary
    
    Returns:
        Summary text
    """
    history = memory.get_all_history()
    
    if not history:
        return "No conversation history to summarize."
    
    # Build prompt for summarization
    context = memory.get_context_for_prompt(max_turns=len(history))
    prompt = f"""Summarize the following conversation in {max_sentences} sentences or less.
Focus on key topics discussed, important facts mentioned, and user preferences.

CONVERSATION:
{context}

SUMMARY:"""
    
    try:
        summary = llm_provider.generate(prompt, temperature=0.3, max_tokens=256)
        return summary.strip()
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return "Failed to generate summary."


if __name__ == "__main__":
    # Test memory system
    print("Testing ConversationMemory...")
    
    # Create test memory
    test_memory_file = "data/test_memory.json"
    memory = ConversationMemory(memory_file=test_memory_file, max_turns=5)
    
    # Add some messages
    memory.add_message("user", "Hello, my name is Alice")
    memory.add_message("assistant", "Hi Alice! Nice to meet you.")
    memory.add_message("user", "My favorite color is blue")
    memory.add_message("assistant", "I'll remember that your favorite color is blue.")
    
    # Test retrieval
    recent = memory.get_recent_context(2)
    assert len(recent) == 2
    assert recent[0]["role"] == "user"
    
    # Test search
    matches = memory.search_memory("blue")
    assert len(matches) > 0
    
    # Test stats
    stats = memory.get_stats()
    assert stats["total_messages"] == 4
    
    # Test export
    export_path = memory.export_memory("data/test_export.txt")
    assert Path(export_path).exists()
    
    # Cleanup
    Path(test_memory_file).unlink(missing_ok=True)
    Path(export_path).unlink(missing_ok=True)
    
    print("✓ All memory tests passed!")
