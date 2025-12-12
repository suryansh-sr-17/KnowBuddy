"""
Prompt templates and configurations for the conversational knowledge bot.
Contains system prompts, tool policies, and message formatting templates.
"""

# System prompt - defines the assistant's behavior and capabilities
SYSTEM_PROMPT = """You are an intelligent assistant that answers user questions using available tools when necessary.

Your capabilities:
- Answer factual questions using web search (DuckDuckGo) or Wikipedia
- Query and cite uploaded documents using document search
- Remember and recall previous conversation context
- Provide concise, accurate answers with proper citations

Guidelines:
- If the user asks for factual or up-to-date information, call tools (web_search or wikipedia)
- If the user asks about uploaded documents, call doc_search
- If the user asks about previous conversation or personal preferences, call memory_query
- Always prefer concise factual answers and include a "SOURCES:" list when tools were used
- If you use direct quotes from a tool, mark them as quotes and provide the source
- If you cannot answer, say "I don't know" and suggest how to find the answer
- Be helpful, accurate, and cite your sources

Remember: You have access to tools. Use them when needed to provide accurate, well-sourced answers.
"""

# Tool policy - detailed instructions for when and how to use each tool
TOOL_POLICY_PROMPT = """TOOL USAGE POLICY:

Available Tools:
1. web_search (DuckDuckGo)
   - Use for: Current facts, news, company information, recent events
   - Best for: Wide web coverage, up-to-date information
   - Example queries: "OpenAI CEO", "latest AI news", "Python 3.12 features"

2. wikipedia
   - Use for: Encyclopedic background, stable facts, historical information
   - Best for: People, places, concepts, definitions, dates
   - Example queries: "Albert Einstein", "quantum computing", "World War II"

3. doc_search
   - Use for: Questions about user-uploaded documents
   - Best for: Specific document content, manuals, reports, PDFs
   - Always cite: Include filename and chunk reference
   - Example queries: "reset procedure in manual", "quarterly revenue from report"

4. memory_query
   - Use for: Recalling previous conversation, user preferences, context
   - Best for: "What did I say about...", "My favorite...", conversation history
   - Example queries: "user's favorite color", "previous discussion about AI"

Tool Selection Strategy:
- Choose the MOST SPECIFIC tool for the query
- Prefer doc_search if the question references uploaded documents
- Use memory_query for context-dependent or personal questions
- Use web_search for current events and broad web information
- Use wikipedia for stable encyclopedic knowledge
- You can use multiple tools if needed, but prefer the minimum necessary

Output Format:
- After using tools, synthesize the answer in your own voice
- Always append "SOURCES: [url1, url2, ...]" on a new line when tools were used
- Keep answers concise but complete
- Use direct quotes sparingly and always attribute them
"""

# Tool output template - how to format tool results in the prompt
TOOL_OUTPUT_TEMPLATE = """TOOL_OUTPUT:
Tool: {tool_name}
Query: {query}
Result: {text}
Sources: {sources}
END_TOOL_OUTPUT
"""

# User message wrapper - format for user input
USER_MESSAGE_TEMPLATE = """USER: {message}"""

# Assistant message wrapper - format for assistant responses
ASSISTANT_MESSAGE_TEMPLATE = """ASSISTANT: {message}"""

# Memory context template - how to include conversation history
MEMORY_CONTEXT_TEMPLATE = """CONVERSATION HISTORY:
{history}
END_HISTORY
"""

# Error message templates
ERROR_NO_TOOL_RESULT = "I tried to use a tool but didn't get a result. Please try rephrasing your question."
ERROR_TOOL_FAILED = "I encountered an error while searching for information: {error}. Please try again."
ERROR_NO_ANSWER = "I don't have enough information to answer that question. You might try uploading relevant documents or asking a different question."

# LLM configuration parameters
LLM_CONFIG = {
    "temperature": 0.7,  # Balance between creativity and consistency
    "max_tokens": 1024,  # Maximum response length
    "top_p": 0.9,        # Nucleus sampling
    "stop_sequences": ["USER:", "TOOL_OUTPUT:"],  # Stop generation at these tokens
}

# RAG configuration
RAG_CONFIG = {
    "chunk_size": 500,        # Tokens per chunk
    "chunk_overlap": 100,     # Overlapping tokens
    "top_k": 4,               # Number of chunks to retrieve
    "similarity_threshold": 0.7,  # Minimum similarity score
}

# Memory configuration
MEMORY_CONFIG = {
    "max_turns": 20,          # Maximum conversation turns to keep in buffer
    "max_context_tokens": 2048,  # Token limit before summarization
    "summary_sentences": 6,   # Sentences in memory summary
}

# Search configuration
SEARCH_CONFIG = {
    "ddg_max_results": 5,     # DuckDuckGo results to fetch
    "wiki_sentences": 3,      # Wikipedia summary sentences
    "max_retries": 3,         # Retry attempts for failed requests
}


def format_tool_output(tool_name: str, query: str, text: str, sources: list) -> str:
    """
    Format tool output for inclusion in the prompt.
    
    Args:
        tool_name: Name of the tool that was called
        query: The query sent to the tool
        text: The text result from the tool
        sources: List of source URLs/references
    
    Returns:
        Formatted tool output string
    """
    sources_str = ", ".join(sources) if sources else "None"
    return TOOL_OUTPUT_TEMPLATE.format(
        tool_name=tool_name,
        query=query,
        text=text,
        sources=sources_str
    )


def format_memory_context(history: list) -> str:
    """
    Format conversation history for inclusion in the prompt.
    
    Args:
        history: List of message dictionaries with 'role' and 'text' keys
    
    Returns:
        Formatted memory context string
    """
    if not history:
        return ""
    
    formatted_messages = []
    for msg in history:
        role = msg.get("role", "unknown").upper()
        text = msg.get("text", "")
        formatted_messages.append(f"{role}: {text}")
    
    history_str = "\n".join(formatted_messages)
    return MEMORY_CONTEXT_TEMPLATE.format(history=history_str)


def build_agent_prompt(
    user_message: str,
    tool_outputs: list = None,
    memory_context: list = None
) -> str:
    """
    Build the complete prompt for the agent.
    
    Args:
        user_message: The current user message
        tool_outputs: List of tool output dictionaries (optional)
        memory_context: List of previous messages (optional)
    
    Returns:
        Complete formatted prompt string
    """
    prompt_parts = [SYSTEM_PROMPT, TOOL_POLICY_PROMPT]
    
    # Add memory context if available
    if memory_context:
        prompt_parts.append(format_memory_context(memory_context))
    
    # Add tool outputs if available
    if tool_outputs:
        for tool_output in tool_outputs:
            formatted = format_tool_output(
                tool_name=tool_output.get("tool", "unknown"),
                query=tool_output.get("query", ""),
                text=tool_output.get("text", ""),
                sources=tool_output.get("sources", [])
            )
            prompt_parts.append(formatted)
    
    # Add current user message
    prompt_parts.append(USER_MESSAGE_TEMPLATE.format(message=user_message))
    
    return "\n\n".join(prompt_parts)


if __name__ == "__main__":
    # Test prompt building
    print("Testing prompt templates...")
    
    # Test tool output formatting
    tool_out = format_tool_output(
        tool_name="web_search",
        query="OpenAI CEO",
        text="Sam Altman is the CEO",
        sources=["https://openai.com"]
    )
    assert "web_search" in tool_out
    assert "Sam Altman" in tool_out
    
    # Test memory context formatting
    history = [
        {"role": "user", "text": "Hello"},
        {"role": "assistant", "text": "Hi there!"}
    ]
    context = format_memory_context(history)
    assert "USER: Hello" in context
    
    # Test full prompt building
    prompt = build_agent_prompt(
        user_message="Who is the CEO?",
        tool_outputs=[{"tool": "web_search", "query": "CEO", "text": "Info", "sources": []}],
        memory_context=history
    )
    assert "SYSTEM" in prompt or "assistant" in prompt.lower()
    assert "Who is the CEO?" in prompt
    
    print("✓ All prompt tests passed!")
