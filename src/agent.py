"""
Agent System - Orchestrates tools and memory for conversational AI.
Builds and manages the LangChain AgentExecutor.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.runnables import RunnablePassthrough

from src.llm_providers import get_llm_and_embedder
from src.memory import ConversationMemory
from src.prompts import SYSTEM_PROMPT, TOOL_POLICY_PROMPT, LLM_CONFIG
from src.tools.serper_tool import SerperTool  # Replaced DuckDuckGo with Serper
from src.tools.wiki_tool import WikipediaTool
from src.tools.rag_tool import RAGTool
from src.tools.memory_tool import MemoryTool  # Fixed: MemoryTool not MemoryQueryTool
from src.utils import log_tool_trace

logger = logging.getLogger(__name__)


class LLMWrapper(BaseLLM):
    """Wrapper to make our LLM provider compatible with LangChain."""
    
    provider: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, provider, **kwargs):
        super().__init__(**kwargs)
        self.provider = provider
    
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """Generate responses for the given prompts."""
        generations = []
        for prompt in prompts:
            text = self.provider.generate(
                prompt,
                temperature=kwargs.get('temperature', LLM_CONFIG['temperature']),
                max_tokens=kwargs.get('max_tokens', LLM_CONFIG['max_tokens']),
                **kwargs
            )
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type."""
        return "custom_llm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {"provider": str(self.provider.__class__.__name__)}


def build_agent_executor(config: Optional[Dict] = None) -> tuple:
    """
    Build and configure the agent executor with all tools and memory.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Tuple of (agent_executor, memory, rag_tool) for external use
    
    Example:
        >>> agent, memory, rag = build_agent_executor()
        >>> response = agent.invoke({"input": "Who is the CEO of OpenAI?"})
    """
    if config is None:
        config = {}
    
    logger.info("Building agent executor...")
    
    # Initialize LLM provider
    provider_name = config.get('llm_provider') or os.getenv('LLM_PROVIDER', 'GEMINI')
    llm_provider, embedder = get_llm_and_embedder(provider_name, config)
    
    # Initialize memory
    memory = ConversationMemory(
        memory_file=config.get('memory_file', 'data/memory.json'),
        max_turns=config.get('max_turns', 20),
        max_context_tokens=config.get('max_context_tokens', 2048)
    )
    
    # Initialize tools
    logger.info("Initializing tools...")
    
    # 1. Serper Web Search (Google-based)
    serper_tool = SerperTool(max_results=5)
    
    # 2. Wikipedia
    wiki_tool = WikipediaTool(sentences=3)
    
    # 3. RAG Document Search
    rag_tool = RAGTool(
        embedder=embedder,
        vector_store_dir=config.get('vector_store_dir'),
        chunk_size=config.get('chunk_size', 500),
        chunk_overlap=config.get('chunk_overlap', 100)
    )
    
    # 4. Memory Query
    memory_tool = MemoryTool(memory=memory, llm_provider=llm_provider)
    
    # Create tool dictionary for easy access
    tools_dict = {
        'web_search': serper_tool,
        'wikipedia': wiki_tool,
        'doc_search': rag_tool,
        'memory_query': memory_tool
    }
    
    logger.info(f"Registered {len(tools_dict)} tools: {list(tools_dict.keys())}")
    
    # Return simple dict instead of AgentExecutor
    agent_data = {
        'llm_provider': llm_provider,
        'tools': tools_dict,
        'memory': memory
    }
    
    logger.info("Agent executor built successfully")
    
    return agent_data, memory, rag_tool


class ConversationalAgent:
    """
    High-level conversational agent interface.
    Manages agent executor, memory, and tool traces.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize conversational agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.agent_data, self.memory, self.rag_tool = build_agent_executor(config)
        self.llm_provider = self.agent_data['llm_provider']
        self.tools = self.agent_data['tools']
        self.tool_traces = []
        
        logger.info("ConversationalAgent initialized")
    
    def chat(self, user_input: str, forced_tool: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user input and generate response using function calling.
        
        Args:
            user_input: User's message
            forced_tool: Optional name of tool to force usage of
        
        Returns:
            Dictionary with response, sources, and metadata
        """
        try:
            logger.info(f"Processing user input: {user_input[:100]}...")
            
            # Add user message to memory
            self.memory.add_message("user", user_input)
            
            # Get recent conversation context for better understanding
            recent_context = self.memory.get_recent_context(max_turns=3)
            
            # Build context-aware prompt
            if recent_context and len(recent_context) > 0:
                context_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['text']}"
                    for msg in recent_context[:-1]  # Exclude current message
                ])
                full_prompt = f"""Previous conversation:
{context_text}

Current question: {user_input}"""
            else:
                full_prompt = user_input
            
            # Define functions for Gemini
            functions = [
                {
                    'name': 'web_search',
                    'description': 'Search the web using DuckDuckGo for current information, news, people, companies, and recent events. Use this for questions about current CEOs, leaders, recent news, or any information that changes over time.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query'
                            }
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'wikipedia',
                    'description': 'Search Wikipedia for encyclopedic knowledge, historical facts, scientific concepts, definitions, and general knowledge. Use this for "what is" questions about concepts, not current people or events.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The Wikipedia search query'
                            }
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'doc_search',
                    'description': 'Search through uploaded documents. Only use this if the user explicitly mentions documents, files, PDFs, or uploaded content.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query for documents'
                            }
                        },
                        'required': ['query']
                    }
                },
                {
                    'name': 'memory_query',
                    'description': 'Search through conversation history or get a summary of what was discussed. Use this when the user asks about previous messages, earlier topics, or wants a conversation summary.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'What to search for in memory or "summarize" for a summary'
                            }
                        },
                        'required': ['query']
                    }
                }
            ]
            
            # Handle forced tool selection
            if forced_tool:
                logger.info(f"Forcing usage of tool: {forced_tool}")
                # Filter functions to only include the forced tool
                functions = [f for f in functions if f['name'] == forced_tool]
                
                # Add strong instruction to prompt
                full_prompt += f"\n\nIMPORTANT SYSTEM INSTRUCTION: You MUST use the '{forced_tool}' tool to answer this query. Do not answer directly. Generate the tool call immediately."
            
            # Call Gemini with function declarations and context
            response = self.llm_provider.generate_with_functions(
                prompt=full_prompt,
                functions=functions,
                temperature=0.7,
                max_tokens=1024
            )
            
            tool_calls = []
            sources = []
            final_response = None
            
            # Handle response
            if response['type'] == 'function_call':
                # Gemini wants to call a function
                function_name = response['function_name']
                function_args = response['function_args']
                query = function_args.get('query', user_input)
                
                logger.info(f"Gemini selected tool: {function_name} with query: {query}")
                
                # Execute the tool
                if function_name in self.tools:
                    try:
                        tool = self.tools[function_name]
                        tool_result = tool._run(query)
                        
                        tool_calls.append({
                            "tool": function_name,
                            "query": query,
                            "result": tool_result
                        })
                        
                        # Extract sources
                        if "SOURCES:" in tool_result:
                            sources_part = tool_result.split("SOURCES:")[-1].strip()
                            sources.extend([s.strip() for s in sources_part.split(",") if s.strip()])
                        
                        # Now ask Gemini to synthesize the final answer
                        synthesis_prompt = f"""Based on the following information from {function_name}, provide a clear and natural answer to the user's question.

User Question: {user_input}

Information from {function_name}:
{tool_result}

Provide a conversational answer. If sources are mentioned, you can reference them."""
                        
                        final_response = self.llm_provider.generate(synthesis_prompt, temperature=0.7, max_tokens=500)
                        
                    except Exception as e:
                        logger.error(f"Tool {function_name} failed: {e}")
                        final_response = f"I tried to use {function_name} but encountered an error: {str(e)}"
                else:
                    final_response = f"I wanted to use {function_name} but it's not available."
            
            elif response['type'] == 'text':
                # Gemini responded directly without needing tools
                final_response = response['text']
            
            else:
                # Error case
                final_response = response.get('text', "I apologize, but I couldn't generate a response.")
            
            # Add assistant response to memory
            self.memory.add_message("assistant", final_response, metadata={
                "sources": sources,
                "tool_calls": len(tool_calls)
            })
            
            # Log tool trace
            if tool_calls:
                log_tool_trace(
                    user_input=user_input,
                    tool_calls=tool_calls,
                    agent_response=final_response
                )
            
            # Store trace for UI display
            self.tool_traces.append({
                "user_input": user_input,
                "tool_calls": tool_calls,
                "response": final_response
            })
            
            logger.info(f"Generated response with {len(tool_calls)} tool calls")
            
            return {
                "response": final_response,
                "sources": list(set(sources)),  # Remove duplicates
                "tool_calls": tool_calls,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            
            error_response = f"I encountered an error while processing your request: {str(e)}"
            
            # Still add to memory
            self.memory.add_message("assistant", error_response, metadata={"error": str(e)})
            
            return {
                "response": error_response,
                "sources": [],
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """
        Upload and ingest a document.
        
        Args:
            file_path: Path to document file
        
        Returns:
            Ingestion result dictionary
        """
        logger.info(f"Uploading document: {file_path}")
        result = self.rag_tool.ingest_file(file_path)
        return result
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear_memory()
        self.tool_traces = []
        logger.info("Cleared conversation memory and tool traces")
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        return self.memory.get_stats()
    
    def get_last_tool_trace(self) -> Optional[Dict]:
        """Get the most recent tool trace."""
        return self.tool_traces[-1] if self.tool_traces else None
    
    def export_conversation(self, export_path: Optional[str] = None) -> str:
        """
        Export conversation history.
        
        Args:
            export_path: Optional path for export
        
        Returns:
            Path to exported file
        """
        return self.memory.export_memory(export_path)


if __name__ == "__main__":
    # Test agent system
    print("Testing Agent System...")
    print("Note: This requires valid API keys in .env file")
    
    try:
        # Build agent
        config = {
            'verbose': True,
            'max_iterations': 3
        }
        
        agent = ConversationalAgent(config)
        print("✓ Agent initialized successfully")
        
        # Test simple query (will use tools if API keys are available)
        print("\nTesting chat functionality...")
        response = agent.chat("What is 2+2?")
        
        if response["success"]:
            print(f"✓ Chat response: {response['response'][:100]}...")
            print(f"✓ Tool calls: {len(response['tool_calls'])}")
        
        # Test memory
        stats = agent.get_memory_stats()
        print(f"✓ Memory stats: {stats['total_messages']} messages")
        
        print("\n✓ Agent system tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: Full testing requires valid API keys")
