"""
UI Helper Functions for Streamlit Interface
Handles message rendering, file uploads, and source display.
"""

import streamlit as st
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def render_message(role: str, text: str, sources: Optional[List[str]] = None, tool_calls: Optional[List[Dict]] = None):
    """
    Render a chat message with optional sources and tool trace.
    
    Args:
        role: Message role ('user' or 'assistant')
        text: Message text content
        sources: Optional list of source URLs/references
        tool_calls: Optional list of tool call dictionaries
    """
    # Determine avatar
    avatar = "🧑" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        # Display message text
        st.markdown(text)
        
        # Display sources if available
        if sources and len(sources) > 0:
            render_sources(sources)
        
        # Display tool trace if available
        if tool_calls and len(tool_calls) > 0:
            with st.expander(f"🔧 Tool Calls ({len(tool_calls)})", expanded=False):
                for i, call in enumerate(tool_calls, 1):
                    st.markdown(f"**{i}. {call.get('tool', 'unknown')}**")
                    st.code(f"Query: {call.get('query', 'N/A')}", language="text")
                    result_preview = str(call.get('result', ''))[:200]
                    st.text(f"Result: {result_preview}...")


def render_sources(sources: List[str]):
    """
    Render a list of sources with clickable links.
    
    Args:
        sources: List of source URLs or references
    """
    if not sources:
        return
    
    st.markdown("**📚 Sources:**")
    
    for i, source in enumerate(sources, 1):
        # Check if it's a URL or file reference
        if source.startswith('http://') or source.startswith('https://'):
            # Clickable URL
            st.markdown(f"{i}. [{source}]({source})")
        else:
            # File reference (not clickable)
            st.markdown(f"{i}. `{source}`")


def handle_file_upload(uploaded_file, rag_tool) -> Dict:
    """
    Handle file upload and ingestion.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        rag_tool: RAGTool instance
    
    Returns:
        Dictionary with upload result
    """
    try:
        # Create uploads directory
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        original_name = uploaded_file.name
        safe_name = f"{file_id}_{original_name}"
        file_path = uploads_dir / safe_name
        
        # Save uploaded file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Saved uploaded file: {safe_name}")
        
        # Ingest into RAG
        with st.spinner(f"Processing {original_name}..."):
            result = rag_tool.ingest_file(str(file_path), doc_id=file_id)
        
        if result.get('success'):
            return {
                'success': True,
                'filename': original_name,
                'chunks': result.get('chunks_added', 0),
                'message': f"✅ Successfully processed {original_name} ({result.get('chunks_added', 0)} chunks)"
            }
        else:
            return {
                'success': False,
                'filename': original_name,
                'error': result.get('error', 'Unknown error'),
                'message': f"❌ Failed to process {original_name}: {result.get('error', 'Unknown error')}"
            }
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return {
            'success': False,
            'filename': uploaded_file.name if uploaded_file else 'unknown',
            'error': str(e),
            'message': f"❌ Upload error: {str(e)}"
        }


def render_sidebar(agent, rag_tool):
    """
    Render sidebar with controls and information.
    
    Args:
        agent: ConversationalAgent instance
        rag_tool: RAGTool instance
    
    Returns:
        Dictionary with sidebar actions
    """
    actions = {}
    
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # LLM Provider Info
        import os
        provider = os.getenv('LLM_PROVIDER', 'GEMINI')
        st.info(f"**LLM Provider:** {provider}")
        
        # Tool Selection
        st.subheader("🔧 Tool Capabilities")
        tool_mode = st.radio(
            "Select Tool Mode:",
            ["Auto (Default)", "Web Search", "Wikipedia", "Document Search"],
            help="Force a specific tool or let the AI decide",
            index=0
        )
        
        # Map selection to internal tool names
        tool_map = {
            "Auto (Default)": None,
            "Web Search": "web_search",
            "Wikipedia": "wikipedia",
            "Document Search": "doc_search"
        }
        actions['forced_tool'] = tool_map.get(tool_mode)
        
        # Memory Stats
        st.subheader("💾 Memory")
        stats = agent.get_memory_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats.get('total_messages', 0))
        with col2:
            st.metric("Tokens", f"~{stats.get('estimated_tokens', 0)}")
        
        # Clear Memory Button
        if st.button("🗑️ Clear Memory", use_container_width=True):
            agent.clear_memory()
            actions['clear_memory'] = True
            st.success("Memory cleared!")
            st.rerun()
        
        # Document Upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload documents to query with doc_search tool"
        )
        
        if uploaded_files:
            actions['uploaded_files'] = []
            for uploaded_file in uploaded_files:
                result = handle_file_upload(uploaded_file, rag_tool)
                actions['uploaded_files'].append(result)
                
                if result['success']:
                    st.success(result['message'])
                else:
                    st.error(result['message'])
        
        # RAG Stats
        rag_stats = rag_tool.get_stats()
        if rag_stats['total_chunks'] > 0:
            st.metric("Document Chunks", rag_stats['total_chunks'])
        
        # Export Conversation
        st.subheader("💾 Export")
        if st.button("📥 Export Conversation", use_container_width=True):
            export_path = agent.export_conversation()
            st.success(f"Exported to: {export_path}")
            actions['export'] = export_path
    
    return actions


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False


def display_welcome_message():
    """Display welcome message for new sessions."""
    st.markdown("""
    ### 👋 Welcome to the Conversational Knowledge Bot!
    
    I can help you with:
    - 🔍 **Web Search**: Current facts and news (DuckDuckGo)
    - 📚 **Wikipedia**: Encyclopedic knowledge
    - 📄 **Document Q&A**: Upload and query your documents
    - 🧠 **Memory**: Remember our conversation
    
    **Try asking:**
    - "Who is the CEO of OpenAI?"
    - "What is quantum computing?"
    - Upload a PDF and ask "What does this document say about...?"
    """)


def show_error_message(error: str):
    """
    Display error message in a user-friendly way.
    
    Args:
        error: Error message
    """
    st.error(f"⚠️ **Error:** {error}")
    
    # Provide helpful hints for common errors
    if "API key" in error or "GEMINI_API_KEY" in error or "DEEPSEEK_API_KEY" in error:
        st.info("""
        **API Key Missing:** Please ensure you have:
        1. Created a `.env` file in the project root
        2. Added your API key: `GEMINI_API_KEY=your_key_here` or `DEEPSEEK_API_KEY=your_key_here`
        3. Restarted the application
        """)


def format_thinking_process(intermediate_steps: List) -> str:
    """
    Format agent's thinking process for display.
    
    Args:
        intermediate_steps: List of (action, observation) tuples
    
    Returns:
        Formatted markdown string
    """
    if not intermediate_steps:
        return ""
    
    formatted = ["**🤔 Thinking Process:**\n"]
    
    for i, (action, observation) in enumerate(intermediate_steps, 1):
        formatted.append(f"{i}. **Action:** {action.tool}")
        formatted.append(f"   **Input:** {action.tool_input}")
        formatted.append(f"   **Result:** {observation[:100]}...\n")
    
    return "\n".join(formatted)


if __name__ == "__main__":
    print("UI Helpers module - import this in main.py")
