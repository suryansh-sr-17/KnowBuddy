"""
Main Streamlit Application
Conversational Knowledge Bot with multi-tool agent.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import logging
from dotenv import load_dotenv

from src.agent import ConversationalAgent
from app.ui_helpers import (
    render_message,
    render_sidebar,
    initialize_session_state,
    display_welcome_message,
    show_error_message
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Conversational Knowledge Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_agent():
    """Initialize the conversational agent."""
    try:
        with st.spinner("🔄 Initializing agent..."):
            config = {
                'verbose': os.getenv('LOG_LEVEL', 'INFO') == 'DEBUG',
                'max_iterations': 5
            }
            agent = ConversationalAgent(config)
            logger.info("Agent initialized successfully")
            return agent, None
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return None, str(e)


def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🤖 Conversational Knowledge Bot")
    st.caption("Powered by LangChain • Multi-Tool Agent • Persistent Memory")
    
    # Initialize agent on first run
    if not st.session_state.initialized:
        agent, error = initialize_agent()
        
        if error:
            show_error_message(error)
            st.stop()
        
        st.session_state.agent = agent
        st.session_state.initialized = True
        logger.info("Application initialized")
    
    agent = st.session_state.agent
    
    # Render sidebar and get actions
    sidebar_actions = render_sidebar(agent, agent.rag_tool)
    
    # Handle sidebar actions
    if sidebar_actions.get('clear_memory'):
        st.session_state.messages = []
        st.success("✅ Memory and chat history cleared!")
    
    if sidebar_actions.get('uploaded_files'):
        # Files were uploaded, show success messages
        successful_uploads = [f for f in sidebar_actions['uploaded_files'] if f['success']]
        if successful_uploads:
            st.success(f"✅ Uploaded {len(successful_uploads)} document(s)")
    
    # Main chat area
    st.markdown("---")
    
    # Display welcome message if no messages
    if len(st.session_state.messages) == 0:
        display_welcome_message()
    
    # Display chat messages
    for message in st.session_state.messages:
        render_message(
            role=message['role'],
            text=message['text'],
            sources=message.get('sources'),
            tool_calls=message.get('tool_calls')
        )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to UI
        st.session_state.messages.append({
            'role': 'user',
            'text': prompt
        })
        
        # Display user message
        render_message('user', prompt)
        
        # Get agent response
        with st.spinner("🤔 Thinking..."):
            try:
                # Get requested tool from sidebar
                forced_tool = sidebar_actions.get('forced_tool')
                
                response = agent.chat(prompt, forced_tool=forced_tool)
                
                if response['success']:
                    # Add assistant message to UI
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'text': response['response'],
                        'sources': response.get('sources', []),
                        'tool_calls': response.get('tool_calls', [])
                    })
                    
                    # Display assistant message
                    render_message(
                        'assistant',
                        response['response'],
                        sources=response.get('sources'),
                        tool_calls=response.get('tool_calls')
                    )
                else:
                    # Error occurred
                    error_msg = f"I encountered an error: {response.get('error', 'Unknown error')}"
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'text': error_msg
                    })
                    render_message('assistant', error_msg)
                    show_error_message(response.get('error', 'Unknown error'))
            
            except Exception as e:
                logger.error(f"Chat error: {e}")
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.session_state.messages.append({
                    'role': 'assistant',
                    'text': error_msg
                })
                render_message('assistant', error_msg)
                show_error_message(str(e))
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Conversational Knowledge Bot** is a multi-tool AI agent that can:
        
        - 🔍 Search the web using DuckDuckGo
        - 📚 Query Wikipedia for encyclopedic knowledge
        - 📄 Answer questions about uploaded documents (RAG)
        - 🧠 Remember conversation context
        
        **Tech Stack:**
        - LangChain for agent orchestration
        - Streamlit for UI
        - ChromaDB for vector storage
        - Gemini/DeepSeek for LLM
        
        **Source Code:** [GitHub](https://github.com/yourusername/conv-knowledge-bot)
        """)
        
        # Display current configuration
        st.markdown("**Current Configuration:**")
        st.code(f"""
LLM Provider: {os.getenv('LLM_PROVIDER', 'GEMINI')}
Vector Store: {os.getenv('VECTOR_STORE_DIR', 'data/vectorstore')}
Max Context: {os.getenv('MAX_CONTEXT_TOKENS', '2048')} tokens
        """, language="yaml")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"⚠️ Application Error: {str(e)}")
        st.info("Please check your configuration and try again.")
