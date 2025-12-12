"""
Unit tests for DuckDuckGo search tool.
Tests search functionality, static KB fallback, and error handling.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.ddg_tool import run_ddg, DuckDuckGoTool, _static_kb_search


class TestDuckDuckGoTool:
    """Test suite for DuckDuckGo search tool."""
    
    @patch('src.tools.ddg_tool.DDGS')
    def test_run_ddg_success(self, mock_ddgs):
        """Test successful DuckDuckGo search."""
        # Mock search results
        mock_results = [
            {
                'title': 'Python Programming',
                'body': 'Python is a high-level programming language.',
                'href': 'https://python.org'
            },
            {
                'title': 'Python Tutorial',
                'body': 'Learn Python programming.',
                'href': 'https://tutorial.com'
            }
        ]
        
        # Configure mock
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Run search
        result = run_ddg("Python programming", max_results=2)
        
        # Assertions
        assert result['meta']['tool'] == 'duckduckgo'
        assert result['meta']['num_results'] == 2
        assert len(result['sources']) == 2
        assert 'https://python.org' in result['sources']
        assert 'Python Programming' in result['text']
    
    @patch('src.tools.ddg_tool.DDGS')
    def test_run_ddg_no_results(self, mock_ddgs):
        """Test DuckDuckGo search with no results."""
        # Mock empty results
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.return_value = []
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Run search
        result = run_ddg("nonexistent query xyz123")
        
        # Assertions
        assert result['meta']['num_results'] == 0
        assert len(result['sources']) == 0
        assert 'No web search results' in result['text']
    
    def test_static_kb_search_no_file(self):
        """Test static KB search when file doesn't exist."""
        with patch.dict('os.environ', {'USE_STATIC_KB': 'true', 'STATIC_KB_PATH': 'nonexistent.json'}):
            result = _static_kb_search("test query")
            
            assert result['meta']['mode'] == 'static'
            assert result['meta']['num_results'] == 0
            assert 'not available' in result['text']
    
    def test_langchain_tool_interface(self):
        """Test LangChain-compatible tool interface."""
        with patch('src.tools.ddg_tool.DDGS') as mock_ddgs:
            # Mock results
            mock_results = [{'title': 'Test', 'body': 'Test body', 'href': 'http://test.com'}]
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.__enter__.return_value.text.return_value = mock_results
            mock_ddgs.return_value = mock_ddgs_instance
            
            # Create tool
            tool = DuckDuckGoTool(max_results=1)
            
            # Test interface
            assert tool.name == "web_search"
            assert "search" in tool.description.lower()
            
            # Test run method
            output = tool._run("test query")
            assert isinstance(output, str)
            assert len(output) > 0
    
    def test_result_format(self):
        """Test that results follow standardized format."""
        with patch('src.tools.ddg_tool.DDGS') as mock_ddgs:
            mock_results = [{'title': 'T', 'body': 'B', 'href': 'http://x.com'}]
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.__enter__.return_value.text.return_value = mock_results
            mock_ddgs.return_value = mock_ddgs_instance
            
            result = run_ddg("test")
            
            # Check standardized format
            assert 'text' in result
            assert 'sources' in result
            assert 'meta' in result
            assert isinstance(result['text'], str)
            assert isinstance(result['sources'], list)
            assert isinstance(result['meta'], dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
