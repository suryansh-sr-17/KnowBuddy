"""
Unit tests for Wikipedia search tool.
Tests search, disambiguation handling, and error cases.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.wiki_tool import run_wiki, WikipediaTool
import wikipedia


class TestWikipediaTool:
    """Test suite for Wikipedia search tool."""
    
    @patch('src.tools.wiki_tool.wikipedia.search')
    @patch('src.tools.wiki_tool.wikipedia.summary')
    def test_run_wiki_success(self, mock_summary, mock_search):
        """Test successful Wikipedia search."""
        # Mock search results
        mock_search.return_value = ['Python (programming language)', 'Python', 'Monty Python']
        mock_summary.return_value = "Python is a high-level, interpreted programming language."
        
        # Run search
        result = run_wiki("Python programming", sentences=2)
        
        # Assertions
        assert result['meta']['tool'] == 'wikipedia'
        assert result['meta']['found'] is True
        assert result['meta']['title'] == 'Python (programming language)'
        assert len(result['sources']) == 1
        assert 'wikipedia.org' in result['sources'][0]
        assert 'Python' in result['text']
    
    @patch('src.tools.wiki_tool.wikipedia.search')
    def test_run_wiki_no_results(self, mock_search):
        """Test Wikipedia search with no results."""
        mock_search.return_value = []
        
        result = run_wiki("nonexistent topic xyz123")
        
        assert result['meta']['found'] is False
        assert len(result['sources']) == 0
        assert 'No Wikipedia page found' in result['text']
    
    @patch('src.tools.wiki_tool.wikipedia.search')
    @patch('src.tools.wiki_tool.wikipedia.summary')
    def test_run_wiki_disambiguation(self, mock_summary, mock_search):
        """Test Wikipedia disambiguation handling."""
        mock_search.return_value = ['Mercury']
        
        # Simulate disambiguation error
        mock_summary.side_effect = wikipedia.DisambiguationError(
            'Mercury',
            ['Mercury (planet)', 'Mercury (element)', 'Mercury (mythology)']
        )
        
        result = run_wiki("Mercury")
        
        assert result['meta'].get('disambiguation') is True
        assert 'ambiguous' in result['text']
        assert 'Mercury (planet)' in result['text']
    
    @patch('src.tools.wiki_tool.wikipedia.search')
    @patch('src.tools.wiki_tool.wikipedia.summary')
    def test_run_wiki_page_error(self, mock_summary, mock_search):
        """Test Wikipedia page not found error."""
        mock_search.return_value = ['Test Page']
        mock_summary.side_effect = wikipedia.PageError('Test Page')
        
        result = run_wiki("Test Page")
        
        assert result['meta']['found'] is False
        assert 'not found' in result['text']
    
    def test_langchain_tool_interface(self):
        """Test LangChain-compatible tool interface."""
        with patch('src.tools.wiki_tool.wikipedia.search') as mock_search, \
             patch('src.tools.wiki_tool.wikipedia.summary') as mock_summary:
            
            mock_search.return_value = ['Test']
            mock_summary.return_value = "Test summary."
            
            tool = WikipediaTool(sentences=2)
            
            assert tool.name == "wikipedia"
            assert "encyclopedic" in tool.description.lower() or "wikipedia" in tool.description.lower()
            
            output = tool._run("test query")
            assert isinstance(output, str)
            assert len(output) > 0
    
    def test_result_format(self):
        """Test that results follow standardized format."""
        with patch('src.tools.wiki_tool.wikipedia.search') as mock_search, \
             patch('src.tools.wiki_tool.wikipedia.summary') as mock_summary:
            
            mock_search.return_value = ['Test']
            mock_summary.return_value = "Test."
            
            result = run_wiki("test")
            
            # Check standardized format
            assert 'text' in result
            assert 'sources' in result
            assert 'meta' in result
            assert isinstance(result['text'], str)
            assert isinstance(result['sources'], list)
            assert isinstance(result['meta'], dict)
    
    @patch('src.tools.wiki_tool.wikipedia.search')
    @patch('src.tools.wiki_tool.wikipedia.summary')
    def test_url_generation(self, mock_summary, mock_search):
        """Test Wikipedia URL generation."""
        mock_search.return_value = ['Albert Einstein']
        mock_summary.return_value = "Albert Einstein was a physicist."
        
        result = run_wiki("Einstein")
        
        assert len(result['sources']) == 1
        assert 'Albert_Einstein' in result['sources'][0]
        assert result['sources'][0].startswith('https://en.wikipedia.org/wiki/')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
