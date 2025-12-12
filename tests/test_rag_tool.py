"""
Unit tests for RAG (Document Search) tool.
Tests document ingestion, chunking, and semantic search.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.rag_tool import RAGTool


class MockEmbedder:
    """Mock embedder for testing."""
    
    def embed_texts(self, texts):
        """Return mock embeddings."""
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, query):
        """Return mock query embedding."""
        return [0.1, 0.2, 0.3]


class TestRAGTool:
    """Test suite for RAG tool."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        return MockEmbedder()
    
    @pytest.fixture
    def rag_tool(self, temp_dir, mock_embedder):
        """Create RAG tool instance."""
        return RAGTool(
            embedder=mock_embedder,
            vector_store_dir=os.path.join(temp_dir, 'vectorstore'),
            chunk_size=100,
            chunk_overlap=20
        )
    
    def test_initialization(self, rag_tool):
        """Test RAG tool initialization."""
        assert rag_tool.name == "doc_search"
        assert rag_tool.chunk_size == 100
        assert rag_tool.chunk_overlap == 20
        assert rag_tool.collection is not None
    
    def test_ingest_text_file(self, rag_tool, temp_dir):
        """Test ingesting a text file."""
        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "This is a test document. " * 50  # Make it long enough to chunk
        test_file.write_text(test_content)
        
        # Ingest file
        result = rag_tool.ingest_file(str(test_file))
        
        # Assertions
        assert result['success'] is True
        assert result['chunks_added'] > 0
        assert result['filename'] == 'test.txt'
    
    def test_ingest_nonexistent_file(self, rag_tool):
        """Test ingesting a file that doesn't exist."""
        result = rag_tool.ingest_file("nonexistent_file.txt")
        
        assert result['success'] is False
        assert 'not found' in result['error'].lower()
        assert result['chunks_added'] == 0
    
    def test_search_empty_collection(self, rag_tool):
        """Test searching when no documents are ingested."""
        result = rag_tool.search("test query")
        
        assert result['meta']['tool'] == 'doc_search'
        assert result['meta']['chunks_found'] == 0
        assert 'No documents' in result['text']
    
    def test_search_with_documents(self, rag_tool, temp_dir):
        """Test searching after ingesting documents."""
        # Create and ingest test file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "Python is a programming language. " * 20
        test_file.write_text(test_content)
        
        rag_tool.ingest_file(str(test_file))
        
        # Search
        result = rag_tool.search("Python programming", top_k=2)
        
        # Assertions
        assert result['meta']['tool'] == 'doc_search'
        assert result['meta']['chunks_found'] > 0
        assert len(result['sources']) > 0
        assert 'test.txt' in result['sources'][0]
    
    def test_result_format(self, rag_tool, temp_dir):
        """Test that results follow standardized format."""
        # Ingest a document
        test_file = Path(temp_dir) / "doc.txt"
        test_file.write_text("Test content " * 30)
        rag_tool.ingest_file(str(test_file))
        
        # Search
        result = rag_tool.search("test")
        
        # Check standardized format
        assert 'text' in result
        assert 'sources' in result
        assert 'meta' in result
        assert isinstance(result['text'], str)
        assert isinstance(result['sources'], list)
        assert isinstance(result['meta'], dict)
    
    def test_langchain_tool_interface(self, rag_tool, temp_dir):
        """Test LangChain-compatible tool interface."""
        # Ingest a document
        test_file = Path(temp_dir) / "doc.txt"
        test_file.write_text("LangChain test content " * 20)
        rag_tool.ingest_file(str(test_file))
        
        # Test interface
        assert rag_tool.name == "doc_search"
        assert "document" in rag_tool.description.lower()
        
        # Test run method
        output = rag_tool._run("test query")
        assert isinstance(output, str)
    
    def test_get_stats(self, rag_tool, temp_dir):
        """Test getting RAG statistics."""
        stats = rag_tool.get_stats()
        
        assert 'total_chunks' in stats
        assert 'vector_store_dir' in stats
        assert stats['total_chunks'] == 0  # No documents yet
        
        # Ingest a document
        test_file = Path(temp_dir) / "doc.txt"
        test_file.write_text("Stats test " * 30)
        rag_tool.ingest_file(str(test_file))
        
        stats = rag_tool.get_stats()
        assert stats['total_chunks'] > 0
    
    @patch('src.tools.rag_tool.PdfReader')
    def test_pdf_extraction(self, mock_pdf_reader, rag_tool, temp_dir):
        """Test PDF text extraction."""
        # Create mock PDF
        test_file = Path(temp_dir) / "test.pdf"
        test_file.write_bytes(b"fake pdf content")
        
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content " * 30
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Ingest PDF
        result = rag_tool.ingest_file(str(test_file))
        
        assert result['success'] is True
        assert result['chunks_added'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
