"""
RAG (Retrieval-Augmented Generation) Tool
Handles document ingestion, chunking, embedding, and semantic search.
"""

import os
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from src.utils import chunk_text, ensure_data_dir, safe_text

logger = logging.getLogger(__name__)


class RAGTool:
    """
    Document ingestion and semantic search tool.
    
    Features:
    - Ingest PDF, TXT, MD files
    - Chunk text with overlap
    - Generate embeddings
    - Semantic search with ChromaDB
    """
    
    name = "doc_search"
    description = """Useful for searching uploaded documents and retrieving relevant information.
    Input should be a question or query about document content.
    Returns relevant passages with file and chunk references."""
    
    def __init__(self, embedder, vector_store_dir: str = None, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize RAG tool.
        
        Args:
            embedder: Embedding function provider
            vector_store_dir: Directory for vector store persistence
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlapping tokens between chunks
        """
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Set up vector store directory
        if vector_store_dir is None:
            vector_store_dir = os.getenv('VECTOR_STORE_DIR', 'data/vectorstore')
        
        self.vector_store_dir = Path(vector_store_dir)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.vector_store_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "User uploaded documents"}
        )
        
        logger.info(f"Initialized RAG tool with vector store at {self.vector_store_dir}")
    
    def ingest_file(self, file_path: str, doc_id: Optional[str] = None) -> Dict:
        """
        Ingest a document file into the vector store.
        
        Args:
            file_path: Path to the file to ingest
            doc_id: Optional document ID (auto-generated if None)
        
        Returns:
            Dictionary with ingestion results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "chunks_added": 0
            }
        
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Ingesting file: {file_path.name}")
            
            # Extract text based on file type
            text = self._extract_text(file_path)
            
            if not text:
                return {
                    "success": False,
                    "error": "No text extracted from file",
                    "chunks_added": 0
                }
            
            # Chunk the text
            chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks created from text",
                    "chunks_added": 0
                }
            
            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for chunk in chunks:
                chunk_id = f"{doc_id}_chunk_{chunk['chunk_index']}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk['text'])
                chunk_metadatas.append({
                    "source": file_path.name,
                    "doc_id": doc_id,
                    "chunk_index": chunk['chunk_index'],
                    "start_char": chunk['start_char'],
                    "end_char": chunk['end_char']
                })
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedder.embed_texts(chunk_texts)
            
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path.name}")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "filename": file_path.name,
                "chunks_added": len(chunks),
                "total_chars": len(text)
            }
        
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_added": 0
            }
    
    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from file based on extension.
        
        Args:
            file_path: Path to file
        
        Returns:
            Extracted text string
        """
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._extract_pdf(file_path)
            elif extension in ['.txt', '.md']:
                return self._extract_text_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {extension}")
                return ""
        
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            return ""
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(str(file_path))
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
        
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from TXT or MD file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Text file reading failed: {e}")
            return ""
    
    def search(self, query: str, top_k: int = 4) -> Dict:
        """
        Search documents for relevant passages.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            Dictionary with standardized format
        """
        try:
            logger.info(f"Document search: {query}")
            
            # Check if collection is empty
            if self.collection.count() == 0:
                return {
                    "text": "No documents have been uploaded yet. Please upload documents first.",
                    "sources": [],
                    "meta": {"tool": "doc_search", "chunks_found": 0}
                }
            
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count())
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    "text": f"No relevant passages found for: {query}",
                    "sources": [],
                    "meta": {"tool": "doc_search", "chunks_found": 0}
                }
            
            # Format results
            formatted_passages = []
            sources = []
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0] if 'distances' in results else [0] * len(documents)
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
                source_file = metadata.get('source', 'unknown')
                chunk_idx = metadata.get('chunk_index', 0)
                
                # Clean passage
                passage = safe_text(doc, max_length=500)
                
                # Format passage
                formatted_passages.append(f"[Passage {i}] From {source_file} (chunk {chunk_idx}):")
                formatted_passages.append(f"{passage}")
                formatted_passages.append("")  # Empty line
                
                # Add source reference
                source_ref = f"{source_file}#chunk-{chunk_idx}"
                sources.append(source_ref)
            
            result_text = "\n".join(formatted_passages).strip()
            
            logger.info(f"Found {len(documents)} relevant passages")
            
            return {
                "text": result_text,
                "sources": sources,
                "meta": {
                    "tool": "doc_search",
                    "chunks_found": len(documents),
                    "query": query
                }
            }
        
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {
                "text": f"Document search error: {str(e)}",
                "sources": [],
                "meta": {"tool": "doc_search", "error": str(e)}
            }
    
    def _run(self, query: str) -> str:
        """
        Run the tool (LangChain interface).
        
        Args:
            query: Search query
        
        Returns:
            Formatted search results as string
        """
        result = self.search(query)
        
        # Format for LangChain
        output = result["text"]
        if result["sources"]:
            sources_str = ", ".join(result["sources"])
            output += f"\n\nSOURCES: {sources_str}"
        
        return output
    
    async def _arun(self, query: str) -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(query)
    
    def get_stats(self) -> Dict:
        """Get statistics about the document collection."""
        return {
            "total_chunks": self.collection.count(),
            "vector_store_dir": str(self.vector_store_dir)
        }


# Convenience functions for standalone use
def ingest_file(file_path: str, embedder, doc_id: Optional[str] = None) -> Dict:
    """
    Standalone function to ingest a file.
    
    Args:
        file_path: Path to file
        embedder: Embedding provider
        doc_id: Optional document ID
    
    Returns:
        Ingestion result dictionary
    """
    tool = RAGTool(embedder)
    return tool.ingest_file(file_path, doc_id)


def run_doc_search(query: str, embedder, top_k: int = 4) -> Dict:
    """
    Standalone function to search documents.
    
    Args:
        query: Search query
        embedder: Embedding provider
        top_k: Number of results
    
    Returns:
        Search result dictionary
    """
    tool = RAGTool(embedder)
    return tool.search(query, top_k)


if __name__ == "__main__":
    print("Testing RAG tool...")
    print("Note: Full testing requires embedder initialization")
    print("Run integration tests with actual embedder for complete validation")
