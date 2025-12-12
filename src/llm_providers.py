"""
LLM Provider Router - Abstracts Gemini and DeepSeek APIs.
Provides unified interface for text generation and embeddings.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from google import genai
from google.genai.errors import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion from prompt."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider using google.genai SDK."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp", 
                 embedding_model: str = "models/text-embedding-004"):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Gemini API key
            model_name: Model name for text generation (default: gemini-2.0-flash-exp)
            embedding_model: Model name for embeddings
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        logger.info(f"Initialized Gemini provider with model: {model_name}")
    
    def generate_with_functions(self, prompt: str, functions: List[Dict], 
                                temperature: float = 0.7, max_tokens: int = 1024) -> Dict:
        """
        Generate with function calling support.
        
        Args:
            prompt: User query
            functions: List of function declarations
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            Dictionary with response and function calls
        """
        try:
            # Remove 'models/' prefix if present
            model_to_use = self.model_name.replace('models/', '')
            
            # Create function declarations for Gemini
            tools = [{'function_declarations': functions}]
            
            response = self.client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config={
                    'temperature': temperature,
                    'max_output_tokens': max_tokens,
                    'tools': tools
                }
            )
            
            # Check if Gemini wants to call a function
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Check for function calls
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                # Gemini wants to call a function
                                function_call = part.function_call
                                return {
                                    'type': 'function_call',
                                    'function_name': function_call.name,
                                    'function_args': dict(function_call.args),
                                    'text': None
                                }
                
                # No function call, check for text response
                if hasattr(response, 'text') and response.text:
                    return {
                        'type': 'text',
                        'function_name': None,
                        'function_args': None,
                        'text': response.text
                    }
                
                # Try to get text from candidate
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        if text_parts:
                            return {
                                'type': 'text',
                                'function_name': None,
                                'function_args': None,
                                'text': ' '.join(text_parts)
                            }
            
            # Fallback
            return {
                'type': 'text',
                'function_name': None,
                'function_args': None,
                'text': "I apologize, but I couldn't generate a response."
            }
            
        except ClientError as e:
            status = getattr(e, "status_code", None)
            logger.error(f"Gemini function calling error: {e}")
            if status == 429:
                return {
                    'type': 'error',
                    'text': "I'm experiencing rate limits. Please wait a moment and try again."
                }
            raise
        except Exception as e:
            logger.error(f"Gemini function calling error: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 1024, **kwargs) -> str:
        """
        Generate text completion using Gemini.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text string
        """
        try:
            # Note: google.genai SDK expects model name without 'models/' prefix
            # It will add the prefix automatically
            model_to_use = self.model_name.replace('models/', '')
            
            response = self.client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config={
                    'temperature': temperature,
                    'max_output_tokens': max_tokens,
                    'top_p': kwargs.get('top_p', 0.9),
                    'top_k': kwargs.get('top_k', 40),
                }
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            else:
                logger.warning("Gemini response has no text attribute")
                return "I apologize, but I couldn't generate a response. Please try again."
            
        except ClientError as e:
            status = getattr(e, "status_code", None)
            logger.error(f"Gemini generation error: {e}")
            if status == 429:
                return "I'm experiencing rate limits. Please wait a moment and try again."
            raise
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            # Remove 'models/' prefix if present
            embedding_model_to_use = self.embedding_model.replace('models/', '')
            
            for text in texts:
                result = self.client.models.embed_content(
                    model=embedding_model_to_use,
                    content=text
                )
                # Extract embedding from response
                if hasattr(result, 'embedding'):
                    embeddings.append(result.embedding)
                elif isinstance(result, dict) and 'embedding' in result:
                    embeddings.append(result['embedding'])
                else:
                    # Fallback: use sentence-transformers
                    logger.warning("Gemini embedding format unexpected, using fallback")
                    return self._fallback_embeddings(texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            logger.warning("Falling back to sentence-transformers")
            return self._fallback_embeddings(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector
        """
        try:
            # Remove 'models/' prefix if present
            embedding_model_to_use = self.embedding_model.replace('models/', '')
            
            result = self.client.models.embed_content(
                model=embedding_model_to_use,
                content=query
            )
            
            if hasattr(result, 'embedding'):
                return result.embedding
            elif isinstance(result, dict) and 'embedding' in result:
                return result['embedding']
            else:
                return self._fallback_embeddings([query])[0]
            
        except Exception as e:
            logger.error(f"Gemini query embedding error: {e}")
            return self._fallback_embeddings([query])[0]
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback to local sentence-transformers if API fails."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Fallback embedding error: {e}")
            raise


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider."""
    
    def __init__(self, api_key: str, model_name: str = "deepseek-chat",
                 embedding_model: str = "deepseek-embedding"):
        """
        Initialize DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key
            model_name: Model name for text generation
            embedding_model: Model name for embeddings
        """
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.base_url = "https://api.deepseek.com/v1"
        
        logger.info(f"Initialized DeepSeek provider with model: {model_name}")
    
    def generate_with_functions(self, prompt: str, functions: List[Dict], 
                                temperature: float = 0.7, max_tokens: int = 1024) -> Dict:
        """
        Generate with function calling support (OpenAI-compatible).
        
        Args:
            prompt: User query
            functions: List of function declarations
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            Dictionary with response and function calls
        """
        try:
            import requests
            import json
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Convert functions to OpenAI format (DeepSeek is compatible)
            tools = [{"type": "function", "function": func} for func in functions]
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            message = result['choices'][0]['message']
            
            # Check if model wants to call a function
            if message.get('tool_calls'):
                tool_call = message['tool_calls'][0]
                function_call = tool_call['function']
                
                return {
                    'type': 'function_call',
                    'function_name': function_call['name'],
                    'function_args': json.loads(function_call['arguments']),
                    'text': None
                }
            
            # No function call, return text
            return {
                'type': 'text',
                'function_name': None,
                'function_args': None,
                'text': message.get('content', '')
            }
            
        except Exception as e:
            logger.error(f"DeepSeek function calling error: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 1024, **kwargs) -> str:
        """
        Generate text completion using DeepSeek.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            Generated text string
        """
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get('top_p', 0.9)
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"DeepSeek generation error: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.embedding_model,
                "input": texts
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return [item['embedding'] for item in result['data']]
            
        except Exception as e:
            logger.error(f"DeepSeek embedding error: {e}")
            # Fallback to sentence-transformers if API fails
            logger.warning("Falling back to local sentence-transformers")
            return self._fallback_embeddings(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector
        """
        return self.embed_texts([query])[0]
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback to local sentence-transformers if API fails."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Fallback embedding error: {e}")
            raise


class GroqProvider(BaseLLMProvider):
    """Groq API provider with function calling support."""
    
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq provider.
        
        Args:
            api_key: Groq API key
            model_name: Model name for text generation
        """
        if not api_key:
            raise ValueError("Groq API key is required")
        
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.model_name = model_name
            
            logger.info(f"Initialized Groq provider with model: {model_name}")
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
    
    def generate_with_functions(self, prompt: str, functions: List[Dict], 
                                temperature: float = 0.7, max_tokens: int = 1024) -> Dict:
        """
        Generate with function calling support (OpenAI-compatible).
        
        Args:
            prompt: User query
            functions: List of function declarations
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            Dictionary with response and function calls
        """
        try:
            import json
            
            # Convert functions to OpenAI format (Groq is compatible)
            tools = [{"type": "function", "function": func} for func in functions]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice="auto"  # Let model decide
            )
            
            message = response.choices[0].message
            
            # Check if model wants to call a function
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_call = tool_call.function
                
                return {
                    'type': 'function_call',
                    'function_name': function_call.name,
                    'function_args': json.loads(function_call.arguments),
                    'text': None
                }
            
            # No function call, return text
            return {
                'type': 'text',
                'function_name': None,
                'function_args': None,
                'text': message.content or ''
            }
            
        except Exception as e:
            logger.error(f"Groq function calling error: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 1024, **kwargs) -> str:
        """
        Generate text completion using Groq.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            Generated text string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings (fallback to sentence-transformers).
        Groq doesn't provide embedding API, so we use local embeddings.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        logger.info("Groq doesn't provide embeddings, using sentence-transformers")
        return self._fallback_embeddings(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector
        """
        return self.embed_texts([query])[0]
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback to local sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Fallback embedding error: {e}")
            raise


def get_llm_and_embedder(provider_name: str = None, config: Dict[str, Any] = None) -> tuple:
    """
    Factory function to get LLM provider and embedder.
    
    Args:
        provider_name: Provider name ('GEMINI' or 'DEEPSEEK'). 
                      If None, reads from LLM_PROVIDER env var.
        config: Optional configuration dictionary
    
    Returns:
        Tuple of (llm_provider, embedding_function)
    
    Raises:
        ValueError: If provider is invalid or API key is missing
    
    Example:
        >>> llm, embedder = get_llm_and_embedder('GEMINI')
        >>> response = llm.generate("Hello, how are you?")
    """
    if config is None:
        config = {}
    
    # Get provider from env if not specified
    if provider_name is None:
        provider_name = os.getenv('LLM_PROVIDER', 'GEMINI').upper()
    else:
        provider_name = provider_name.upper()
    
    logger.info(f"Initializing LLM provider: {provider_name}")
    
    if provider_name == 'GEMINI':
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        
        provider = GeminiProvider(
            api_key=api_key,
            model_name=config.get('model_name', 'gemini-2.0-flash-exp'),
            embedding_model=config.get('embedding_model', 'models/text-embedding-004')
        )
        
    elif provider_name == 'DEEPSEEK':
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        
        provider = DeepSeekProvider(
            api_key=api_key,
            model_name=config.get('model_name', 'deepseek-chat'),
            embedding_model=config.get('embedding_model', 'deepseek-embedding')
        )
    
    elif provider_name == 'GROQ':
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        
        provider = GroqProvider(
            api_key=api_key,
            model_name=config.get('model_name', 'llama-3.1-8b-instant')
        )
        
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: GEMINI, DEEPSEEK, GROQ"
        )
    
    return provider, provider


if __name__ == "__main__":
    # Test provider initialization
    print("Testing LLM providers...")
    
    try:
        # Test with environment variable
        llm, embedder = get_llm_and_embedder()
        print(f"✓ Successfully initialized provider from environment")
        
        # Test generation
        response = llm.generate("Say 'Hello World' and nothing else.", temperature=0.1, max_tokens=10)
        print(f"✓ Generation test: {response[:50]}...")
        
        # Test embedding
        embedding = embedder.embed_query("test query")
        print(f"✓ Embedding test: vector length = {len(embedding)}")
        
    except Exception as e:
        print(f"✗ Provider test failed: {e}")
        print("Note: Make sure you have set up your .env file with API keys")
