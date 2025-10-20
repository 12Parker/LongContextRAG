"""
Configuration settings for Long Context RAG research project.
"""
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class RAGConfig:
    """Configuration class for RAG system settings."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "400"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "20"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "50000"))
    
    # Research Settings
    ENABLE_LONG_CONTEXT: bool = os.getenv("ENABLE_LONG_CONTEXT", "True").lower() == "true"
    CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "32000"))
    
    # File paths
    DATA_DIR: str = "data"
    VECTOR_STORE_DIR: str = "vector_store"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        return True

# Global config instance
config = RAGConfig()
