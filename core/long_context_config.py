"""
Long Context Configuration for RAG System

This module provides configuration options specifically for testing and using
the RAG system with context lengths above 32k tokens.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class ContextSize(Enum):
    """Predefined context size configurations."""
    STANDARD = "standard"      # 8k tokens
    MEDIUM = "medium"          # 16k tokens  
    LARGE = "large"           # 32k tokens
    XLARGE = "xlarge"         # 64k tokens
    ULTRA = "ultra"           # 128k+ tokens

@dataclass
class LongContextConfig:
    """Configuration for long context RAG testing."""
    
    # Context size configuration
    context_size: ContextSize = ContextSize.LARGE
    max_context_tokens: int = 32000
    target_context_tokens: int = 30000
    
    # Document processing
    num_documents: int = 10000
    chunk_size: int = 2000
    chunk_overlap: int = 200
    min_text_length: int = 50
    
    # Vector database
    db_path: str = "./vector_store_long"
    collection_name: str = "long_context"
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 50
    
    # Retrieval settings
    top_k_results: int = 10  # More results for longer context
    max_retrieved_chunks: int = 20
    
    # Model settings
    openai_model: str = "gpt-4-turbo-preview"  # Better for long context
    temperature: float = 0.1
    
    # Performance settings
    enable_hybrid_attention: bool = True
    enable_neural_retrieval: bool = True
    enable_token_counting: bool = True
    
    @classmethod
    def for_context_size(cls, size: ContextSize) -> 'LongContextConfig':
        """Create configuration for specific context size."""
        configs = {
            ContextSize.STANDARD: cls(
                context_size=ContextSize.STANDARD,
                max_context_tokens=8000,
                target_context_tokens=7000,
                num_documents=5000,
                chunk_size=1000,
                top_k_results=5,
                max_retrieved_chunks=10
            ),
            ContextSize.MEDIUM: cls(
                context_size=ContextSize.MEDIUM,
                max_context_tokens=16000,
                target_context_tokens=14000,
                num_documents=7500,
                chunk_size=1500,
                top_k_results=7,
                max_retrieved_chunks=15
            ),
            ContextSize.LARGE: cls(
                context_size=ContextSize.LARGE,
                max_context_tokens=32000,
                target_context_tokens=28000,
                num_documents=10000,
                chunk_size=2000,
                top_k_results=10,
                max_retrieved_chunks=20
            ),
            ContextSize.XLARGE: cls(
                context_size=ContextSize.XLARGE,
                max_context_tokens=64000,
                target_context_tokens=56000,
                num_documents=15000,
                chunk_size=3000,
                top_k_results=15,
                max_retrieved_chunks=30
            ),
            ContextSize.ULTRA: cls(
                context_size=ContextSize.ULTRA,
                max_context_tokens=128000,
                target_context_tokens=112000,
                num_documents=20000,
                chunk_size=4000,
                top_k_results=20,
                max_retrieved_chunks=50
            )
        }
        return configs[size]
    
    def to_vectordb_config(self) -> Dict[str, Any]:
        """Convert to VectorDBBuilder configuration."""
        return {
            'db_path': self.db_path,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'batch_size': self.batch_size
        }
    
    def get_test_queries(self) -> List[str]:
        """Get appropriate test queries for this context size."""
        base_queries = [
            "Analyze the narrative structure and character development",
            "Examine thematic elements and literary devices",
            "Compare dialogue patterns and character relationships",
            "Identify recurring motifs and their significance",
            "Evaluate pacing and resolution patterns"
        ]
        
        if self.context_size in [ContextSize.XLARGE, ContextSize.ULTRA]:
            # More complex queries for larger contexts
            base_queries.extend([
                "Provide a comprehensive analysis of narrative structure, character development, thematic elements, literary devices, dialogue patterns, plot progression, recurring motifs, and resolution patterns across the entire collection",
                "Examine the interconnections between different storylines, character arcs, and thematic threads throughout the corpus, identifying patterns of development, conflict resolution, and narrative coherence",
                "Analyze the linguistic patterns, stylistic choices, and narrative techniques employed across multiple texts, comparing their effectiveness in conveying meaning and engaging readers"
            ])
        
        return base_queries

class LongContextManager:
    """Manager for long context testing and configuration."""
    
    def __init__(self, config: Optional[LongContextConfig] = None):
        self.config = config or LongContextConfig()
        self.results = []
    
    def create_hybrid_rag(self) -> 'WorkingHybridRAG':
        """Create a WorkingHybridRAG instance with long context configuration."""
        from hybrid.working_hybrid_rag import WorkingHybridRAG
        
        vectordb_config = self.config.to_vectordb_config()
        
        return WorkingHybridRAG(
            use_hybrid_attention=self.config.enable_hybrid_attention,
            vectordb_config=vectordb_config,
            max_retrieved_docs=self.config.top_k_results
        )
    
    def test_context_length(self, context_size: ContextSize) -> Dict[str, Any]:
        """Test a specific context length configuration."""
        self.config = LongContextConfig.for_context_size(context_size)
        
        print(f"\n🧪 Testing {context_size.value.upper()} context ({self.config.max_context_tokens:,} tokens)")
        print("=" * 60)
        
        # Create hybrid RAG with this configuration
        hybrid_rag = self.create_hybrid_rag()
        
        # Create vector store
        print(f"📚 Creating vector store with {self.config.num_documents:,} documents...")
        hybrid_rag.create_vectorstore(
            use_vectordb=True,
            num_documents=self.config.num_documents
        )
        
        # Get database stats
        stats = hybrid_rag.get_vectordb_stats()
        
        # Test queries
        test_queries = self.config.get_test_queries()
        query_results = []
        
        for query in test_queries:
            print(f"\n🔍 Testing: {query[:60]}...")
            
            response = hybrid_rag.generate_response(query, task_type='qa')
            
            query_result = {
                'query': query,
                'response_length': len(response.get('response', '')),
                'retrieved_docs': response.get('retrieved_docs', 0),
                'context_length': response.get('context_length', 0),
                'method': response.get('method', 'unknown')
            }
            
            query_results.append(query_result)
        
        result = {
            'context_size': context_size.value,
            'max_context_tokens': self.config.max_context_tokens,
            'num_documents': self.config.num_documents,
            'total_chunks': stats.get('total_chunks', 0),
            'query_results': query_results,
            'config': self.config
        }
        
        self.results.append(result)
        return result
    
    def run_comprehensive_test(self, context_sizes: Optional[List[ContextSize]] = None):
        """Run comprehensive tests across multiple context sizes."""
        if context_sizes is None:
            context_sizes = [
                ContextSize.STANDARD,
                ContextSize.MEDIUM, 
                ContextSize.LARGE,
                ContextSize.XLARGE,
                ContextSize.ULTRA
            ]
        
        print("🚀 Comprehensive Long Context Testing")
        print("=" * 60)
        
        for context_size in context_sizes:
            try:
                result = self.test_context_length(context_size)
                self._print_result_summary(result)
            except Exception as e:
                print(f"❌ Error testing {context_size.value}: {e}")
        
        self._print_comprehensive_summary()
    
    def _print_result_summary(self, result: Dict[str, Any]):
        """Print summary for a single test result."""
        print(f"\n✅ {result['context_size'].upper()} Context Results:")
        print(f"   Max tokens: {result['max_context_tokens']:,}")
        print(f"   Documents: {result['num_documents']:,}")
        print(f"   Total chunks: {result['total_chunks']:,}")
        print(f"   Queries tested: {len(result['query_results'])}")
        
        avg_context = sum(qr['context_length'] for qr in result['query_results']) / len(result['query_results'])
        print(f"   Avg context length: {avg_context:.0f} chars")
    
    def _print_comprehensive_summary(self):
        """Print comprehensive summary of all tests."""
        print(f"\n{'='*60}")
        print("📈 COMPREHENSIVE LONG CONTEXT TESTING SUMMARY")
        print(f"{'='*60}")
        
        for result in self.results:
            context_size = result['context_size']
            max_tokens = result['max_context_tokens']
            total_chunks = result['total_chunks']
            avg_context = sum(qr['context_length'] for qr in result['query_results']) / len(result['query_results'])
            
            print(f"{context_size.upper():>8}: {max_tokens:>8,} tokens, "
                  f"{total_chunks:>6,} chunks, {avg_context:>6.0f} avg context")

# Environment variable configuration
def get_long_context_config_from_env() -> LongContextConfig:
    """Get long context configuration from environment variables."""
    context_size_str = os.getenv("LONG_CONTEXT_SIZE", "large").lower()
    
    context_size_map = {
        "standard": ContextSize.STANDARD,
        "medium": ContextSize.MEDIUM,
        "large": ContextSize.LARGE,
        "xlarge": ContextSize.XLARGE,
        "ultra": ContextSize.ULTRA
    }
    
    context_size = context_size_map.get(context_size_str, ContextSize.LARGE)
    
    config = LongContextConfig.for_context_size(context_size)
    
    # Override with environment variables if present
    if os.getenv("LONG_CONTEXT_MAX_TOKENS"):
        config.max_context_tokens = int(os.getenv("LONG_CONTEXT_MAX_TOKENS"))
    
    if os.getenv("LONG_CONTEXT_NUM_DOCS"):
        config.num_documents = int(os.getenv("LONG_CONTEXT_NUM_DOCS"))
    
    if os.getenv("LONG_CONTEXT_CHUNK_SIZE"):
        config.chunk_size = int(os.getenv("LONG_CONTEXT_CHUNK_SIZE"))
    
    return config
