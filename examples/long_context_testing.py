#!/usr/bin/env python3
"""
Long Context Testing for WorkingHybridRAG with VectorDBBuilder

This script demonstrates how to test the hybrid RAG system with context lengths
above 32k tokens, including configuration, token counting, and performance analysis.
"""

import sys
import os
from pathlib import Path
import tiktoken
import time
import json
from typing import List, Dict, Any, Optional
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from hybrid.working_hybrid_rag import WorkingHybridRAG
from core.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LongContextTester:
    """Test the hybrid RAG system with various long context configurations."""
    
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.results = []
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def create_long_context_config(self, context_size: int) -> Dict[str, Any]:
        """Create configuration for long context testing."""
        return {
            'db_path': f'./vector_store_long_{context_size}',
            'collection_name': f'long_context_{context_size}',
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': min(2000, context_size // 20),  # Adaptive chunk size
            'chunk_overlap': 200,
            'batch_size': 50
        }
    
    def test_context_lengths(self, context_sizes: List[int] = None):
        """Test the system with different context lengths."""
        if context_sizes is None:
            context_sizes = [8000, 16000, 32000, 64000, 128000]
        
        print("🚀 Long Context Testing for WorkingHybridRAG")
        print("=" * 60)
        
        for context_size in context_sizes:
            print(f"\n{'='*60}")
            print(f"Testing Context Length: {context_size:,} tokens")
            print(f"{'='*60}")
            
            try:
                result = self._test_single_context_length(context_size)
                self.results.append(result)
                self._print_context_results(result)
                
            except Exception as e:
                logger.error(f"Error testing context length {context_size}: {e}")
                self.results.append({
                    'context_size': context_size,
                    'error': str(e),
                    'success': False
                })
        
        self._save_results()
        self._print_summary()
    
    def _test_single_context_length(self, context_size: int) -> Dict[str, Any]:
        """Test a single context length configuration."""
        start_time = time.time()
        
        # Create configuration for this context size
        vectordb_config = self.create_long_context_config(context_size)
        
        # Initialize hybrid RAG with long context settings
        hybrid_rag = WorkingHybridRAG(
            use_hybrid_attention=True,
            vectordb_config=vectordb_config
        )
        
        # Create vector store with more documents for longer contexts
        num_documents = min(10000, context_size // 10)  # Adaptive document count
        
        print(f"📚 Creating vector store with {num_documents:,} documents...")
        hybrid_rag.create_vectorstore(
            use_vectordb=True,
            num_documents=num_documents
        )
        
        # Get database stats
        stats = hybrid_rag.get_vectordb_stats()
        
        # Test queries designed for long context
        test_queries = self._get_long_context_queries()
        
        query_results = []
        total_context_tokens = 0
        
        for query in test_queries:
            print(f"\n🔍 Testing query: {query[:50]}...")
            
            # Generate response
            response = hybrid_rag.generate_response(query, task_type='qa')
            
            # Count tokens in context
            context_tokens = self.count_tokens(response.get('response', ''))
            total_context_tokens += context_tokens
            
            query_result = {
                'query': query,
                'response_length': len(response.get('response', '')),
                'context_tokens': context_tokens,
                'retrieved_docs': response.get('retrieved_docs', 0),
                'context_length': response.get('context_length', 0),
                'method': response.get('method', 'unknown')
            }
            
            query_results.append(query_result)
        
        elapsed_time = time.time() - start_time
        
        return {
            'context_size': context_size,
            'success': True,
            'elapsed_time': elapsed_time,
            'num_documents': num_documents,
            'db_stats': stats,
            'query_results': query_results,
            'avg_context_tokens': total_context_tokens / len(test_queries),
            'max_context_tokens': max([qr['context_tokens'] for qr in query_results]),
            'total_chunks': stats.get('total_chunks', 0)
        }
    
    def _get_long_context_queries(self) -> List[str]:
        """Get test queries designed for long context analysis."""
        return [
            "Analyze the narrative structure and character development patterns across multiple chapters",
            "Compare and contrast the thematic elements and literary devices used throughout the story",
            "Examine the dialogue patterns, character relationships, and plot progression in detail",
            "Identify recurring motifs, symbols, and their significance in the overall narrative",
            "Evaluate the pacing, tension, and resolution patterns across different story arcs"
        ]
    
    def _print_context_results(self, result: Dict[str, Any]):
        """Print results for a single context length test."""
        if not result['success']:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            return
        
        print(f"✅ Success!")
        print(f"   Documents processed: {result['num_documents']:,}")
        print(f"   Total chunks: {result['total_chunks']:,}")
        print(f"   Elapsed time: {result['elapsed_time']:.2f}s")
        print(f"   Avg context tokens: {result['avg_context_tokens']:.0f}")
        print(f"   Max context tokens: {result['max_context_tokens']:,}")
        
        print(f"\n📊 Query Results:")
        for i, qr in enumerate(result['query_results'], 1):
            print(f"   Query {i}: {qr['context_tokens']:,} tokens, {qr['retrieved_docs']} docs")
    
    def _save_results(self):
        """Save test results to file."""
        results_file = "results/long_context_test_results.json"
        os.makedirs("results", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _print_summary(self):
        """Print summary of all tests."""
        print(f"\n{'='*60}")
        print("📈 LONG CONTEXT TESTING SUMMARY")
        print(f"{'='*60}")
        
        successful_tests = [r for r in self.results if r.get('success', False)]
        failed_tests = [r for r in self.results if not r.get('success', False)]
        
        print(f"✅ Successful tests: {len(successful_tests)}")
        print(f"❌ Failed tests: {len(failed_tests)}")
        
        if successful_tests:
            print(f"\n📊 Performance by Context Size:")
            for result in successful_tests:
                context_size = result['context_size']
                avg_tokens = result['avg_context_tokens']
                max_tokens = result['max_context_tokens']
                elapsed_time = result['elapsed_time']
                
                print(f"   {context_size:,} tokens: "
                      f"avg={avg_tokens:.0f}, max={max_tokens:,}, time={elapsed_time:.1f}s")
        
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for result in failed_tests:
                print(f"   {result['context_size']:,} tokens: {result.get('error', 'Unknown error')}")

def test_with_custom_config():
    """Test with custom configuration for very long contexts."""
    print("\n🔧 Custom Long Context Configuration Test")
    print("=" * 60)
    
    # Custom configuration for 128k+ context
    custom_config = {
        'db_path': './vector_store_ultra_long',
        'collection_name': 'ultra_long_context',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': 4000,  # Larger chunks for longer context
        'chunk_overlap': 400,
        'batch_size': 25
    }
    
    # Initialize with custom config
    hybrid_rag = WorkingHybridRAG(
        use_hybrid_attention=True,
        vectordb_config=custom_config
    )
    
    # Create vector store with maximum documents
    print("📚 Creating ultra-long context vector store...")
    hybrid_rag.create_vectorstore(
        use_vectordb=True,
        num_documents=20000  # Maximum documents for ultra-long context
    )
    
    # Test with very long queries
    ultra_long_queries = [
        "Provide a comprehensive analysis of the narrative structure, character development, thematic elements, literary devices, dialogue patterns, plot progression, recurring motifs, and resolution patterns across the entire collection of documents",
        "Examine the interconnections between different storylines, character arcs, and thematic threads throughout the corpus, identifying patterns of development, conflict resolution, and narrative coherence",
        "Analyze the linguistic patterns, stylistic choices, and narrative techniques employed across multiple texts, comparing their effectiveness in conveying meaning and engaging readers"
    ]
    
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    for i, query in enumerate(ultra_long_queries, 1):
        print(f"\n🔍 Ultra-Long Query {i}:")
        print(f"   Query tokens: {len(tokenizer.encode(query))}")
        
        start_time = time.time()
        response = hybrid_rag.generate_response(query, task_type='qa')
        elapsed_time = time.time() - start_time
        
        response_tokens = len(tokenizer.encode(response.get('response', '')))
        
        print(f"   Response tokens: {response_tokens:,}")
        print(f"   Retrieved docs: {response.get('retrieved_docs', 0)}")
        print(f"   Context length: {response.get('context_length', 0):,} chars")
        print(f"   Elapsed time: {elapsed_time:.2f}s")
        print(f"   Method: {response.get('method', 'unknown')}")

def main():
    """Main function to run long context tests."""
    print("🧪 Long Context Testing for WorkingHybridRAG")
    print("=" * 60)
    
    # Check if we have the required environment
    if not config.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    # Initialize tester
    tester = LongContextTester()
    
    # Test different context lengths
    print("🚀 Starting long context length tests...")
    tester.test_context_lengths([8000, 16000, 32000, 64000, 128000])
    
    # Test with custom ultra-long configuration
    test_with_custom_config()
    
    print("\n✅ Long context testing completed!")
    print("\n💡 Tips for longer contexts:")
    print("   - Use larger chunk sizes (2000-4000) for better context retention")
    print("   - Increase document count for richer context")
    print("   - Consider using gpt-4-turbo for better long context handling")
    print("   - Monitor token usage to stay within API limits")

if __name__ == "__main__":
    main()
