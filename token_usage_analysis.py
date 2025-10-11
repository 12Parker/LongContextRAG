"""
Token Usage Analysis for Hybrid Attention RAG System

This script analyzes token usage across different components and scenarios
to help you understand the computational costs of your system.
"""

import tiktoken
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class TokenUsageConfig:
    """Configuration for token usage analysis."""
    # Model configurations
    embedding_model: str = "text-embedding-3-large"  # 3072 dimensions
    chat_model: str = "gpt-4-turbo-preview"
    
    # BookCorpus settings
    max_books: int = 8
    avg_book_length: int = 50000  # characters
    chunk_size: int = 2000
    chunk_overlap: int = 200
    max_chunks_per_book: int = 20
    
    # Testing parameters
    test_queries_per_book: int = 3
    avg_query_length: int = 50  # characters
    
    # Hybrid attention settings
    window_size: int = 512
    num_landmark_tokens: int = 32
    max_retrieved_segments: int = 8

class TokenUsageAnalyzer:
    """
    Analyzer for token usage across the hybrid attention RAG system.
    """
    
    def __init__(self, config: TokenUsageConfig):
        self.config = config
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def analyze_complete_system(self) -> Dict[str, Any]:
        """Analyze token usage for the complete system."""
        analysis = {
            'bookcorpus_setup': self._analyze_bookcorpus_setup(),
            'hybrid_attention_processing': self._analyze_hybrid_attention(),
            'neural_retrieval': self._analyze_neural_retrieval(),
            'llm_generation': self._analyze_llm_generation(),
            'total_per_query': self._calculate_total_per_query(),
            'total_for_full_test': self._calculate_total_for_full_test(),
            'cost_estimates': self._calculate_cost_estimates()
        }
        
        return analysis
    
    def _analyze_bookcorpus_setup(self) -> Dict[str, Any]:
        """Analyze token usage for BookCorpus setup."""
        # Calculate total content
        total_books = self.config.max_books
        avg_book_tokens = self._chars_to_tokens(self.config.avg_book_length)
        total_book_tokens = total_books * avg_book_tokens
        
        # Calculate chunks
        chunks_per_book = self.config.max_chunks_per_book
        total_chunks = total_books * chunks_per_book
        chunk_tokens = self._chars_to_tokens(self.config.chunk_size)
        total_chunk_tokens = total_chunks * chunk_tokens
        
        # Embedding costs (text-embedding-3-large)
        embedding_tokens = total_chunk_tokens  # Same tokens for embedding
        
        return {
            'total_books': total_books,
            'total_book_tokens': total_book_tokens,
            'total_chunks': total_chunks,
            'total_chunk_tokens': total_chunk_tokens,
            'embedding_tokens': embedding_tokens,
            'avg_tokens_per_book': avg_book_tokens,
            'avg_tokens_per_chunk': chunk_tokens
        }
    
    def _analyze_hybrid_attention(self) -> Dict[str, Any]:
        """Analyze token usage for hybrid attention processing."""
        # Sliding window attention
        window_tokens = self.config.window_size
        window_overlap = self._chars_to_tokens(200)  # Assuming 200 char overlap
        
        # Sparse global attention
        landmark_tokens = self.config.num_landmark_tokens
        
        # Retrieval-augmented segments
        max_segments = self.config.max_retrieved_segments
        segment_tokens = self._chars_to_tokens(256)  # Average segment length
        total_segment_tokens = max_segments * segment_tokens
        
        # Per query processing
        query_tokens = self._chars_to_tokens(self.config.avg_query_length)
        
        # Total processing tokens per query
        total_processing_tokens = (
            query_tokens +           # Input query
            window_tokens +          # Sliding window processing
            landmark_tokens +        # Landmark token processing
            total_segment_tokens     # Retrieved segments
        )
        
        return {
            'window_tokens': window_tokens,
            'landmark_tokens': landmark_tokens,
            'segment_tokens': total_segment_tokens,
            'query_tokens': query_tokens,
            'total_processing_tokens_per_query': total_processing_tokens,
            'max_segments': max_segments
        }
    
    def _analyze_neural_retrieval(self) -> Dict[str, Any]:
        """Analyze token usage for neural retrieval."""
        # Query processing
        query_tokens = self._chars_to_tokens(self.config.avg_query_length)
        
        # Document processing (for retrieval)
        chunk_tokens = self._chars_to_tokens(self.config.chunk_size)
        num_candidates = 100  # Typical number of candidate documents
        total_doc_tokens = num_candidates * chunk_tokens
        
        # Dynamic query generation
        dynamic_query_tokens = query_tokens * 2  # Expanded query
        
        # Per query retrieval tokens
        retrieval_tokens_per_query = (
            query_tokens +           # Original query
            dynamic_query_tokens +   # Dynamic query generation
            total_doc_tokens         # Document processing
        )
        
        return {
            'query_tokens': query_tokens,
            'dynamic_query_tokens': dynamic_query_tokens,
            'document_tokens': total_doc_tokens,
            'retrieval_tokens_per_query': retrieval_tokens_per_query,
            'num_candidates': num_candidates
        }
    
    def _analyze_llm_generation(self) -> Dict[str, Any]:
        """Analyze token usage for LLM generation."""
        # Input context
        query_tokens = self._chars_to_tokens(self.config.avg_query_length)
        context_tokens = self._chars_to_tokens(4000)  # Average context length
        prompt_tokens = self._chars_to_tokens(500)    # System prompt
        
        # Total input tokens
        input_tokens = query_tokens + context_tokens + prompt_tokens
        
        # Output tokens (estimated)
        output_tokens = self._chars_to_tokens(800)    # Average response length
        
        # Total LLM tokens per query
        total_llm_tokens = input_tokens + output_tokens
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_llm_tokens_per_query': total_llm_tokens,
            'context_tokens': context_tokens,
            'prompt_tokens': prompt_tokens
        }
    
    def _calculate_total_per_query(self) -> Dict[str, Any]:
        """Calculate total token usage per query."""
        hybrid_attention = self._analyze_hybrid_attention()
        neural_retrieval = self._analyze_neural_retrieval()
        llm_generation = self._analyze_llm_generation()
        
        # Note: Some tokens are shared between components
        # We'll calculate the unique tokens for each component
        
        total_tokens = (
            hybrid_attention['total_processing_tokens_per_query'] +
            neural_retrieval['retrieval_tokens_per_query'] +
            llm_generation['total_llm_tokens_per_query']
        )
        
        # Subtract overlapping tokens (query is processed multiple times)
        query_tokens = self._chars_to_tokens(self.config.avg_query_length)
        overlapping_tokens = query_tokens * 2  # Query processed in multiple components
        
        net_total_tokens = total_tokens - overlapping_tokens
        
        return {
            'hybrid_attention_tokens': hybrid_attention['total_processing_tokens_per_query'],
            'neural_retrieval_tokens': neural_retrieval['retrieval_tokens_per_query'],
            'llm_generation_tokens': llm_generation['total_llm_tokens_per_query'],
            'overlapping_tokens': overlapping_tokens,
            'net_total_tokens_per_query': net_total_tokens,
            'gross_total_tokens_per_query': total_tokens
        }
    
    def _calculate_total_for_full_test(self) -> Dict[str, Any]:
        """Calculate total token usage for full BookCorpus test."""
        per_query = self._calculate_total_per_query()
        total_queries = self.config.max_books * self.config.test_queries_per_book
        
        # Setup costs (one-time)
        setup = self._analyze_bookcorpus_setup()
        
        # Per-query costs
        per_query_tokens = per_query['net_total_tokens_per_query']
        total_query_tokens = total_queries * per_query_tokens
        
        # Total system tokens
        total_system_tokens = setup['embedding_tokens'] + total_query_tokens
        
        return {
            'total_queries': total_queries,
            'setup_tokens': setup['embedding_tokens'],
            'query_processing_tokens': total_query_tokens,
            'total_system_tokens': total_system_tokens,
            'tokens_per_query': per_query_tokens
        }
    
    def _calculate_cost_estimates(self) -> Dict[str, Any]:
        """Calculate estimated costs for different models."""
        full_test = self._calculate_total_for_full_test()
        per_query = self._calculate_total_per_query()
        
        # Pricing (as of 2024, approximate)
        pricing = {
            'gpt-4-turbo-preview': {
                'input': 0.01 / 1000,  # $0.01 per 1K tokens
                'output': 0.03 / 1000   # $0.03 per 1K tokens
            },
            'text-embedding-3-large': {
                'input': 0.00013 / 1000  # $0.00013 per 1K tokens
            }
        }
        
        # Calculate costs
        llm_tokens = per_query['llm_generation_tokens']
        embedding_tokens = full_test['setup_tokens']
        
        # LLM costs (per query)
        llm_input_tokens = llm_tokens * 0.7  # Rough estimate
        llm_output_tokens = llm_tokens * 0.3
        llm_cost_per_query = (
            llm_input_tokens * pricing['gpt-4-turbo-preview']['input'] +
            llm_output_tokens * pricing['gpt-4-turbo-preview']['output']
        )
        
        # Embedding costs (one-time setup)
        embedding_cost = embedding_tokens * pricing['text-embedding-3-large']['input']
        
        # Total costs
        total_queries = full_test['total_queries']
        total_llm_cost = llm_cost_per_query * total_queries
        total_cost = embedding_cost + total_llm_cost
        
        return {
            'llm_cost_per_query': llm_cost_per_query,
            'embedding_cost_total': embedding_cost,
            'total_llm_cost': total_llm_cost,
            'total_cost': total_cost,
            'cost_per_query': total_cost / total_queries if total_queries > 0 else 0
        }
    
    def _chars_to_tokens(self, chars: int) -> int:
        """Convert character count to token count (rough estimate)."""
        # Rough estimate: 1 token ≈ 4 characters for English text
        return int(chars / 4)
    
    def generate_usage_report(self) -> str:
        """Generate a comprehensive usage report."""
        analysis = self.analyze_complete_system()
        
        report = f"""
# 🔢 Token Usage Analysis for Hybrid Attention RAG

## 📊 System Overview
- **Books**: {analysis['bookcorpus_setup']['total_books']}
- **Total Book Tokens**: {analysis['bookcorpus_setup']['total_book_tokens']:,}
- **Document Chunks**: {analysis['bookcorpus_setup']['total_chunks']}
- **Test Queries**: {analysis['total_for_full_test']['total_queries']}

## 🧠 Hybrid Attention Processing
- **Window Size**: {analysis['hybrid_attention_processing']['window_tokens']} tokens
- **Landmark Tokens**: {analysis['hybrid_attention_processing']['landmark_tokens']} tokens
- **Retrieved Segments**: {analysis['hybrid_attention_processing']['segment_tokens']} tokens
- **Total per Query**: {analysis['hybrid_attention_processing']['total_processing_tokens_per_query']:,} tokens

## 🔍 Neural Retrieval
- **Query Processing**: {analysis['neural_retrieval']['query_tokens']} tokens
- **Dynamic Query**: {analysis['neural_retrieval']['dynamic_query_tokens']} tokens
- **Document Processing**: {analysis['neural_retrieval']['document_tokens']:,} tokens
- **Total per Query**: {analysis['neural_retrieval']['retrieval_tokens_per_query']:,} tokens

## 💬 LLM Generation
- **Input Tokens**: {analysis['llm_generation']['input_tokens']:,} tokens
- **Output Tokens**: {analysis['llm_generation']['output_tokens']:,} tokens
- **Total per Query**: {analysis['llm_generation']['total_llm_tokens_per_query']:,} tokens

## 📈 Per Query Summary
- **Hybrid Attention**: {analysis['total_per_query']['hybrid_attention_tokens']:,} tokens
- **Neural Retrieval**: {analysis['total_per_query']['neural_retrieval_tokens']:,} tokens
- **LLM Generation**: {analysis['total_per_query']['llm_generation_tokens']:,} tokens
- **Net Total per Query**: {analysis['total_per_query']['net_total_tokens_per_query']:,} tokens

## 💰 Cost Estimates
- **Setup Cost**: ${analysis['cost_estimates']['embedding_cost_total']:.4f}
- **Cost per Query**: ${analysis['cost_estimates']['cost_per_query']:.4f}
- **Total Test Cost**: ${analysis['cost_estimates']['total_cost']:.4f}

## 🎯 Scaling Estimates
"""
        
        # Add scaling information
        for scale in [1, 5, 10, 50, 100]:
            scaled_queries = analysis['total_for_full_test']['total_queries'] * scale
            scaled_cost = analysis['cost_estimates']['cost_per_query'] * scaled_queries
            report += f"- **{scale}x Scale** ({scaled_queries} queries): ${scaled_cost:.2f}\n"
        
        return report

def analyze_token_usage():
    """Run token usage analysis."""
    print("🔢 Analyzing Token Usage for Hybrid Attention RAG")
    print("=" * 60)
    
    # Create configuration
    config = TokenUsageConfig()
    
    # Create analyzer
    analyzer = TokenUsageAnalyzer(config)
    
    # Generate report
    report = analyzer.generate_usage_report()
    print(report)
    
    # Save detailed analysis
    analysis = analyzer.analyze_complete_system()
    with open('token_usage_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print("📁 Detailed analysis saved to: token_usage_analysis.json")
    
    return analysis

if __name__ == "__main__":
    analyze_token_usage()
