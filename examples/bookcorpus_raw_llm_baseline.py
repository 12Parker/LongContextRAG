#!/usr/bin/env python3
"""
BookCorpus Raw LLM Baseline

This script implements a proper raw LLM baseline for querying BookCorpus.
It loads actual BookCorpus data and passes it directly to the LLM as context,
providing a true baseline to compare against RAG systems.

Key Features:
- Loads actual BookCorpus dataset
- Handles token limits properly
- Provides realistic book content as context
- Simple, focused implementation without RAG components

Usage:
    python examples/bookcorpus_raw_llm_baseline.py
    python examples/bookcorpus_raw_llm_baseline.py "Your query here"
    python examples/bookcorpus_raw_llm_baseline.py --interactive
"""

import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import config
from langchain_openai import ChatOpenAI
import tiktoken
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookCorpusRawLLMBaseline:
    """
    Raw LLM baseline using actual BookCorpus data.
    
    This provides a proper baseline by:
    1. Loading real BookCorpus documents
    2. Passing them directly to the LLM as context
    3. No retrieval, no chunking, no RAG - just raw LLM with full context
    """
    
    def __init__(self, max_context_tokens: int = 32000, num_books: int = 10):
        """
        Initialize the BookCorpus raw LLM baseline.
        
        Args:
            max_context_tokens: Maximum number of tokens to include in context
            num_books: Number of books to load from BookCorpus
        """
        self.max_context_tokens = max_context_tokens
        self.num_books = num_books
        
        # Initialize the LLM directly
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Load BookCorpus data
        self.books = self._load_bookcorpus_data()
        
    def _load_bookcorpus_data(self) -> List[Dict[str, str]]:
        """Load actual BookCorpus data."""
        print("📚 Loading BookCorpus dataset...")
        
        try:
            # Load BookCorpus dataset
            dataset = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)
            
            books = []
            count = 0
            
            print(f"🔄 Loading {self.num_books} books from BookCorpus...")
            for example in dataset:
                if count >= self.num_books:
                    break
                
                # Extract text content
                text = example.get('text', '')
                
                # Filter out very short or very long texts
                if 100 < len(text) < 10000:  # Reasonable book excerpt length
                    books.append({
                        'text': text,
                        'book_id': count,
                        'length': len(text)
                    })
                    count += 1
                    
                    if count % 5 == 0:
                        print(f"  Loaded {count}/{self.num_books} books...")
            
            print(f"✅ Successfully loaded {len(books)} books from BookCorpus")
            return books
            
        except Exception as e:
            logger.error(f"Failed to load BookCorpus: {e}")  
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def _truncate_books(self, books: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Truncate books to fit within token limit."""
        total_tokens = 0
        truncated_books = []
        
        for book in books:
            book_tokens = self._count_tokens(book['text'])
            
            if total_tokens + book_tokens <= max_tokens:
                truncated_books.append(book)
                total_tokens += book_tokens
            else:
                # Try to fit a partial book
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only if we have meaningful space left
                    # Truncate the book to fit
                    truncated_text = self.tokenizer.decode(
                        self.tokenizer.encode(book['text'])[:remaining_tokens]
                    )
                    truncated_books.append({
                        'text': truncated_text,
                        'book_id': book['book_id'],
                        'length': len(truncated_text)
                    })
                break
        
        logger.info(f"Selected {len(truncated_books)} books, total tokens: {total_tokens}")
        return truncated_books
    
    def _prepare_context(self, query: str) -> str:
        """Prepare context by combining selected books."""
        # Reserve tokens for query and instructions
        available_tokens = self.max_context_tokens - 500
        
        # Select books that fit within token limit
        selected_books = self._truncate_books(self.books, available_tokens)
        
        # Combine book texts
        book_texts = [book['text'] for book in selected_books]
        combined_text = "\n\n---\n\n".join(book_texts)
        
        # Create final context
        context = f"""Context from {len(selected_books)} books:

{combined_text}

Question: {query}

Please provide a comprehensive answer based on the provided context from these books. 
If the context doesn't contain enough information to answer the question, 
please say so and provide what information you can from the available context."""
        
        return context
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response using raw LLM with BookCorpus context.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare context
            context = self._prepare_context(query)
            
            # Count tokens
            context_tokens = self._count_tokens(context)
            
            # Generate response using raw LLM
            response = self.llm.invoke(context)
            
            elapsed_time = time.time() - start_time
            
            return {
                'response': response.content,
                'method': 'raw_llm_bookcorpus',
                'context_length': len(context),
                'context_tokens': context_tokens,
                'response_time': elapsed_time,
                'response_length': len(response.content),
                'books_used': len(self.books),
                'max_context_tokens': self.max_context_tokens,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"Error: {str(e)}",
                'method': 'raw_llm_bookcorpus',
                'context_length': 0,
                'context_tokens': 0,
                'response_time': time.time() - start_time,
                'response_length': 0,
                'books_used': 0,
                'error': str(e),
                'query': query
            }
    
    def run_baseline_tests(self, queries: List[str] = None) -> Dict[str, Any]:
        """
        Run baseline tests with multiple queries.
        
        Args:
            queries: List of test queries (optional)
            
        Returns:
            Dictionary containing test results
        """
        if queries is None:
            queries = [
                "What are the main themes in these stories?",
                "Describe the main characters and their development",
                "What conflicts and challenges do the characters face?",
                "How do the authors create atmosphere and setting?",
                "What narrative techniques are used in these stories?",
                "Compare the different story genres and styles",
                "What emotions and feelings do these stories evoke?",
                "How do the stories begin and end?"
            ]
        
        print("🧪 Running BookCorpus Raw LLM Baseline Tests")
        print("=" * 60)
        print(f"📚 Books loaded: {len(self.books)}")
        print(f"🔢 Max context tokens: {self.max_context_tokens}")
        print(f"❓ Test queries: {len(queries)}")
        print("=" * 60)
        
        results = []
        total_time = 0
        total_context_length = 0
        total_response_length = 0
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 Query {i}/{len(queries)}: {query}")
            print("-" * 60)
            
            result = self.generate_response(query)
            results.append(result)
            
            total_time += result['response_time']
            total_context_length += result['context_length']
            total_response_length += result['response_length']
            
            print(f"⏱️  Time: {result['response_time']:.2f}s")
            print(f"📄 Context: {result['context_length']} chars ({result['context_tokens']} tokens)")
            print(f"💬 Response: {result['response_length']} chars")
            print(f"📚 Books used: {result['books_used']}")
            
            # Show response preview
            response_preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
            print(f"💭 Response preview: {response_preview}")
        
        # Summary statistics
        summary = {
            'total_queries': len(queries),
            'avg_response_time': total_time / len(queries),
            'avg_context_length': total_context_length / len(queries),
            'avg_response_length': total_response_length / len(queries),
            'total_context_tokens': sum(r['context_tokens'] for r in results),
            'method': 'raw_llm_bookcorpus',
            'timestamp': datetime.now().isoformat(),
            'books_loaded': len(self.books)
        }
        
        print(f"\n📊 BASELINE SUMMARY")
        print("=" * 60)
        print(f"📝 Total queries: {summary['total_queries']}")
        print(f"⏱️  Average response time: {summary['avg_response_time']:.2f}s")
        print(f"📄 Average context length: {summary['avg_context_length']:.0f} chars")
        print(f"💬 Average response length: {summary['avg_response_length']:.0f} chars")
        print(f"🔢 Total context tokens: {summary['total_context_tokens']}")
        print(f"📚 Books loaded: {summary['books_loaded']}")
        
        return {
            'summary': summary,
            'results': results,
            'queries': queries
        }

def main():
    """Main function to run the BookCorpus raw LLM baseline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BookCorpus Raw LLM Baseline")
    parser.add_argument("query", nargs="?", help="Query to test (optional)")
    parser.add_argument("--max-tokens", type=int, default=32000, help="Maximum context tokens")
    parser.add_argument("--num-books", type=int, default=10, help="Number of books to load")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize baseline
    baseline = BookCorpusRawLLMBaseline(
        max_context_tokens=args.max_tokens,
        num_books=args.num_books
    )
    
    if args.interactive:
        # Interactive mode
        print("🔍 Interactive BookCorpus Raw LLM Baseline Mode")
        print("Enter queries to test the raw LLM with BookCorpus context (type 'quit' to exit)")
        print("=" * 60)
        
        while True:
            query = input("\nEnter query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                result = baseline.generate_response(query)
                print(f"\n📊 RAW LLM BASELINE RESULT")
                print("=" * 60)
                print(f"Response: {result['response']}")
                print(f"Method: {result['method']}")
                print(f"Context Length: {result['context_length']} chars")
                print(f"Context Tokens: {result['context_tokens']}")
                print(f"Response Time: {result['response_time']:.2f}s")
                print(f"Books Used: {result['books_used']}")
    
    elif args.query:
        # Single query test
        result = baseline.generate_response(args.query)
        print(f"\n📊 RAW LLM BASELINE RESULT")
        print("=" * 80)
        print(f"Response: {result['response']}")
        print(f"Method: {result['method']}")
        print(f"Context Length: {result['context_length']} chars")
        print(f"Context Tokens: {result['context_tokens']}")
        print(f"Response Time: {result['response_time']:.2f}s")
        print(f"Books Used: {result['books_used']}")
        
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/bookcorpus_raw_llm_baseline_{timestamp}.json"
            os.makedirs("results", exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {filename}")
    
    else:
        # Run full baseline tests
        results = baseline.run_baseline_tests()
        
        if args.save_results:
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/bookcorpus_raw_llm_baseline_{timestamp}.json"
            
            os.makedirs("results", exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
