#!/usr/bin/env python3
"""
Standard RAG Baseline using VectorDB/build_db

This script implements a standard RAG approach using the VectorDBBuilder
to create a vector database from BookCorpus and retrieve relevant chunks
for query answering. This provides a proper RAG baseline to compare against
the raw LLM baseline.

Key Features:
- Uses VectorDBBuilder for vector database creation
- Retrieves relevant chunks based on query similarity
- Combines retrieved chunks as context for LLM
- Standard RAG pipeline: Retrieve → Combine → Generate

Usage:
    python examples/standard_rag_baseline.py
    python examples/standard_rag_baseline.py "Your query here"
    python examples/standard_rag_baseline.py --interactive
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

# Add VectorDB to path
sys.path.append(os.path.join(project_root, 'VectorDB'))

from core.config import config
from langchain_openai import ChatOpenAI
from build_db import VectorDBBuilder
from vectordb_manager import VectorDBManager
import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandardRAGBaseline:
    """
    Standard RAG baseline using VectorDBBuilder.
    
    This implements the classic RAG pipeline:
    1. Build vector database from BookCorpus
    2. Retrieve relevant chunks for queries
    3. Combine chunks as context for LLM
    4. Generate response using retrieved context
    """
    
    def __init__(self, 
                 max_context_tokens: int = 8000,
                 num_documents: int = 5000,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 top_k_results: int = 10,
                 use_persistent_db: bool = True,
                 db_path: str = "./full_bookcorpus_db"):
        """
        Initialize the standard RAG baseline.
        
        Args:
            max_context_tokens: Maximum tokens for context
            num_documents: Number of documents to process from BookCorpus
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k_results: Number of top results to retrieve
            use_persistent_db: Whether to use persistent database
            db_path: Path to persistent database
        """
        self.max_context_tokens = max_context_tokens
        self.num_documents = num_documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_results = top_k_results
        self.use_persistent_db = use_persistent_db
        self.db_path = db_path
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize vector database
        self._initialize_vectordb()
    
    def _initialize_vectordb(self):
        """Initialize the vector database with BookCorpus data."""
        print("🔧 Initializing Standard RAG Vector Database...")
        
        if self.use_persistent_db:
            # Use persistent database manager
            print("📚 Using persistent database...")
            print(f"📁 Database path: {self.db_path}")
            print(f"📄 Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
            print(f"🔍 Top-k results: {self.top_k_results}")
            
            try:
                # Initialize VectorDBManager
                self.vectordb_manager = VectorDBManager(
                    db_path=self.db_path,
                    collection_name="full_bookcorpus",
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    batch_size=100,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                # Check if database is ready
                if self.vectordb_manager.is_database_ready():
                    print("✅ Persistent database found and ready")
                    self.vectordb_initialized = True
                else:
                    print("🔧 Database not found, initializing...")
                    # Initialize with smaller dataset first
                    success = self.vectordb_manager.initialize_database()
                    if success:
                        print("✅ Database initialized successfully")
                        self.vectordb_initialized = True
                    else:
                        print("❌ Database initialization failed")
                        self.vectordb_initialized = False
                
            except Exception as e:
                logger.error(f"Failed to initialize persistent database: {e}")
                print(f"❌ Persistent database initialization failed: {e}")
                self.vectordb_initialized = False
        else:
            # Use original VectorDBBuilder approach
            print(f"📚 Processing {self.num_documents} documents from BookCorpus")
            print(f"📄 Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
            print(f"🔍 Top-k results: {self.top_k_results}")
            
            try:
                # Initialize VectorDBBuilder
                self.vectordb_config = {
                    'db_path': './vector_store_standard_rag',
                    'collection_name': 'standard_rag_baseline',
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'batch_size': 100
                }
                
                self.vectordb_builder = VectorDBBuilder(**self.vectordb_config)
                
                # Create or reset collection
                self.vectordb_builder.create_or_reset_collection(reset=True)
                
                # Process BookCorpus dataset
                self.vectordb_builder.process_dataset(
                    dataset_name="rojagtap/bookcorpus",
                    num_documents=self.num_documents,
                    min_text_length=100  # Minimum text length for processing
                )
                
                self.vectordb_initialized = True
                print("✅ Standard RAG vector database initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize vector database: {e}")
                print(f"❌ Vector database initialization failed: {e}")
                self.vectordb_initialized = False
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit within token limit."""
        tokens = self.tokenizer.encode(context)
        
        if len(tokens) <= max_tokens:
            return context
        
        # Truncate and decode back to text
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        logger.info(f"Truncated context from {len(tokens)} to {len(truncated_tokens)} tokens")
        return truncated_text
    
    def retrieve_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for the query.
        
        Args:
            query: The user query
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.vectordb_initialized:
            raise ValueError("Vector database not initialized")
        
        try:
            if self.use_persistent_db:
                # Use persistent database manager
                results = self.vectordb_manager.query(query, n_results=self.top_k_results)
            else:
                # Use original VectorDBBuilder
                results = self.vectordb_builder.query(query, n_results=self.top_k_results)
            
            # Convert results to chunk format
            chunks = []
            for i, (doc_text, distance, metadata) in enumerate(zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                chunks.append({
                    'text': doc_text,
                    'distance': distance,
                    'metadata': metadata,
                    'chunk_id': i
                })
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _prepare_context(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved chunks."""
        if not chunks:
            return f"Question: {query}\n\nNo relevant context found."
        
        # Combine chunk texts
        chunk_texts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_texts.append(f"Context {i}:\n{chunk['text']}")
        
        combined_context = "\n\n---\n\n".join(chunk_texts)
        
        # Create final context
        context = f"""Context from {len(chunks)} relevant passages:

{combined_context}

Question: {query}

Please provide a comprehensive answer based on the provided context. 
If the context doesn't contain enough information to answer the question, 
please say so and provide what information you can from the available context."""
        
        # Truncate if necessary
        context_tokens = self._count_tokens(context)
        if context_tokens > self.max_context_tokens:
            context = self._truncate_context(context, self.max_context_tokens)
        
        return context
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response using standard RAG approach.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(query)
            
            # Step 2: Prepare context
            context = self._prepare_context(query, chunks)
            
            # Step 3: Count tokens
            context_tokens = self._count_tokens(context)
            
            # Step 4: Generate response using LLM
            response = self.llm.invoke(context)
            
            elapsed_time = time.time() - start_time
            
            return {
                'response': response.content,
                'method': 'standard_rag',
                'context_length': len(context),
                'context_tokens': context_tokens,
                'response_time': elapsed_time,
                'response_length': len(response.content),
                'chunks_retrieved': len(chunks),
                'top_k_results': self.top_k_results,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'query': query,
                'chunks': chunks  # Include chunk details for analysis
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"Error: {str(e)}",
                'method': 'standard_rag',
                'context_length': 0,
                'context_tokens': 0,
                'response_time': time.time() - start_time,
                'response_length': 0,
                'chunks_retrieved': 0,
                'error': str(e),
                'query': query
            }
    
    def get_vectordb_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        if not self.vectordb_initialized:
            return {'error': 'Vector database not initialized'}
        
        try:
            if self.use_persistent_db:
                # Get stats from persistent database manager
                stats = self.vectordb_manager.get_database_stats()
                if 'error' in stats:
                    return stats
                
                return {
                    'collection_name': stats['collection_name'],
                    'total_chunks': stats['total_chunks'],
                    'chunk_size': stats['chunk_size'],
                    'chunk_overlap': stats['chunk_overlap'],
                    'embedding_model': stats['embedding_model'],
                    'top_k_results': self.top_k_results,
                    'database_path': stats['database_path'],
                    'is_persistent': True
                }
            else:
                # Get collection info from original builder
                collection = self.vectordb_builder.client.get_collection(
                    name=self.vectordb_config['collection_name']
                )
                
                count = collection.count()
                
                return {
                    'collection_name': self.vectordb_config['collection_name'],
                    'total_chunks': count,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'embedding_model': self.vectordb_config['embedding_model'],
                    'top_k_results': self.top_k_results,
                    'is_persistent': False
                }
            
        except Exception as e:
            return {'error': str(e)}
    
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
        
        print("🧪 Running Standard RAG Baseline Tests")
        print("=" * 60)
        print(f"📚 Documents processed: {self.num_documents}")
        print(f"📄 Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        print(f"🔍 Top-k results: {self.top_k_results}")
        print(f"🔢 Max context tokens: {self.max_context_tokens}")
        print(f"❓ Test queries: {len(queries)}")
        print("=" * 60)
        
        # Show vector database stats
        stats = self.get_vectordb_stats()
        if 'error' not in stats:
            print(f"📊 Vector DB: {stats['total_chunks']} chunks")
        
        results = []
        total_time = 0
        total_context_length = 0
        total_response_length = 0
        total_chunks_retrieved = 0
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 Query {i}/{len(queries)}: {query}")
            print("-" * 60)
            
            result = self.generate_response(query)
            results.append(result)
            
            total_time += result['response_time']
            total_context_length += result['context_length']
            total_response_length += result['response_length']
            total_chunks_retrieved += result['chunks_retrieved']
            
            print(f"⏱️  Time: {result['response_time']:.2f}s")
            print(f"📄 Context: {result['context_length']} chars ({result['context_tokens']} tokens)")
            print(f"💬 Response: {result['response_length']} chars")
            print(f"🔍 Chunks retrieved: {result['chunks_retrieved']}")
            
            # Show response preview
            response_preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
            print(f"💭 Response preview: {response_preview}")
        
        # Summary statistics
        summary = {
            'total_queries': len(queries),
            'avg_response_time': total_time / len(queries),
            'avg_context_length': total_context_length / len(queries),
            'avg_response_length': total_response_length / len(queries),
            'avg_chunks_retrieved': total_chunks_retrieved / len(queries),
            'total_context_tokens': sum(r['context_tokens'] for r in results),
            'method': 'standard_rag',
            'timestamp': datetime.now().isoformat(),
            'vectordb_stats': stats
        }
        
        print(f"\n📊 BASELINE SUMMARY")
        print("=" * 60)
        print(f"📝 Total queries: {summary['total_queries']}")
        print(f"⏱️  Average response time: {summary['avg_response_time']:.2f}s")
        print(f"📄 Average context length: {summary['avg_context_length']:.0f} chars")
        print(f"💬 Average response length: {summary['avg_response_length']:.0f} chars")
        print(f"🔍 Average chunks retrieved: {summary['avg_chunks_retrieved']:.1f}")
        print(f"🔢 Total context tokens: {summary['total_context_tokens']}")
        print(f"📊 Vector DB chunks: {stats.get('total_chunks', 'N/A')}")
        
        return {
            'summary': summary,
            'results': results,
            'queries': queries
        }

def main():
    """Main function to run the standard RAG baseline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standard RAG Baseline using VectorDB")
    parser.add_argument("query", nargs="?", help="Query to test (optional)")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Maximum context tokens")
    parser.add_argument("--num-documents", type=int, default=5000, help="Number of documents to process")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to retrieve")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--use-persistent-db", action="store_true", default=True, help="Use persistent database (default: True)")
    parser.add_argument("--no-persistent-db", action="store_true", help="Disable persistent database")
    parser.add_argument("--db-path", type=str, default="./full_bookcorpus_db", help="Path to persistent database")
    
    args = parser.parse_args()
    
    # Determine if persistent database should be used
    use_persistent_db = args.use_persistent_db and not args.no_persistent_db
    
    # Initialize baseline
    baseline = StandardRAGBaseline(
        max_context_tokens=args.max_tokens,
        num_documents=args.num_documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k_results=args.top_k,
        use_persistent_db=use_persistent_db,
        db_path=args.db_path
    )
    
    if args.interactive:
        # Interactive mode
        print("🔍 Interactive Standard RAG Baseline Mode")
        print("Enter queries to test the standard RAG system (type 'quit' to exit)")
        print("=" * 60)
        
        while True:
            query = input("\nEnter query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                result = baseline.generate_response(query)
                print(f"\n📊 STANDARD RAG BASELINE RESULT")
                print("=" * 60)
                print(f"Response: {result['response']}")
                print(f"Method: {result['method']}")
                print(f"Context Length: {result['context_length']} chars")
                print(f"Context Tokens: {result['context_tokens']}")
                print(f"Response Time: {result['response_time']:.2f}s")
                print(f"Chunks Retrieved: {result['chunks_retrieved']}")
    
    elif args.query:
        # Single query test
        result = baseline.generate_response(args.query)
        print(f"\n📊 STANDARD RAG BASELINE RESULT")
        print("=" * 80)
        print(f"Response: {result['response']}")
        print(f"Method: {result['method']}")
        print(f"Context Length: {result['context_length']} chars")
        print(f"Context Tokens: {result['context_tokens']}")
        print(f"Response Time: {result['response_time']:.2f}s")
        print(f"Chunks Retrieved: {result['chunks_retrieved']}")
        
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/standard_rag_baseline_{timestamp}.json"
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
            filename = f"results/standard_rag_baseline_{timestamp}.json"
            
            os.makedirs("results", exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
