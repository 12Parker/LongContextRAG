#!/usr/bin/env python3
"""
Optimized NarrativeQA Hybrid RAG System (BM25 Only)

This optimized version focuses on:
1. Reducing latency through faster retrieval and optimized prompts
2. Reducing token usage through smart chunk selection and concise prompts
3. Maintaining quality while improving efficiency
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import sys
import os
import time
from collections import defaultdict
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  Warning: rank_bm25 not installed. Install with: pip install rank-bm25")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import config
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import tiktoken

logger = logging.getLogger(__name__)


class OptimizedBM25Retriever:
    """
    Optimized BM25-only retriever with performance improvements.
    
    Optimizations:
    1. Early termination for top-k retrieval
    2. Caching of tokenized documents
    3. Efficient score computation
    4. Smart chunk filtering
    """
    
    def __init__(self, 
                 documents: List[str],
                 k1: float = 1.5,
                 b: float = 0.75):
        """
        Initialize optimized BM25 retriever.
        
        Args:
            documents: List of document chunks (text)
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling document length normalization
        """
        self.documents = documents
        
        if not BM25_AVAILABLE:
            raise ValueError("BM25 not available. Install with: pip install rank-bm25")
        
        # Pre-tokenize documents for faster retrieval
        logger.info("Pre-tokenizing documents for BM25...")
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Calculate average document length for normalization
        self.avg_doc_length = sum(len(doc) for doc in self.tokenized_docs) / len(self.tokenized_docs) if documents else 0
        
        # Initialize BM25
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)
        logger.info("✅ BM25 index ready")
    
    def retrieve(self, 
                 query: str, 
                 k: int = 5,
                 min_score: float = 0.0,
                 return_scores: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using BM25 (optimized).
        
        Args:
            query: Query string
            k: Number of documents to retrieve (reduced default for speed)
            min_score: Minimum BM25 score threshold (filters low-quality matches)
            return_scores: Whether to return scores with documents
            
        Returns:
            List of document dictionaries with text and metadata
        """
        start_time = time.time()
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        if not tokenized_query:
            return []
        
        # Get BM25 scores (this is fast - O(n) where n = num docs)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices efficiently using argpartition (faster than full sort)
        # Only sort the top-k elements
        if k < len(bm25_scores):
            # Use argpartition for O(n) instead of O(n log n) full sort
            top_k_indices = np.argpartition(bm25_scores, -k)[-k:]
            # Sort only the top-k by score (descending)
            top_k_indices = top_k_indices[np.argsort(bm25_scores[top_k_indices])[::-1]]
        else:
            # If k >= len(scores), just sort all
            top_k_indices = np.argsort(bm25_scores)[::-1]
        
        # Filter by minimum score and build results
        results = []
        for idx in top_k_indices:
            score = float(bm25_scores[idx])
            if score >= min_score:
                result = {
                    'text': self.documents[idx],
                    'metadata': {'chunk_id': int(idx)},
                    'score': score
                }
                if return_scores:
                    result['scores'] = {'bm25': score}
                results.append(result)
                
                # Early termination if we have enough high-quality results
                if len(results) >= k:
                    break
        
        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieved {len(results)} documents in {retrieval_time*1000:.1f}ms")
        
        return results


class OptimizedNarrativeQAHybridRAG:
    """
    Optimized Hybrid RAG system for NarrativeQA with BM25-only retrieval.
    
    Key optimizations:
    1. Reduced top-k retrieval (5-7 chunks instead of 10)
    2. Smart chunk truncation (keep only relevant parts)
    3. Concise prompts (reduced token usage)
    4. Faster BM25-only retrieval (no dense search overhead)
    5. Context window optimization (target ~3000-4000 tokens)
    """
    
    def __init__(self, 
                 max_context_tokens: int = 4000,  # Reduced from 50000
                 chunk_size: int = 1200,  # Slightly smaller chunks
                 chunk_overlap: int = 150,
                 top_k_results: int = 5,  # Reduced from 10
                 db_path: str = "./narrativeqa_hybrid_vectordb",
                 story_text: str = None,
                 min_retrieval_score: float = 0.0):
        """
        Initialize optimized NarrativeQA Hybrid RAG system.
        
        Args:
            max_context_tokens: Maximum tokens for context (reduced for efficiency)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k_results: Number of top results to retrieve (reduced)
            db_path: Path to vector database
            story_text: The actual story text to use
            min_retrieval_score: Minimum BM25 score to include a chunk
        """
        self.max_context_tokens = max_context_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_results = top_k_results
        self.db_path = db_path
        self.story_text = story_text
        self.min_retrieval_score = min_retrieval_score
        
        # Initialize the LLM with optimized settings
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.2,  # Lower temperature for faster, more consistent responses
            max_tokens=150  # Limit response length for speed
        )
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize embeddings (for vector store, but we'll use BM25 only)
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name="text-embedding-3-large"
        )
        
        self.text_splitter = ClusterSemanticChunker(
            embedding_function=embedding_function,
            max_chunk_size=chunk_size,
            length_function=len
        )
        
        # Initialize embeddings (minimal - only for vector store creation)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize components
        self.vectorstore = None
        self.documents = []
        self.bm25_retriever = None
        
        self._initialize_vectordb()
        self._initialize_retrievers()
    
    def _initialize_vectordb(self):
        """Initialize the vector database with NarrativeQA stories."""
        if not self.story_text:
            print("  ⚠️  No story text provided. Please provide story_text parameter.")
            return
        
        # Check if we already have documents (from previous initialization)
        # This allows reusing the same system instance for multiple questions
        if self.documents:
            print(f"  ✅ Using existing {len(self.documents)} chunks")
            return
        
        print("🔧 Initializing Optimized NarrativeQA Vector Database...")
        self._create_vectordb()
    
    def _create_vectordb(self):
        """Create vector database from NarrativeQA stories."""
        try:
            # Split story into chunks
            print(f"  📄 Splitting story into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
            chunks = self.text_splitter.split_text(self.story_text)
            self.documents = chunks
            
            print(f"  ✅ Created {len(chunks)} chunks")
            
            # We don't actually need a vector store for BM25-only retrieval
            # Skip vector store creation to save time and avoid database issues
            self.vectorstore = None
            print(f"  ✅ Skipping vector store (BM25-only mode)")
            
        except Exception as e:
            print(f"❌ Error creating chunks: {e}")
            raise
    
    def _initialize_retrievers(self):
        """Initialize retrieval components."""
        print(f"🔧 Initializing optimized BM25 retriever...")
        
        if not BM25_AVAILABLE:
            raise ValueError("BM25 not available. Install with: pip install rank-bm25")
        
        print(f"  📊 Building BM25 index for {len(self.documents)} documents...")
        self.bm25_retriever = OptimizedBM25Retriever(
            documents=self.documents
        )
        print(f"  ✅ Optimized BM25 retriever ready")
    
    def _truncate_chunk_smart(self, chunk: str, max_tokens: int, query: str) -> str:
        """
        Intelligently truncate a chunk to fit token budget.
        
        Prioritizes:
        1. Sentences containing query terms
        2. Beginning and end of chunk (context)
        3. Middle content
        
        Args:
            chunk: Chunk text to truncate
            max_tokens: Maximum tokens allowed
            query: Query string (for finding relevant sentences)
            
        Returns:
            Truncated chunk text
        """
        # Check if chunk fits
        tokens = len(self.tokenizer.encode(chunk))
        if tokens <= max_tokens:
            return chunk
        
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', chunk)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            # Fallback: simple truncation
            words = chunk.split()
            max_words = max_tokens * 0.75  # Conservative estimate
            return " ".join(words[:int(max_words)])
        
        # Score sentences by query term presence
        query_terms = set(query.lower().split())
        sentence_scores = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            sentence_scores.append((score, sentence))
        
        # Sort by score (highest first)
        sentence_scores.sort(reverse=True)
        
        # Build truncated chunk: prioritize high-scoring sentences
        selected_sentences = []
        selected_tokens = 0
        
        # First, add high-scoring sentences
        for score, sentence in sentence_scores:
            if score > 0:  # Has query terms
                sent_tokens = len(self.tokenizer.encode(sentence))
                if selected_tokens + sent_tokens <= max_tokens * 0.8:
                    selected_sentences.append(sentence)
                    selected_tokens += sent_tokens
        
        # Then add remaining sentences if space allows
        remaining = [s for s in sentences if s not in selected_sentences]
        for sentence in remaining:
            sent_tokens = len(self.tokenizer.encode(sentence))
            if selected_tokens + sent_tokens <= max_tokens:
                selected_sentences.append(sentence)
                selected_tokens += sent_tokens
            else:
                break
        
        # If still too long, truncate last sentence
        result = ". ".join(selected_sentences)
        tokens = len(self.tokenizer.encode(result))
        if tokens > max_tokens:
            words = result.split()
            max_words = max_tokens * 0.75
            result = " ".join(words[:int(max_words)]) + "..."
        
        return result
    
    def retrieve_documents(self, 
                          query: str, 
                          k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using optimized BM25.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (defaults to top_k_results)
            
        Returns:
            List of document dictionaries
        """
        if k is None:
            k = self.top_k_results
        
        start_time = time.time()
        
        # Use optimized BM25 retriever
        results = self.bm25_retriever.retrieve(
            query, 
            k=k,
            min_score=self.min_retrieval_score,
            return_scores=True
        )
        
        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieved {len(results)} documents in {retrieval_time*1000:.1f}ms")
        
        return results
    
    def generate_response(self, question: str) -> Dict[str, Any]:
        """
        Generate a response using optimized hybrid RAG.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Retrieve documents (optimized - fewer chunks)
            docs = self.retrieve_documents(question, k=self.top_k_results)
            
            if not docs:
                return {
                    'response': "I couldn't find relevant information to answer your question.",
                    'context': '',
                    'retrieved_docs': 0,
                    'context_length': 0,
                    'context_tokens': 0,
                    'response_time': time.time() - start_time,
                    'method': 'hybrid_rag_bm25_optimized',
                    'retrieval_mode': 'bm25_optimized'
                }
            
            # Smart context building with token budget
            # Allocate tokens: ~70% for context, ~30% for prompt/question
            available_context_tokens = int(self.max_context_tokens * 0.7)
            tokens_per_chunk = available_context_tokens // len(docs) if docs else available_context_tokens
            
            context_parts = []
            total_tokens = 0
            
            for i, doc in enumerate(docs):
                chunk_text = doc['text']
                
                # Truncate chunk intelligently if needed
                if tokens_per_chunk < len(self.tokenizer.encode(chunk_text)):
                    chunk_text = self._truncate_chunk_smart(
                        chunk_text, 
                        tokens_per_chunk, 
                        question
                    )
                
                chunk_tokens = len(self.tokenizer.encode(chunk_text))
                if total_tokens + chunk_tokens <= available_context_tokens:
                    context_parts.append(chunk_text)
                    total_tokens += chunk_tokens
                else:
                    # Stop if we've exceeded budget
                    break
            
            context = "\n\n".join(context_parts)
            context_tokens = len(self.tokenizer.encode(context))
            
            # Optimized concise prompt
            prompt = f"""Context:
{context}

Question: {question}

Answer concisely (1-2 sentences):"""

            # Generate response
            response_start = time.time()
            response = self.llm.invoke(prompt)
            generation_time = time.time() - response_start
            
            generated_answer = response.content.strip()
            
            elapsed_time = time.time() - start_time
            
            return {
                'response': generated_answer,
                'context': context,
                'retrieved_docs': len(docs),
                'context_length': len(context),
                'context_tokens': context_tokens,
                'response_time': elapsed_time,
                'retrieval_time': elapsed_time - generation_time,
                'generation_time': generation_time,
                'method': 'hybrid_rag_bm25_optimized',
                'retrieval_mode': 'bm25_optimized'
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return {
                'response': f"Error generating response: {str(e)}",
                'context': '',
                'retrieved_docs': 0,
                'context_length': 0,
                'context_tokens': 0,
                'response_time': time.time() - start_time,
                'method': 'hybrid_rag_bm25_optimized',
                'error': str(e)
            }


def main():
    """Main function for testing the optimized NarrativeQA Hybrid RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized NarrativeQA Hybrid RAG (BM25 Only)")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size for text splitting")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--db-path", type=str, default="./narrativeqa_hybrid_vectordb_optimized", 
                       help="Path to vector database")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum context tokens")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum BM25 score threshold")
    
    args = parser.parse_args()
    
    # Check BM25 availability
    if not BM25_AVAILABLE:
        print("❌ Error: BM25 mode requires rank-bm25 package")
        print("Install with: pip install rank-bm25")
        return
    
    # Initialize Optimized Hybrid RAG system
    print("🚀 Initializing Optimized NarrativeQA Hybrid RAG...")
    print(f"  Mode: BM25-only (optimized)")
    print(f"  Top-k: {args.top_k}")
    print(f"  Max context tokens: {args.max_tokens}")
    
    # Note: story_text should be provided in actual usage
    hybrid_rag = OptimizedNarrativeQAHybridRAG(
        chunk_size=args.chunk_size,
        top_k_results=args.top_k,
        db_path=args.db_path,
        max_context_tokens=args.max_tokens,
        min_retrieval_score=args.min_score
    )
    
    if args.interactive:
        print("\n💬 Interactive mode - Enter queries (type 'quit' to exit)")
        while True:
            query = input("\n🔍 Enter your question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\n🤖 Processing: {query}")
            result = hybrid_rag.generate_response(query)
            
            print(f"\n📝 Response:")
            print(f"{result['response']}")
            print(f"\n📊 Metadata:")
            print(f"  Retrieved docs: {result['retrieved_docs']}")
            print(f"  Context tokens: {result['context_tokens']}")
            print(f"  Total time: {result['response_time']:.2f}s")
            if 'retrieval_time' in result:
                print(f"  Retrieval time: {result['retrieval_time']:.2f}s")
            if 'generation_time' in result:
                print(f"  Generation time: {result['generation_time']:.2f}s")
    
    elif args.query:
        print(f"\n🤖 Processing: {args.query}")
        result = hybrid_rag.generate_response(args.query)
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Retrieved docs: {result['retrieved_docs']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Total time: {result['response_time']:.2f}s")
        if 'retrieval_time' in result:
            print(f"  Retrieval time: {result['retrieval_time']:.2f}s")
        if 'generation_time' in result:
            print(f"  Generation time: {result['generation_time']:.2f}s")
    
    else:
        print("❌ Please provide a query or use --interactive mode")
        print("Usage: python narrativeqa_hybrid_rag_optimized.py 'Your question here'")

if __name__ == "__main__":
    main()

