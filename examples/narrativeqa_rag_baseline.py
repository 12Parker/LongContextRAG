#!/usr/bin/env python3
"""
NarrativeQA RAG Baseline

This script implements a RAG system specifically for NarrativeQA stories.
It creates a vector database from the correct story for each question,
ensuring proper context for answering.

Key Features:
- Uses the correct story for each question
- Creates vector embeddings from story chunks
- Retrieves relevant chunks for questions
- Combines retrieved context with LLM for answers

Usage:
    python examples/narrativeqa_rag_baseline.py
    python examples/narrativeqa_rag_baseline.py "Your query here"
    python examples/narrativeqa_rag_baseline.py --interactive
"""

import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import hashlib

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NarrativeQARAGBaseline:
    """
    RAG baseline using NarrativeQA stories.
    
    This implements a RAG pipeline specifically for NarrativeQA:
    1. Load the correct story for each question
    2. Create vector database from story chunks
    3. Retrieve relevant chunks for questions
    4. Generate answers using retrieved context
    """
    
    def __init__(self, 
                 max_context_tokens: int = 50000,
                 chunk_size: int = 2000,
                 chunk_overlap: int = 400,
                 top_k_results: int = 20,
                 db_path: str = "./narrativeqa_vectordb",
                 story_text: str = None):
        """
        Initialize the NarrativeQA RAG baseline.
        
        Args:
            max_context_tokens: Maximum tokens for context
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k_results: Number of top results to retrieve
            db_path: Path to vector database
            story_text: The actual story text to use
        """
        self.max_context_tokens = max_context_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_results = top_k_results
        self.db_path = db_path
        self.story_text = story_text
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize semantic chunker
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name="text-embedding-3-large"
        )
        
        self.text_splitter = ClusterSemanticChunker(
            embedding_function=embedding_function,
            max_chunk_size=chunk_size,
            length_function=len
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vector store
        self.vectorstore = None
        self._initialize_vectordb()
    
    def _initialize_vectordb(self):
        """Initialize the vector database with NarrativeQA stories."""
        print("🔧 Initializing NarrativeQA Vector Database...")
        
        # Always create a fresh database to ensure correct story
        print("🔄 Creating fresh database with correct story...")
        self._create_vectordb()
    
    def _create_vectordb(self):
        """Create vector database from NarrativeQA stories."""
        print("📚 Loading NarrativeQA story...")
        
        try:
            # Use the provided story text or load from dataset
            if self.story_text:
                story = self.story_text
                print(f"  ✅ Using provided story text ({len(story)} characters)")
            else:
                from datasets import load_dataset
                dataset = load_dataset("narrativeqa", split="train")
                story_data = dataset[0].get('document', '')
                if isinstance(story_data, dict):
                    story = story_data.get('text', '')
                else:
                    story = str(story_data)
                print(f"  ✅ Loaded story from dataset ({len(story)} characters)")
            
            if not story:
                raise ValueError("No story text available")
            
            # Split story into chunks
            chunks = self.text_splitter.split_text(story)
            print(f"  📊 Created {len(chunks)} chunks from story")
            
            # Create documents and metadata
            documents = []
            metadatas = []
            
            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    'story_id': 'current_story',
                    'chunk_id': j,
                    'total_chunks': len(chunks),
                    'story_index': 0
                })
            
            # Create vector store
            print("🔧 Creating vector database...")
            self.vectorstore = Chroma.from_texts(
                texts=documents,
                metadatas=metadatas,
                embedding=self.embeddings,
                persist_directory=self.db_path
            )
            
            print(f"✅ Vector database created at {self.db_path}")
            
        except Exception as e:
            print(f"❌ Error creating vector database: {e}")
            raise
    
    def generate_response(self, question: str) -> Dict[str, Any]:
        """
        Generate a response using RAG.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Retrieve relevant chunks
            print(f"🔍 Retrieving relevant chunks for: {question[:50]}...")
            docs = self.vectorstore.similarity_search(
                question, 
                k=self.top_k_results
            )
            
            if not docs:
                return {
                    'response': "I couldn't find relevant information to answer your question.",
                    'context': '',
                    'retrieved_docs': 0,
                    'context_length': 0,
                    'context_tokens': 0,
                    'response_time': time.time() - start_time,
                    'method': 'narrativeqa_rag'
                }
            
            # Combine retrieved chunks
            context_parts = []
            for i, doc in enumerate(docs):
                context_parts.append(f"Chunk {i+1}: {doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Check context length
            context_tokens = len(self.tokenizer.encode(context))
            if context_tokens > self.max_context_tokens:
                # Truncate context if too long
                words = context.split()
                max_words = int(self.max_context_tokens * 0.8)  # Leave room for question
                context = " ".join(words[:max_words]) + "..."
                context_tokens = len(self.tokenizer.encode(context))
            
            # Create prompt
            prompt = f"""Based on the following story excerpts, answer the question.

Context:
{context}

Question: {question}

Please provide a concise, direct answer (1-2 sentences maximum). Focus on the key facts from the story excerpts. If the excerpts don't contain enough information to answer the question, please say so briefly."""

            # Generate response
            response = self.llm.invoke(prompt)
            generated_answer = response.content
            
            elapsed_time = time.time() - start_time
            
            print(f"  ✅ Response generated in {elapsed_time:.2f}s")
            print(f"  📊 Retrieved {len(docs)} chunks")
            print(f"  📊 Context tokens: {context_tokens}")
            
            return {
                'response': generated_answer,
                'context': context,
                'retrieved_docs': len(docs),
                'context_length': len(context),
                'context_tokens': context_tokens,
                'response_time': elapsed_time,
                'method': 'narrativeqa_rag'
            }
            
        except Exception as e:
            print(f"  ❌ Error generating response: {e}")
            return {
                'response': f"Error generating response: {str(e)}",
                'context': '',
                'retrieved_docs': 0,
                'context_length': 0,
                'context_tokens': 0,
                'response_time': time.time() - start_time,
                'method': 'narrativeqa_rag',
                'error': str(e)
            }

def main():
    """Main function for testing the NarrativeQA RAG baseline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NarrativeQA RAG Baseline")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--db-path", type=str, default="./narrativeqa_vectordb", help="Path to vector database")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    print("🚀 Initializing NarrativeQA RAG Baseline...")
    rag = NarrativeQARAGBaseline(
        chunk_size=args.chunk_size,
        top_k_results=args.top_k,
        db_path=args.db_path
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
            result = rag.generate_response(query)
            
            print(f"\n📝 Response:")
            print(f"{result['response']}")
            print(f"\n📊 Metadata:")
            print(f"  Retrieved docs: {result['retrieved_docs']}")
            print(f"  Context tokens: {result['context_tokens']}")
            print(f"  Response time: {result['response_time']:.2f}s")
    
    elif args.query:
        print(f"\n🤖 Processing: {args.query}")
        result = rag.generate_response(args.query)
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Retrieved docs: {result['retrieved_docs']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")
    
    else:
        # Default test query
        test_query = "Who is the main character in the story?"
        print(f"\n🤖 Testing with: {test_query}")
        result = rag.generate_response(test_query)
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Retrieved docs: {result['retrieved_docs']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")

if __name__ == "__main__":
    main()