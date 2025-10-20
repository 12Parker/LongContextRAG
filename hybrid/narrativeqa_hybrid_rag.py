#!/usr/bin/env python3
"""
NarrativeQA Hybrid RAG System

This provides a hybrid attention RAG system specifically designed for NarrativeQA stories.
It combines neural retrieval with attention mechanisms for better question answering.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import sys
import os
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import config
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import tiktoken

logger = logging.getLogger(__name__)

class NarrativeQAHybridRAG:
    """
    Hybrid attention RAG system specifically for NarrativeQA stories.
    
    This system combines:
    1. Neural retrieval for finding relevant story segments
    2. Attention mechanisms for focusing on important parts
    3. LLM generation for final answer synthesis
    """
    
    def __init__(self, 
                 max_context_tokens: int = 50000,
                 chunk_size: int = 2000,
                 chunk_overlap: int = 400,
                 top_k_results: int = 20,
                 db_path: str = "./narrativeqa_hybrid_vectordb",
                 story_text: str = None,
                 use_hybrid_attention: bool = True):
        """
        Initialize the NarrativeQA Hybrid RAG system.
        
        Args:
            max_context_tokens: Maximum tokens for context
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k_results: Number of top results to retrieve
            db_path: Path to vector database
            story_text: The actual story text to use
            use_hybrid_attention: Whether to use hybrid attention mechanisms
        """
        self.max_context_tokens = max_context_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_results = top_k_results
        self.db_path = db_path
        self.story_text = story_text
        self.use_hybrid_attention = use_hybrid_attention
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.document_embeddings = None
        self._initialize_vectordb()
        
        if use_hybrid_attention:
            self._initialize_attention_components()
    
    def _initialize_attention_components(self):
        """Initialize attention components for hybrid processing."""
        # Get actual embedding dimension from the model
        test_embedding = self.embeddings.embed_query("test")
        self.attention_dim = len(test_embedding)
        self.num_attention_heads = 8
        self.attention_dropout = 0.1
        
        # Initialize simple attention layer
        self.attention_layer = torch.nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )
        
        # Initialize neural retriever (simplified)
        self.retriever_hidden_dim = 512
        self.neural_retriever = torch.nn.Sequential(
            torch.nn.Linear(self.attention_dim * 2, self.retriever_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.retriever_hidden_dim, 1)
        )
        
        logger.info(f"Hybrid attention components initialized with embedding dim: {self.attention_dim}")
    
    def _initialize_vectordb(self):
        """Initialize the vector database with NarrativeQA stories."""
        print("🔧 Initializing NarrativeQA Hybrid Vector Database...")
        
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
            
            # Store document embeddings for hybrid processing
            if self.use_hybrid_attention:
                self._prepare_document_embeddings(documents)
            
            print(f"✅ Vector database created at {self.db_path}")
            
        except Exception as e:
            print(f"❌ Error creating vector database: {e}")
            raise
    
    def _prepare_document_embeddings(self, documents: List[str]):
        """Prepare document embeddings for hybrid processing."""
        try:
            # Create embeddings for all documents
            embeddings = self.embeddings.embed_documents(documents)
            embeddings_array = np.array(embeddings)
            
            # Store as tensor for hybrid processing
            self.document_embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
            
            logger.info(f"Prepared {len(embeddings)} document embeddings for hybrid processing")
            
        except Exception as e:
            logger.warning(f"Failed to prepare document embeddings: {e}")
            self.document_embeddings = None
    
    def _neural_retrieve(self, query_embedding: torch.Tensor, top_k: int = None) -> List[int]:
        """Neural retrieval to find most relevant document chunks."""
        if self.document_embeddings is None or not self.use_hybrid_attention:
            return list(range(min(self.top_k_results, len(self.document_embeddings))))
        
        if top_k is None:
            top_k = self.top_k_results
        
        try:
            # Ensure query embedding has correct dimension
            if query_embedding.shape[0] != self.attention_dim:
                logger.warning(f"Query embedding dimension mismatch: {query_embedding.shape[0]} vs {self.attention_dim}")
                return list(range(min(top_k, len(self.document_embeddings))))
            
            # Calculate similarity scores using neural retriever
            query_expanded = query_embedding.unsqueeze(0).expand(self.document_embeddings.shape[0], -1)
            combined_features = torch.cat([query_expanded, self.document_embeddings], dim=1)
            
            with torch.no_grad():
                scores = self.neural_retriever(combined_features)
                if scores.dim() > 1:
                    scores = scores.squeeze()
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
            
            # Get top-k indices
            _, top_indices = torch.topk(scores, min(top_k, len(scores)))
            return top_indices.tolist()
            
        except Exception as e:
            logger.warning(f"Neural retrieval failed: {e}")
            return list(range(min(top_k, len(self.document_embeddings))))
    
    def _neural_retrieve_on_chunks(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor, top_k: int = None) -> List[int]:
        """Neural retrieval to find most relevant chunks from retrieved chunks."""
        if not self.use_hybrid_attention:
            return list(range(min(top_k, len(doc_embeddings))))
        
        if top_k is None:
            top_k = self.top_k_results
        
        try:
            # Ensure query embedding has correct dimension
            if query_embedding.shape[0] != self.attention_dim:
                logger.warning(f"Query embedding dimension mismatch: {query_embedding.shape[0]} vs {self.attention_dim}")
                return list(range(min(top_k, len(doc_embeddings))))
            
            # Calculate similarity scores using neural retriever
            query_expanded = query_embedding.unsqueeze(0).expand(doc_embeddings.shape[0], -1)
            combined_features = torch.cat([query_expanded, doc_embeddings], dim=1)
            
            with torch.no_grad():
                scores = self.neural_retriever(combined_features)
                if scores.dim() > 1:
                    scores = scores.squeeze()
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
            
            # Get top-k indices
            _, top_indices = torch.topk(scores, min(top_k, len(scores)))
            return top_indices.tolist()
            
        except Exception as e:
            logger.warning(f"Neural retrieval on chunks failed: {e}")
            return list(range(min(top_k, len(doc_embeddings))))
    
    def _apply_attention(self, query_embedding: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to focus on relevant parts."""
        if not self.use_hybrid_attention:
            return doc_embeddings
        
        try:
            # Ensure dimensions match
            if query_embedding.shape[0] != self.attention_dim:
                logger.warning(f"Query embedding dimension mismatch in attention: {query_embedding.shape[0]} vs {self.attention_dim}")
                return doc_embeddings
            
            if doc_embeddings.shape[1] != self.attention_dim:
                logger.warning(f"Document embedding dimension mismatch in attention: {doc_embeddings.shape[1]} vs {self.attention_dim}")
                return doc_embeddings
            
            # Prepare inputs for attention
            query = query_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, attention_dim]
            key_value = doc_embeddings.unsqueeze(0)  # [1, num_docs, attention_dim]
            
            # Apply multi-head attention
            with torch.no_grad():
                attended_output, attention_weights = self.attention_layer(
                    query, key_value, key_value
                )
            
            return attended_output.squeeze(0)  # [num_docs, attention_dim]
            
        except Exception as e:
            logger.warning(f"Attention mechanism failed: {e}")
            return doc_embeddings
    
    def generate_response(self, question: str) -> Dict[str, Any]:
        """
        Generate a response using hybrid RAG.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant chunks using similarity search
            print(f"🔍 Retrieving relevant chunks for: {question[:50]}...")
            docs = self.vectorstore.similarity_search(
                question, 
                k=self.top_k_results
            )
            
            print(f"  📊 Retrieved {len(docs)} chunks from similarity search")
            
            if not docs:
                print("  ⚠️  No documents retrieved from similarity search")
                return {
                    'response': "I couldn't find relevant information to answer your question.",
                    'context': '',
                    'retrieved_docs': 0,
                    'context_length': 0,
                    'context_tokens': 0,
                    'response_time': time.time() - start_time,
                    'method': 'narrativeqa_hybrid_rag'
                }
            
            # Step 2: Get query embedding
            query_embedding = torch.tensor(
                self.embeddings.embed_query(question), 
                dtype=torch.float32
            )
            
            # Step 3: Neural retrieval (if hybrid attention is enabled)
            if self.use_hybrid_attention and self.document_embeddings is not None:
                # Get document embeddings for retrieved chunks
                doc_contents = [doc.page_content for doc in docs]
                doc_embeddings = torch.tensor(
                    self.embeddings.embed_documents(doc_contents),
                    dtype=torch.float32
                )
                
                # Apply neural retrieval to re-rank chunks (only on retrieved chunks)
                neural_indices = self._neural_retrieve_on_chunks(query_embedding, doc_embeddings, len(docs))
                neural_retrieved_docs = [docs[i] for i in neural_indices if i < len(docs)]
                
                # Apply attention mechanism
                attended_embeddings = self._apply_attention(query_embedding, doc_embeddings)
                
                # Use neural retrieval results
                docs = neural_retrieved_docs
            else:
                attended_embeddings = None
            
            # Step 4: Combine retrieved chunks
            context_parts = []
            for i, doc in enumerate(docs):
                if attended_embeddings is not None and i < attended_embeddings.shape[0]:
                    # Use attention weights if available
                    attention_weight = torch.softmax(attended_embeddings[i], dim=0).mean().item()
                    context_parts.append(f"[Attention: {attention_weight:.3f}] {doc.page_content}")
                else:
                    context_parts.append(f"Chunk {i+1}: {doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Step 5: Check context length
            context_tokens = len(self.tokenizer.encode(context))
            if context_tokens > self.max_context_tokens:
                # Truncate context if too long
                words = context.split()
                max_words = int(self.max_context_tokens * 0.8)  # Leave room for question
                context = " ".join(words[:max_words]) + "..."
                context_tokens = len(self.tokenizer.encode(context))
            
            # Step 6: Create prompt
            prompt = f"""Based on the following story excerpts, answer the question.

Context:
{context}

Question: {question}

Please provide a concise, direct answer (1-2 sentences maximum). Focus on the key facts from the story excerpts. If the excerpts don't contain enough information to answer the question, please say so briefly."""

            # Step 7: Generate response
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
                'method': 'narrativeqa_hybrid_rag',
                'hybrid_attention_used': self.use_hybrid_attention,
                'neural_retrieval_used': attended_embeddings is not None
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
                'method': 'narrativeqa_hybrid_rag',
                'error': str(e)
            }

def main():
    """Main function for testing the NarrativeQA Hybrid RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NarrativeQA Hybrid RAG")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--db-path", type=str, default="./narrativeqa_hybrid_vectordb", help="Path to vector database")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid attention mechanisms")
    
    args = parser.parse_args()
    
    # Initialize Hybrid RAG system
    print("🚀 Initializing NarrativeQA Hybrid RAG...")
    hybrid_rag = NarrativeQAHybridRAG(
        chunk_size=args.chunk_size,
        top_k_results=args.top_k,
        db_path=args.db_path,
        use_hybrid_attention=not args.no_hybrid
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
            print(f"  Response time: {result['response_time']:.2f}s")
            print(f"  Hybrid attention: {result.get('hybrid_attention_used', False)}")
            print(f"  Neural retrieval: {result.get('neural_retrieval_used', False)}")
    
    elif args.query:
        print(f"\n🤖 Processing: {args.query}")
        result = hybrid_rag.generate_response(args.query)
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Retrieved docs: {result['retrieved_docs']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")
        print(f"  Hybrid attention: {result.get('hybrid_attention_used', False)}")
        print(f"  Neural retrieval: {result.get('neural_retrieval_used', False)}")
    
    else:
        # Default test query
        test_query = "Who is the main character in the story?"
        print(f"\n🤖 Testing with: {test_query}")
        result = hybrid_rag.generate_response(test_query)
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Retrieved docs: {result['retrieved_docs']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")
        print(f"  Hybrid attention: {result.get('hybrid_attention_used', False)}")
        print(f"  Neural retrieval: {result.get('neural_retrieval_used', False)}")

if __name__ == "__main__":
    main()
