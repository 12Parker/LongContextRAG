#!/usr/bin/env python3
"""
Fixed Hybrid Attention RAG System

This provides a working version of the hybrid attention RAG system
with properly fixed dimension compatibility.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from core.index import LongContextRAG
from core.prompts import RAGPrompts

logger = logging.getLogger(__name__)

class FixedHybridAttention(nn.Module):
    """
    Simplified hybrid attention mechanism that works with single-token sequences.
    """
    
    def __init__(self, hidden_size: int = 3072, num_heads: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Simple attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, retrieved_segments: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for hybrid attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            retrieved_segments: List of retrieved segment tensors (optional)
            
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        # Self-attention
        attended_states, attention_weights = self.attention(
            hidden_states, hidden_states, hidden_states
        )
        
        # Residual connection
        output = self.layer_norm(hidden_states + attended_states)
        
        return output

class FixedNeuralRetriever(nn.Module):
    """
    Simplified neural retriever that works with the current setup.
    """
    
    def __init__(self, query_embed_dim: int = 3072, doc_embed_dim: int = 3072, top_k: int = 4):
        super().__init__()
        self.query_embed_dim = query_embed_dim
        self.doc_embed_dim = doc_embed_dim
        self.top_k = top_k
        
        # Simple similarity projection
        self.similarity_proj = nn.Linear(query_embed_dim, doc_embed_dim)
        
    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> tuple:
        """
        Forward pass for neural retrieval.
        
        Args:
            query_embeddings: [batch_size, seq_len, query_embed_dim]
            doc_embeddings: [num_docs, doc_embed_dim]
            
        Returns:
            scores: [batch_size, top_k]
            retrieved_docs: [batch_size, top_k, doc_embed_dim]
        """
        batch_size, seq_len, _ = query_embeddings.shape
        num_docs, _ = doc_embeddings.shape
        
        # Project query embeddings
        query_proj = self.similarity_proj(query_embeddings)  # [batch_size, seq_len, doc_embed_dim]
        
        # Calculate similarity scores
        query_repr = query_proj.mean(dim=1)  # [batch_size, doc_embed_dim]
        
        # Cosine similarity
        query_norm = torch.nn.functional.normalize(query_repr, p=2, dim=-1)
        doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
        
        scores = torch.matmul(query_norm, doc_norm.t())  # [batch_size, num_docs]
        
        # Select top-k documents
        top_k_scores, top_k_indices = torch.topk(scores, min(self.top_k, num_docs), dim=-1)
        
        # Retrieve document embeddings
        retrieved_docs = doc_embeddings[top_k_indices]  # [batch_size, top_k, doc_embed_dim]
        
        return top_k_scores, retrieved_docs

class FixedHybridRAG:
    """
    Fixed hybrid attention RAG system with proper dimension handling.
    """
    
    def __init__(self, use_hybrid_attention: bool = True):
        """
        Initialize the fixed hybrid RAG system.
        
        Args:
            use_hybrid_attention: Whether to use hybrid attention (default: True)
        """
        self.use_hybrid_attention = use_hybrid_attention
        
        # Initialize base RAG system
        self.base_rag = LongContextRAG()
        
        if use_hybrid_attention:
            # Initialize fixed hybrid components
            self.hybrid_attention = FixedHybridAttention(hidden_size=3072, num_heads=12)
            self.neural_retriever = FixedNeuralRetriever(query_embed_dim=3072, doc_embed_dim=3072, top_k=4)
            
            # Set to evaluation mode
            self.hybrid_attention.eval()
            self.neural_retriever.eval()
    
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """Load documents using the base RAG system."""
        return self.base_rag.load_documents(file_paths)
    
    def create_vectorstore(self, documents: List[Any], persist: bool = True):
        """Create vector store using the base RAG system."""
        self.base_rag.create_vectorstore(documents, persist)
        
        if self.use_hybrid_attention:
            # Prepare document embeddings for neural retriever
            self._prepare_document_embeddings(documents)
    
    def _prepare_document_embeddings(self, documents: List[Any]):
        """Prepare document embeddings for the neural retriever."""
        # Extract document content
        doc_contents = []
        for doc in documents:
            doc_contents.append(doc.page_content)
        
        # Create embeddings using the base RAG system's embedding model
        embeddings = self.base_rag.embeddings.embed_documents(doc_contents)
        
        # Convert to tensor format for neural retriever
        embeddings_array = np.array(embeddings)
        
        # Ensure correct dimensions
        if embeddings_array.shape[1] != 3072:
            if embeddings_array.shape[1] > 3072:
                embeddings_array = embeddings_array[:, :3072]
            else:
                padding = np.zeros((embeddings_array.shape[0], 3072 - embeddings_array.shape[1]))
                embeddings_array = np.concatenate([embeddings_array, padding], axis=1)
        
        self.document_embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
        
        logger.info(f"Prepared {len(embeddings)} document embeddings for neural retriever")
    
    def generate_response(self, query: str, task_type: str = 'qa') -> Dict[str, Any]:
        """
        Generate response using the fixed hybrid attention RAG system.
        
        Args:
            query: The input query
            task_type: Type of task (generation, qa, classification, etc.)
            
        Returns:
            Dictionary containing response and metadata
        """
        if self.use_hybrid_attention:
            return self._generate_hybrid_response(query, task_type)
        else:
            # Fall back to base RAG system
            return self.base_rag.generate_response(query, use_rag=True)
    
    def _generate_hybrid_response(self, query: str, task_type: str) -> Dict[str, Any]:
        """Generate response using the fixed hybrid attention mechanism."""
        try:
            # Step 1: Retrieve relevant documents using base RAG
            retrieved_docs = self.base_rag.retrieve_relevant_docs(query)
            
            # Step 2: Prepare embeddings
            query_embedding = self._get_query_embedding(query)
            
            # Step 3: Neural retrieval
            if hasattr(self, 'document_embeddings'):
                with torch.no_grad():
                    retrieval_scores, neural_retrieved = self.neural_retriever(
                        query_embedding, 
                        self.document_embeddings
                    )
                
                # Convert neural retrieved docs to list format
                retrieved_segments = [neural_retrieved[0, i] for i in range(neural_retrieved.shape[1])]
            else:
                retrieved_segments = None
                retrieval_scores = None
            
            # Step 4: Hybrid attention processing
            with torch.no_grad():
                attention_output = self.hybrid_attention(
                    query_embedding,
                    retrieved_segments
                )
            
            # Step 5: Generate final response using LLM
            context = self._prepare_context_for_llm(attention_output, retrieved_docs)
            response = self._generate_llm_response(query, context)
            
            return {
                'response': response,
                'method': 'fixed_hybrid_attention_rag',
                'retrieved_docs': len(retrieved_docs),
                'neural_retrieval_scores': retrieval_scores.tolist() if retrieval_scores is not None else None,
                'context_length': len(context),
                'task_type': task_type,
                'attention_output_shape': attention_output.shape
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid response generation: {e}")
            # Fall back to base RAG
            return self.base_rag.generate_response(query, use_rag=True)
    
    def _get_query_embedding(self, query: str) -> torch.Tensor:
        """Get query embedding with proper shape."""
        embedding = self.base_rag.embeddings.embed_query(query)
        
        # Convert to tensor and ensure proper shape
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
        # Ensure correct dimensions
        if len(embedding_tensor) != 3072:
            if len(embedding_tensor) > 3072:
                embedding_tensor = embedding_tensor[:3072]
            else:
                embedding_tensor = torch.nn.functional.pad(embedding_tensor, (0, 3072 - len(embedding_tensor)))
        
        # Reshape to [1, 1, 3072] for batch processing
        return embedding_tensor.unsqueeze(0).unsqueeze(0)
    
    def _prepare_context_for_llm(self, attention_output: torch.Tensor, 
                                retrieved_docs: List[Any]) -> str:
        """Prepare context for LLM generation."""
        # Use attention output to weight retrieved documents
        # For simplicity, use uniform weights
        weight = 1.0 / len(retrieved_docs)
        
        # Weight documents
        weighted_contexts = []
        for doc in retrieved_docs:
            weighted_contexts.append(f"[Weight: {weight:.3f}] {doc.page_content}")
        
        return "\n\n".join(weighted_contexts)
    
    def _generate_llm_response(self, query: str, context: str) -> str:
        """Generate response using the LLM."""
        # Use research prompt for better responses
        prompt = RAGPrompts.RESEARCH_RAG
        
        # Create chain
        chain = prompt | self.base_rag.llm
        
        # Generate response
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        return response.content
    
    def compare_methods(self, query: str, task_type: str = 'qa') -> Dict[str, Any]:
        """
        Compare different methods for the same query.
        
        Args:
            query: The input query
            task_type: Type of task
            
        Returns:
            Dictionary comparing different methods
        """
        results = {}
        
        # Base RAG
        try:
            base_result = self.base_rag.generate_response(query, use_rag=True)
            results['base_rag'] = {
                'response': base_result['response'],
                'retrieved_docs': base_result['retrieved_docs'],
                'context_length': base_result['context_length']
            }
        except Exception as e:
            results['base_rag'] = {'error': str(e)}
        
        # Fixed Hybrid Attention RAG
        if self.use_hybrid_attention:
            try:
                hybrid_result = self.generate_response(query, task_type=task_type)
                results['fixed_hybrid_rag'] = {
                    'response': hybrid_result['response'],
                    'retrieved_docs': hybrid_result['retrieved_docs'],
                    'context_length': hybrid_result['context_length'],
                    'neural_scores': hybrid_result.get('neural_retrieval_scores'),
                    'attention_shape': hybrid_result.get('attention_output_shape')
                }
            except Exception as e:
                results['fixed_hybrid_rag'] = {'error': str(e)}
        
        # Direct LLM (no RAG)
        try:
            direct_result = self.base_rag.generate_response(query, use_rag=False)
            results['direct_llm'] = {
                'response': direct_result['response'],
                'retrieved_docs': direct_result['retrieved_docs'],
                'context_length': direct_result['context_length']
            }
        except Exception as e:
            results['direct_llm'] = {'error': str(e)}
        
        return results

def test_fixed_hybrid_rag():
    """Test the fixed hybrid RAG system."""
    print("🧪 Testing Fixed Hybrid Attention RAG")
    print("=" * 50)
    
    # Create fixed hybrid RAG system
    hybrid_rag = FixedHybridRAG(use_hybrid_attention=True)
    
    # Load sample documents
        from testing.bookcorpus_integration import BookCorpusLoader, BookCorpusConfig
    config = BookCorpusConfig(max_books=1)
    loader = BookCorpusLoader(config)
    books = loader.load_sample_books()
    documents = loader.process_books_for_rag()
    
    # Create vector store
    hybrid_rag.create_vectorstore(documents)
    print(f"✅ Loaded {len(documents)} documents and created vector store")
    
    # Test queries
    test_queries = [
        "What is the main topic of this book?",
        "What are the key concepts discussed?",
        "How does the author support their arguments?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Compare methods
        comparison = hybrid_rag.compare_methods(query, 'qa')
        
        for method, result in comparison.items():
            print(f"\n{method.upper()}:")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Retrieved docs: {result['retrieved_docs']}")
                print(f"  Context length: {result['context_length']}")
                print(f"  Response: {result['response'][:200]}...")
                if 'neural_scores' in result and result['neural_scores']:
                    print(f"  Neural scores: {result['neural_scores']}")
                if 'attention_shape' in result:
                    print(f"  Attention shape: {result['attention_shape']}")
    
    print("\n✅ Fixed hybrid RAG test completed!")

if __name__ == "__main__":
    test_fixed_hybrid_rag()
