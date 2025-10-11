"""
Integration of Hybrid Attention RAG with Existing RAG Framework

This module integrates the novel hybrid attention mechanism with the existing
Long Context RAG system, providing a seamless interface for research and experimentation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from core.index import LongContextRAG
from .hybrid_attention_rag import HybridAttentionRAG, AttentionConfig
from training.neural_retriever import NeuralRetriever, DynamicQueryGenerator, RetrieverConfig
from core.prompts import RAGPrompts

logger = logging.getLogger(__name__)

class HybridRAGIntegration:
    """
    Integration class that combines the existing RAG system with the hybrid attention mechanism.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid RAG integration.
        
        Args:
            config: Configuration dictionary for the hybrid system
        """
        # Default configuration
        self.config = config or self._get_default_config()
        
        # Initialize components
        self._initialize_components()
        
        # Load existing RAG system
        self.base_rag = LongContextRAG()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the hybrid system."""
        return {
            'attention': {
                'window_size': 512,
                'num_landmark_tokens': 32,
                'max_retrieved_segments': 8,
                'hidden_size': 3072,  # Match text-embedding-3-large dimensions
                'num_attention_heads': 12
            },
            'retriever': {
                'query_embed_dim': 3072,  # Match text-embedding-3-large dimensions
                'doc_embed_dim': 3072,    # Match text-embedding-3-large dimensions
                'hidden_dim': 1024,
                'num_candidates': 100,
                'top_k': 8
            },
            'integration': {
                'use_hybrid_attention': True,
                'use_neural_retriever': True,
                'use_dynamic_queries': True,
                'attention_weight': 0.3,
                'retrieval_weight': 0.4,
                'generation_weight': 0.3
            }
        }
    
    def _initialize_components(self):
        """Initialize the hybrid attention components."""
        # Attention configuration
        attention_config = AttentionConfig(**self.config['attention'])
        
        # Retriever configuration
        retriever_config = RetrieverConfig(**self.config['retriever'])
        
        # Initialize models
        self.hybrid_attention = HybridAttentionRAG(attention_config)
        self.neural_retriever = NeuralRetriever(retriever_config)
        self.query_generator = DynamicQueryGenerator(retriever_config)
        
        # Set to evaluation mode
        self.hybrid_attention.eval()
        self.neural_retriever.eval()
        self.query_generator.eval()
        
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """Load documents using the base RAG system."""
        return self.base_rag.load_documents(file_paths)
    
    def create_vectorstore(self, documents: List[Any], persist: bool = True):
        """Create vector store using the base RAG system."""
        self.base_rag.create_vectorstore(documents, persist)
        
        # Prepare document embeddings for neural retriever
        self._prepare_document_embeddings(documents)
    
    def _prepare_document_embeddings(self, documents: List[Any]):
        """Prepare document embeddings for the neural retriever."""
        # Extract document content and create embeddings
        doc_contents = []
        for doc in documents:
            doc_contents.append(doc.page_content)
        
        # Create embeddings using the base RAG system's embedding model
        embeddings = self.base_rag.embeddings.embed_documents(doc_contents)
        
        # Convert to tensor format for neural retriever
        self.document_embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        logger.info(f"Prepared {len(embeddings)} document embeddings for neural retriever")
    
    def generate_response(self, query: str, use_hybrid: bool = True, 
                         task_type: str = 'generation') -> Dict[str, Any]:
        """
        Generate response using the hybrid attention RAG system.
        
        Args:
            query: The input query
            use_hybrid: Whether to use hybrid attention mechanism
            task_type: Type of task (generation, qa, classification, etc.)
            
        Returns:
            Dictionary containing response and metadata
        """
        if use_hybrid and self.config['integration']['use_hybrid_attention']:
            return self._generate_hybrid_response(query, task_type)
        else:
            # Fall back to base RAG system
            return self.base_rag.generate_response(query, use_rag=True)
    
    def _generate_hybrid_response(self, query: str, task_type: str) -> Dict[str, Any]:
        """Generate response using the hybrid attention mechanism."""
        try:
            # Step 1: Retrieve relevant documents using base RAG
            retrieved_docs = self.base_rag.retrieve_relevant_docs(query)
            
            # Step 2: Prepare embeddings
            query_embedding = self._get_query_embedding(query)
            doc_embeddings = self._get_document_embeddings(retrieved_docs)
            
            # Step 3: Dynamic query generation
            if self.config['integration']['use_dynamic_queries']:
                # Extract processing state from query embedding
                processing_state = query_embedding.squeeze(0).squeeze(0)  # [3072]
                dynamic_query = self.query_generator(
                    query_embedding,  # Already has correct shape [1, 1, 3072]
                    processing_state, 
                    task_type
                )
            else:
                dynamic_query = query_embedding
            
            # Step 4: Neural retrieval
            if self.config['integration']['use_neural_retriever']:
                retrieval_scores, neural_retrieved = self.neural_retriever(
                    dynamic_query, 
                    self.document_embeddings
                )
            else:
                neural_retrieved = None
                retrieval_scores = None
            
            # Step 5: Hybrid attention processing
            if neural_retrieved is not None:
                # Convert neural retrieved docs to list format
                retrieved_segments = [neural_retrieved[0, i] for i in range(neural_retrieved.shape[1])]
            else:
                retrieved_segments = None
            
            # Process with hybrid attention
            attention_output = self.hybrid_attention(
                query_embedding,  # Already has correct shape [1, 1, 3072]
                retrieved_segments
            )
            
            # Step 6: Generate final response using LLM
            context = self._prepare_context_for_llm(attention_output, retrieved_docs)
            response = self._generate_llm_response(query, context)
            
            return {
                'response': response,
                'method': 'hybrid_attention_rag',
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
        """Get query embedding."""
        embedding = self.base_rag.embeddings.embed_query(query)
        
        # Convert to tensor and ensure proper shape for hybrid attention
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
        # Reshape to [1, 1, embed_dim] for hybrid attention compatibility
        # This creates a sequence of length 1 with the embedding
        return embedding_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 3072]
    
    def _get_document_embeddings(self, documents: List[Any]) -> torch.Tensor:
        """Get document embeddings for retrieved documents."""
        doc_contents = [doc.page_content for doc in documents]
        embeddings = self.base_rag.embeddings.embed_documents(doc_contents)
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def _prepare_context_for_llm(self, attention_output: torch.Tensor, 
                                retrieved_docs: List[Any]) -> str:
        """Prepare context for LLM generation."""
        # Use attention output to weight retrieved documents
        # attention_output shape: [1, 1, 3072] or [1, seq_len, 3072]
        if attention_output.shape[1] > 1:
            attention_weights = torch.softmax(attention_output.mean(dim=-1), dim=-1)
        else:
            # For single token sequences, create uniform weights
            attention_weights = torch.ones(1, len(retrieved_docs)) / len(retrieved_docs)
        
        # Weight documents by attention
        weighted_contexts = []
        for i, doc in enumerate(retrieved_docs):
            weight = attention_weights[0, i].item() if i < attention_weights.shape[1] else 1.0 / len(retrieved_docs)
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
    
    def compare_methods(self, query: str, task_type: str = 'generation') -> Dict[str, Any]:
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
        
        # Hybrid Attention RAG
        try:
            hybrid_result = self.generate_response(query, use_hybrid=True, task_type=task_type)
            results['hybrid_rag'] = {
                'response': hybrid_result['response'],
                'retrieved_docs': hybrid_result['retrieved_docs'],
                'context_length': hybrid_result['context_length'],
                'neural_scores': hybrid_result.get('neural_retrieval_scores'),
                'attention_shape': hybrid_result.get('attention_output_shape')
            }
        except Exception as e:
            results['hybrid_rag'] = {'error': str(e)}
        
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
    
    def analyze_attention_patterns(self, query: str) -> Dict[str, Any]:
        """
        Analyze attention patterns in the hybrid system.
        
        Args:
            query: The input query
            
        Returns:
            Dictionary containing attention analysis
        """
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Retrieve documents
            retrieved_docs = self.base_rag.retrieve_relevant_docs(query)
            
            # Process with hybrid attention
            attention_output = self.hybrid_attention(
                query_embedding,  # Already has correct shape [1, 1, 3072]
                None  # No retrieved segments for analysis
            )
            
            # Analyze attention patterns
            attention_analysis = {
                'query_length': query_embedding.shape[1],  # Should be 1
                'attention_output_shape': attention_output.shape,
                'attention_mean': attention_output.mean().item(),
                'attention_std': attention_output.std().item(),
                'attention_max': attention_output.max().item(),
                'attention_min': attention_output.min().item(),
                'retrieved_docs_count': len(retrieved_docs)
            }
            
            return attention_analysis
            
        except Exception as e:
            logger.error(f"Error in attention analysis: {e}")
            return {'error': str(e)}

def create_hybrid_rag_system(config: Optional[Dict[str, Any]] = None) -> HybridRAGIntegration:
    """Create and initialize the hybrid RAG system."""
    return HybridRAGIntegration(config)

# Example usage and testing
def test_hybrid_integration():
    """Test the hybrid RAG integration."""
    print("🧪 Testing Hybrid RAG Integration")
    print("=" * 50)
    
    # Create hybrid system
    hybrid_rag = create_hybrid_rag_system()
    
    # Load sample documents
    sample_file = "data/sample_documents.txt"
    if Path(sample_file).exists():
        documents = hybrid_rag.load_documents([sample_file])
        hybrid_rag.create_vectorstore(documents)
        print(f"✅ Loaded {len(documents)} documents")
    else:
        print("⚠️  Sample documents not found, using base RAG system")
    
    # Test queries
    test_queries = [
        "What is machine learning and what are its main types?",
        "How do transformer models work in NLP?",
        "What are the benefits of RAG systems?"
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
        
        # Analyze attention patterns
        attention_analysis = hybrid_rag.analyze_attention_patterns(query)
        print(f"\nATTENTION ANALYSIS:")
        for key, value in attention_analysis.items():
            print(f"  {key}: {value}")
    
    print("\n✅ Hybrid RAG integration test completed!")

if __name__ == "__main__":
    test_hybrid_integration()
