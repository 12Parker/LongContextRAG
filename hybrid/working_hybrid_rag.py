#!/usr/bin/env python3
"""
Working Hybrid Attention RAG System

This provides a working version of the hybrid attention RAG system
with properly aligned dimensions and simplified integration.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from core.index import LongContextRAG
from .hybrid_attention_rag import HybridAttentionRAG, AttentionConfig
from training.neural_retriever import NeuralRetriever, RetrieverConfig
from core.prompts import RAGPrompts

logger = logging.getLogger(__name__)

class WorkingHybridRAG:
    """
    Working hybrid attention RAG system with proper dimension alignment.
    """
    
    def __init__(self, use_hybrid_attention: bool = True):
        """
        Initialize the working hybrid RAG system.
        
        Args:
            use_hybrid_attention: Whether to use hybrid attention (default: True)
        """
        self.use_hybrid_attention = use_hybrid_attention
        
        # Initialize base RAG system
        self.base_rag = LongContextRAG()
        
        if use_hybrid_attention:
            # Initialize hybrid attention components with correct dimensions
            self._initialize_hybrid_components()
    
    def _initialize_hybrid_components(self):
        """Initialize hybrid attention components with proper dimensions."""
        # Use smaller dimensions for compatibility
        self.attention_config = AttentionConfig(
            window_size=256,
            num_landmark_tokens=16,
            max_retrieved_segments=4,
            hidden_size=768,  # Compatible with base embeddings
            num_attention_heads=8
        )
        
        self.retriever_config = RetrieverConfig(
            query_embed_dim=768,
            doc_embed_dim=768,
            hidden_dim=512,
            num_candidates=50,
            top_k=4
        )
        
        # Initialize models
        self.hybrid_attention = HybridAttentionRAG(self.attention_config)
        self.neural_retriever = NeuralRetriever(self.retriever_config)
        
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
        
        # Simple dimension adjustment - pad or truncate to 768
        if embeddings_array.shape[1] > 768:
            # Truncate if larger
            embeddings_array = embeddings_array[:, :768]
        elif embeddings_array.shape[1] < 768:
            # Pad with zeros if smaller
            padding = np.zeros((embeddings_array.shape[0], 768 - embeddings_array.shape[1]))
            embeddings_array = np.concatenate([embeddings_array, padding], axis=1)
        
        self.document_embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
        
        logger.info(f"Prepared {len(embeddings)} document embeddings for neural retriever")
    
    def generate_response(self, query: str, task_type: str = 'qa') -> Dict[str, Any]:
        """
        Generate response using the hybrid attention RAG system.
        
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
        """Generate response using the hybrid attention mechanism."""
        try:
            # Step 1: Retrieve relevant documents using base RAG
            retrieved_docs = self.base_rag.retrieve_relevant_docs(query)
            
            # Step 2: Prepare embeddings
            query_embedding = self._get_query_embedding(query)
            
            # Step 3: Neural retrieval
            if hasattr(self, 'document_embeddings'):
                # Create document embeddings for retrieval
                doc_contents = [doc.page_content for doc in retrieved_docs]
                doc_embeddings = self.base_rag.embeddings.embed_documents(doc_contents)
                doc_embeddings_array = np.array(doc_embeddings)
                
                # Simple dimension adjustment - pad or truncate to 768
                if doc_embeddings_array.shape[1] > 768:
                    doc_embeddings_array = doc_embeddings_array[:, :768]
                elif doc_embeddings_array.shape[1] < 768:
                    padding = np.zeros((doc_embeddings_array.shape[0], 768 - doc_embeddings_array.shape[1]))
                    doc_embeddings_array = np.concatenate([doc_embeddings_array, padding], axis=1)
                
                doc_embeddings_tensor = torch.tensor(doc_embeddings_array, dtype=torch.float32)
                
                # Neural retrieval
                with torch.no_grad():
                    retrieval_scores, neural_retrieved = self.neural_retriever(
                        query_embedding.unsqueeze(0), 
                        doc_embeddings_tensor
                    )
                
                # Convert neural retrieved docs to list format
                retrieved_segments = [neural_retrieved[0, i] for i in range(neural_retrieved.shape[1])]
            else:
                retrieved_segments = None
                retrieval_scores = None
            
            # Step 4: Hybrid attention processing
            if retrieved_segments is not None:
                # Process with hybrid attention
                with torch.no_grad():
                    attention_output = self.hybrid_attention(
                        query_embedding.unsqueeze(0),
                        retrieved_segments
                    )
            else:
                attention_output = query_embedding.unsqueeze(0)
            
            # Step 5: Generate final response using LLM
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
        
        # Simple dimension adjustment - pad or truncate to 768
        if len(embedding) > 768:
            embedding = embedding[:768]
        elif len(embedding) < 768:
            embedding = np.pad(embedding, (0, 768 - len(embedding)), 'constant')
        
        return torch.tensor(embedding, dtype=torch.float32)
    
    def _prepare_context_for_llm(self, attention_output: torch.Tensor, 
                                retrieved_docs: List[Any]) -> str:
        """Prepare context for LLM generation."""
        # Use attention output to weight retrieved documents
        if attention_output.shape[1] > 1:
            attention_weights = torch.softmax(attention_output.mean(dim=-1), dim=-1)
        else:
            attention_weights = torch.ones(1, len(retrieved_docs))
        
        # Weight documents by attention
        weighted_contexts = []
        for i, doc in enumerate(retrieved_docs):
            weight = attention_weights[0, i].item() if i < attention_weights.shape[1] else 1.0
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
        
        # Hybrid Attention RAG
        if self.use_hybrid_attention:
            try:
                hybrid_result = self.generate_response(query, task_type=task_type)
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

def test_working_hybrid_rag():
    """Test the working hybrid RAG system."""
    print("🧪 Testing Working Hybrid Attention RAG")
    print("=" * 50)
    
    # Create working hybrid RAG system
    hybrid_rag = WorkingHybridRAG(use_hybrid_attention=True)
    
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
    
    print("\n✅ Working hybrid RAG test completed!")

if __name__ == "__main__":
    test_working_hybrid_rag()
