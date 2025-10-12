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
import sys
import os

# Add VectorDB to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VectorDB'))

from core.index import LongContextRAG
from .hybrid_attention_rag import HybridAttentionRAG, AttentionConfig
from training.neural_retriever import NeuralRetriever, RetrieverConfig
from core.prompts import RAGPrompts
from build_db import VectorDBBuilder

logger = logging.getLogger(__name__)

class WorkingHybridRAG:
    """
    Working hybrid attention RAG system with proper dimension alignment.
    """
    
    def __init__(self, use_hybrid_attention: bool = True, vectordb_config: Optional[Dict[str, Any]] = None, 
                 max_retrieved_docs: int = 5):
        """
        Initialize the working hybrid RAG system.
        
        Args:
            use_hybrid_attention: Whether to use hybrid attention (default: True)
            vectordb_config: Configuration for VectorDBBuilder (optional)
            max_retrieved_docs: Maximum number of documents to retrieve (default: 5)
        """
        self.use_hybrid_attention = use_hybrid_attention
        self.max_retrieved_docs = max_retrieved_docs
        
        # Initialize base RAG system
        self.base_rag = LongContextRAG()
        
        # Initialize VectorDBBuilder with default or provided config
        default_vectordb_config = {
            'db_path': './vector_store',
            'collection_name': 'hybrid_rag',
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'batch_size': 100
        }
        
        if vectordb_config:
            default_vectordb_config.update(vectordb_config)
        
        self.vectordb_builder = VectorDBBuilder(**default_vectordb_config)
        self.vectordb_initialized = False
        
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
    
    def create_vectorstore(self, documents: List[Any] = None, persist: bool = True, 
                          use_vectordb: bool = True, num_documents: int = 5000):
        """
        Create vector store using VectorDBBuilder or base RAG system.
        
        Args:
            documents: List of documents (optional if using VectorDBBuilder)
            persist: Whether to persist the vector store
            use_vectordb: Whether to use VectorDBBuilder for BookCorpus dataset
            num_documents: Number of documents to process (for VectorDBBuilder)
        """
        if use_vectordb:
            # Use VectorDBBuilder for BookCorpus dataset
            logger.info("Creating vector store using VectorDBBuilder...")
            self.vectordb_builder.create_or_reset_collection(reset=True)
            self.vectordb_builder.process_dataset(
                dataset_name="rojagtap/bookcorpus",
                num_documents=num_documents,
                min_text_length=50  # Lower minimum length to process more documents
            )
            self.vectordb_initialized = True
            logger.info("VectorDBBuilder vector store created successfully")
            
            # Initialize base RAG retriever to use VectorDBBuilder
            self._setup_base_rag_with_vectordb()
            
            # Also create base RAG vector store for compatibility if documents provided
            if documents:
                self.base_rag.create_vectorstore(documents, persist)
        else:
            # Use base RAG system
            if not documents:
                raise ValueError("Documents required when not using VectorDBBuilder")
            self.base_rag.create_vectorstore(documents, persist)
        
        if self.use_hybrid_attention and documents:
            # Prepare document embeddings for neural retriever
            self._prepare_document_embeddings(documents)
    
    def _setup_base_rag_with_vectordb(self):
        """Setup base RAG to use VectorDBBuilder for retrieval."""
        # Create a custom retriever that uses VectorDBBuilder
        class VectorDBRetriever:
            def __init__(self, vectordb_builder, n_results=5):
                self.vectordb_builder = vectordb_builder
                self.n_results = n_results
            
            def invoke(self, query: str):
                """Retrieve documents using VectorDBBuilder."""
                results = self.vectordb_builder.query(query, n_results=self.n_results)
                
                # Convert to Document-like objects
                documents = []
                for doc_text, distance, metadata in zip(
                    results['documents'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                ):
                    doc = type('Document', (), {
                        'page_content': doc_text,
                        'metadata': metadata
                    })()
                    documents.append(doc)
                
                return documents
        
        # Set the retriever with configurable number of results
        self.base_rag.retriever = VectorDBRetriever(self.vectordb_builder, n_results=self.max_retrieved_docs)
        logger.info(f"Base RAG configured to use VectorDBBuilder with {self.max_retrieved_docs} documents")
    
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
    
    def retrieve_relevant_docs(self, query: str, n_results: Optional[int] = None) -> List[Any]:
        """
        Retrieve relevant documents using VectorDBBuilder or base RAG.
        
        Args:
            query: The search query
            n_results: Number of results to return (defaults to max_retrieved_docs)
            
        Returns:
            List of relevant documents
        """
        if n_results is None:
            n_results = self.max_retrieved_docs
            
        if self.vectordb_initialized:
            # Use VectorDBBuilder for retrieval
            try:
                results = self.vectordb_builder.query(query, n_results=n_results)
                
                # Convert VectorDBBuilder results to Document-like objects
                documents = []
                for i, (doc_text, distance, metadata) in enumerate(zip(
                    results['documents'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                )):
                    # Create a simple document-like object
                    doc = type('Document', (), {
                        'page_content': doc_text,
                        'metadata': metadata
                    })()
                    documents.append(doc)
                
                logger.info(f"Retrieved {len(documents)} documents using VectorDBBuilder")
                return documents
                
            except Exception as e:
                logger.warning(f"VectorDBBuilder retrieval failed: {e}, falling back to base RAG")
                return self.base_rag.retrieve_relevant_docs(query)
        else:
            # Use base RAG system
            return self.base_rag.retrieve_relevant_docs(query)
    
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
            # Step 1: Retrieve relevant documents using VectorDBBuilder or base RAG
            retrieved_docs = self.retrieve_relevant_docs(query)
            
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
                    try:
                        # Ensure correct tensor shapes for neural retriever
                        query_tensor = query_embedding.unsqueeze(0)  # [1, 768]
                        doc_tensor = doc_embeddings_tensor.unsqueeze(0)  # [1, num_docs, 768]
                        
                        retrieval_scores, neural_retrieved = self.neural_retriever(
                            query_tensor, 
                            doc_tensor
                        )
                    except Exception as e:
                        logger.warning(f"Neural retrieval failed: {e}")
                        retrieval_scores = None
                        neural_retrieved = None
                
                # Convert neural retrieved docs to list format
                if neural_retrieved is not None:
                    retrieved_segments = [neural_retrieved[0, i] for i in range(neural_retrieved.shape[1])]
                else:
                    retrieved_segments = None
            else:
                retrieved_segments = None
                retrieval_scores = None
            
            # Step 4: Hybrid attention processing
            if retrieved_segments is not None and len(retrieved_segments) > 0:
                # Process with hybrid attention
                try:
                    with torch.no_grad():
                        attention_output = self.hybrid_attention(
                            query_embedding.unsqueeze(0),
                            retrieved_segments
                        )
                except Exception as e:
                    logger.warning(f"Hybrid attention failed, using fallback: {e}")
                    attention_output = query_embedding.unsqueeze(0)
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
                'attention_output_shape': list(attention_output.shape) if attention_output is not None else None
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
            try:
                if attention_weights is not None and i < attention_weights.shape[1]:
                    weight = attention_weights[0, i].item()
                else:
                    weight = 1.0
                weighted_contexts.append(f"[Weight: {weight:.3f}] {doc.page_content}")
            except (IndexError, AttributeError):
                # Fallback if attention weights are malformed
                weighted_contexts.append(f"[Weight: 1.000] {doc.page_content}")
        
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
            # Fallback: create a simple direct LLM response
            try:
                response = self.base_rag.llm.invoke(query)
                results['direct_llm'] = {
                    'response': response.content,
                    'retrieved_docs': 0,
                    'context_length': 0
                }
            except Exception as e2:
                results['direct_llm'] = {'error': str(e2)}
        
        return results
    
    def load_vectordb_from_existing(self, db_path: str = None, collection_name: str = None):
        """
        Load an existing VectorDBBuilder database.
        
        Args:
            db_path: Path to the existing database (optional)
            collection_name: Name of the collection (optional)
        """
        if db_path:
            self.vectordb_builder.db_path = db_path
        if collection_name:
            self.vectordb_builder.collection_name = collection_name
            
        # Reinitialize the client and collection
        self.vectordb_builder.client = self.vectordb_builder.client.__class__(path=self.vectordb_builder.db_path)
        self.vectordb_builder.collection = self.vectordb_builder.client.get_collection(
            name=self.vectordb_builder.collection_name
        )
        self.vectordb_initialized = True
        logger.info(f"Loaded existing VectorDB from {self.vectordb_builder.db_path}")
    
    def get_vectordb_stats(self) -> Dict[str, Any]:
        """Get statistics about the VectorDBBuilder database."""
        if not self.vectordb_initialized:
            return {"error": "VectorDB not initialized"}
        
        try:
            count = self.vectordb_builder.collection.count()
            return {
                "collection_name": self.vectordb_builder.collection_name,
                "db_path": self.vectordb_builder.db_path,
                "total_chunks": count,
                "embedding_model": self.vectordb_builder.embedding_function.model_name,
                "chunk_size": self.vectordb_builder.chunk_size,
                "chunk_overlap": self.vectordb_builder.chunk_overlap
            }
        except Exception as e:
            return {"error": str(e)}

def test_working_hybrid_rag():
    """Test the working hybrid RAG system."""
    print("🧪 Testing Working Hybrid Attention RAG with VectorDBBuilder")
    print("=" * 60)
    
    # Create working hybrid RAG system with VectorDBBuilder
    vectordb_config = {
        'db_path': './vector_store',
        'collection_name': 'hybrid_rag_test',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': 500,
        'chunk_overlap': 50,
        'batch_size': 100
    }
    
    hybrid_rag = WorkingHybridRAG(
        use_hybrid_attention=True, 
        vectordb_config=vectordb_config,
        max_retrieved_docs=10  # Retrieve more documents for better context
    )
    
    # Create vector store using VectorDBBuilder with BookCorpus dataset
    print("📚 Creating vector store using VectorDBBuilder with BookCorpus dataset...")
    hybrid_rag.create_vectorstore(
        use_vectordb=True, 
        num_documents=5000  # Use more documents for better results
    )
    print("✅ Vector store created successfully using VectorDBBuilder")
    
    # Show VectorDB stats
    stats = hybrid_rag.get_vectordb_stats()
    print(f"📊 VectorDB Stats: {stats}")
    
    # Test queries appropriate for BookCorpus dataset
    test_queries = [
        "character development and relationships",
        "plot twists and surprises",
        "dialogue and conversation",
        "setting and atmosphere",
        "conflict and resolution"
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
