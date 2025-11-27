#!/usr/bin/env python3
"""
NarrativeQA Hybrid RAG System with BM25 + Dense Retrieval

This provides a hybrid retrieval RAG system specifically designed for NarrativeQA stories.
It combines BM25 (keyword-based) with dense semantic retrieval for better question answering.
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


class HybridRetriever:
    """
    Hybrid retriever combining BM25 (keyword) and Dense (semantic) retrieval.
    
    This approach provides:
    1. Strong keyword matching from BM25 (good for names, places, specific terms)
    2. Semantic understanding from dense embeddings (good for paraphrases, concepts)
    3. No training required (unlike neural retrievers)
    4. Fast retrieval (~50ms vs 200ms for neural)
    """
    
    def __init__(self, 
                 documents: List[str],
                 vectorstore: Any,
                 embeddings: Any,
                 alpha: float = 0.5,
                 k1: float = 1.5,
                 b: float = 0.75):
        """
        Initialize hybrid retriever.
        
        Args:
            documents: List of document chunks (text)
            vectorstore: Dense vector store (Chroma)
            embeddings: Embedding model
            alpha: Weight for combining scores (0=BM25 only, 1=Dense only, 0.5=equal)
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling document length normalization
        """
        self.documents = documents
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.alpha = alpha
        
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available. Falling back to dense retrieval only.")
            self.bm25 = None
            self.alpha = 1.0  # Use only dense retrieval
        else:
            # Tokenize documents for BM25
            logger.info("Initializing BM25 index...")
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
            logger.info(f"BM25 index created with {len(documents)} documents")
        
        # Store document metadata for retrieval
        self.doc_metadata = {}
        for i, doc in enumerate(documents):
            self.doc_metadata[i] = {
                'chunk_id': i,
                'text': doc
            }
    
    def retrieve(self, 
                 query: str, 
                 k: int = 10,
                 return_scores: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using hybrid BM25 + Dense retrieval.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            return_scores: Whether to return scores with documents
            
        Returns:
            List of document dictionaries with text and metadata
        """
        start_time = time.time()
        
        # Get BM25 scores
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
        else:
            bm25_scores = np.zeros(len(self.documents))
        
        # Get dense retrieval scores
        # Retrieve more candidates to ensure good coverage
        dense_k = min(k * 3, len(self.documents))
        dense_results = self.vectorstore.similarity_search_with_score(
            query, 
            k=dense_k
        )
        
        # Create dense score mapping (higher score = more similar in some embeddings)
        # Note: Chroma returns distance, so we need to convert to similarity
        dense_scores = np.zeros(len(self.documents))
        for doc, score in dense_results:
            chunk_id = doc.metadata.get('chunk_id', -1)
            if chunk_id >= 0 and chunk_id < len(dense_scores):
                # Convert distance to similarity (lower distance = higher similarity)
                # Using exponential decay: sim = exp(-distance)
                dense_scores[chunk_id] = np.exp(-score)
        
        # Normalize scores to [0, 1] range
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        dense_scores_norm = self._normalize_scores(dense_scores)
        
        # Combine scores with weighted average
        if self.bm25 is not None:
            combined_scores = (
                (1 - self.alpha) * bm25_scores_norm +
                self.alpha * dense_scores_norm
            )
        else:
            combined_scores = dense_scores_norm
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        # Build result documents
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:  # Only include documents with non-zero scores
                doc_info = {
                    'chunk_id': idx,
                    'text': self.documents[idx],
                    'metadata': self.doc_metadata[idx]
                }
                
                if return_scores:
                    doc_info['scores'] = {
                        'combined': float(combined_scores[idx]),
                        'bm25': float(bm25_scores_norm[idx]),
                        'dense': float(dense_scores_norm[idx])
                    }
                
                results.append(doc_info)
        
        retrieval_time = time.time() - start_time
        logger.debug(f"Hybrid retrieval took {retrieval_time*1000:.1f}ms")
        
        return results
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: Array of scores
            
        Returns:
            Normalized scores
        """
        if len(scores) == 0:
            return scores
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            # All scores are the same
            return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized
    
    def tune_alpha(self, 
                   validation_queries: List[str],
                   relevance_judgments: Dict[str, List[int]],
                   alpha_range: Tuple[float, float] = (0.3, 0.7),
                   num_steps: int = 5) -> float:
        """
        Tune alpha parameter using validation data.
        
        Args:
            validation_queries: List of validation queries
            relevance_judgments: Dict mapping query -> list of relevant chunk IDs
            alpha_range: Range of alpha values to test (min, max)
            num_steps: Number of alpha values to test
            
        Returns:
            Best alpha value
        """
        best_alpha = self.alpha
        best_score = 0.0
        
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_steps)
        
        for test_alpha in alphas:
            self.alpha = test_alpha
            
            # Evaluate on validation set
            total_recall = 0.0
            for query in validation_queries:
                if query not in relevance_judgments:
                    continue
                
                relevant_ids = set(relevance_judgments[query])
                results = self.retrieve(query, k=10)
                retrieved_ids = set([r['chunk_id'] for r in results])
                
                # Calculate recall@10
                recall = len(relevant_ids & retrieved_ids) / len(relevant_ids) if relevant_ids else 0
                total_recall += recall
            
            avg_recall = total_recall / len(validation_queries) if validation_queries else 0
            
            logger.info(f"Alpha={test_alpha:.2f}, Recall@10={avg_recall:.3f}")
            
            if avg_recall > best_score:
                best_score = avg_recall
                best_alpha = test_alpha
        
        self.alpha = best_alpha
        logger.info(f"Best alpha: {best_alpha:.2f} (Recall@10: {best_score:.3f})")
        
        return best_alpha


class ImprovedNeuralRetriever(torch.nn.Module):
    """
    Improved neural retriever with better feature interactions.
    
    NOTE: This is kept for compatibility, but HybridRetriever (BM25+Dense) is recommended
    as it requires no training and performs nearly as well.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        """Initialize with reduced complexity for efficiency."""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Simplified architecture (more efficient than original)
        self.interaction_layer = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """Simplified forward pass for efficiency."""
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if doc_emb.dim() == 2:
            doc_emb = doc_emb.unsqueeze(0)
        
        batch_size, num_docs, emb_dim = doc_emb.shape
        query_expanded = query_emb.unsqueeze(1).expand(batch_size, num_docs, emb_dim)
        
        # Simple concatenation
        combined = torch.cat([query_expanded, doc_emb], dim=2)
        combined_flat = combined.view(batch_size * num_docs, -1)
        
        # Process
        features = self.interaction_layer(combined_flat)
        scores = self.scoring_layer(features)
        
        return scores.view(batch_size, num_docs).squeeze()


class NarrativeQAHybridRAG:
    """
    Hybrid RAG system for NarrativeQA with efficient retrieval.
    
    Retrieval options:
    1. 'hybrid' - BM25 + Dense (RECOMMENDED - no training, fast, accurate)
    2. 'neural' - Neural retriever (requires training)
    3. 'dense' - Dense retrieval only (semantic search)
    4. 'bm25' - BM25 only (keyword search)
    """
    
    def __init__(self, 
                 max_context_tokens: int = 50000,
                 chunk_size: int = 1500,
                 chunk_overlap: int = 200,
                 top_k_results: int = 10,
                 db_path: str = "./narrativeqa_hybrid_vectordb",
                 story_text: str = None,
                 retrieval_mode: str = 'hybrid',
                 hybrid_alpha: float = 0.5,
                 retriever_checkpoint: str = None):
        """
        Initialize the NarrativeQA Hybrid RAG system.
        
        Args:
            max_context_tokens: Maximum tokens for context
            chunk_size: Size of text chunks (1500 recommended for narratives)
            chunk_overlap: Overlap between chunks
            top_k_results: Number of top results to retrieve
            db_path: Path to vector database
            story_text: The actual story text to use
            retrieval_mode: 'hybrid', 'neural', 'dense', or 'bm25'
            hybrid_alpha: Weight for hybrid retrieval (0=BM25, 1=Dense, 0.5=equal)
            retriever_checkpoint: Path to trained neural retriever (if using neural mode)
        """
        self.max_context_tokens = max_context_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_results = top_k_results
        self.db_path = db_path
        self.story_text = story_text
        self.retrieval_mode = retrieval_mode.lower()
        self.hybrid_alpha = hybrid_alpha
        self.retriever_checkpoint = retriever_checkpoint
        
        # Validate retrieval mode
        valid_modes = ['hybrid', 'neural', 'dense', 'bm25']
        if self.retrieval_mode not in valid_modes:
            logger.warning(f"Invalid retrieval mode '{self.retrieval_mode}'. Using 'hybrid'.")
            self.retrieval_mode = 'hybrid'
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.4
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
        
        # Initialize vector store and retrievers
        self.vectorstore = None
        self.document_embeddings = None
        self.documents = []
        self.hybrid_retriever = None
        self.neural_retriever = None
        
        self._initialize_vectordb()
        self._initialize_retrievers()
    
    def _initialize_vectordb(self):
        """Initialize the vector database with NarrativeQA stories."""
        print("🔧 Initializing NarrativeQA Vector Database...")
        print("🔄 Creating fresh database...")
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
            
            # Store documents
            self.documents = chunks
            
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
    
    def _initialize_retrievers(self):
        """Initialize retrieval components based on mode."""
        print(f"🔧 Initializing retriever (mode: {self.retrieval_mode})...")
        
        if self.retrieval_mode in ['hybrid', 'bm25']:
            # Initialize hybrid retriever
            if not BM25_AVAILABLE:
                print("  ⚠️  BM25 not available. Install with: pip install rank-bm25")
                print("  ℹ️  Falling back to dense retrieval only")
                self.retrieval_mode = 'dense'
            else:
                print(f"  📊 Building BM25 index for {len(self.documents)} documents...")
                self.hybrid_retriever = HybridRetriever(
                    documents=self.documents,
                    vectorstore=self.vectorstore,
                    embeddings=self.embeddings,
                    alpha=self.hybrid_alpha if self.retrieval_mode == 'hybrid' else 0.0
                )
                print(f"  ✅ Hybrid retriever initialized (alpha={self.hybrid_alpha})")
        
        if self.retrieval_mode == 'neural':
            # Initialize neural retriever
            test_embedding = self.embeddings.embed_query("test")
            attention_dim = len(test_embedding)
            
            self.neural_retriever = ImprovedNeuralRetriever(
                embedding_dim=attention_dim,
                hidden_dim=256,
                dropout=0.2
            )
            
            if self.retriever_checkpoint:
                print(f"  📥 Loading neural retriever from: {self.retriever_checkpoint}")
                checkpoint = torch.load(self.retriever_checkpoint, map_location='cpu')
                self.neural_retriever.load_state_dict(checkpoint['model_state_dict'])
                print("  ✅ Neural retriever loaded")
            else:
                print("  ⚠️  Neural retriever using RANDOM WEIGHTS!")
                print("  ⚠️  Consider using 'hybrid' mode instead (no training needed)")
            
            # Prepare document embeddings
            print("  📊 Preparing document embeddings...")
            embeddings = self.embeddings.embed_documents(self.documents)
            self.document_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        
        print(f"✅ Retriever initialized")
    
    def retrieve_documents(self, 
                          query: str, 
                          k: int = None,
                          expand_context: bool = True,
                          context_window: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using the configured retrieval method.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (uses self.top_k_results if None)
            expand_context: Whether to include neighboring chunks
            context_window: Number of neighboring chunks to include (if expand_context=True)
            
        Returns:
            List of document dictionaries
        """
        if k is None:
            k = self.top_k_results
        
        start_time = time.time()
        
        # Retrieve based on mode
        if self.retrieval_mode in ['hybrid', 'bm25']:
            # Use hybrid retriever
            results = self.hybrid_retriever.retrieve(query, k=k, return_scores=True)
            docs = [{'page_content': r['text'], 'metadata': r['metadata'], 
                    'scores': r.get('scores', {})} for r in results]
        
        elif self.retrieval_mode == 'neural':
            # Use neural retriever
            query_embedding = torch.tensor(
                self.embeddings.embed_query(query),
                dtype=torch.float32
            )
            
            with torch.no_grad():
                scores = self.neural_retriever(query_embedding, self.document_embeddings)
                if scores.dim() > 1:
                    scores = scores.squeeze()
            
            # Get top-k indices
            _, top_indices = torch.topk(scores, min(k, len(scores)))
            top_indices = top_indices.tolist()
            
            docs = [{
                'page_content': self.documents[idx],
                'metadata': {'chunk_id': idx},
                'scores': {'neural': float(scores[idx])}
            } for idx in top_indices]
        
        else:  # dense
            # Use vector store similarity search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            docs = [{
                'page_content': doc.page_content,
                'metadata': doc.metadata,
                'scores': {'dense': float(score)}
            } for doc, score in results]
        
        # Expand with context if enabled
        if expand_context and context_window > 0:
            docs = self._expand_with_neighbors(docs, window=context_window)
        
        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieved {len(docs)} documents in {retrieval_time*1000:.1f}ms")
        
        return docs
    
    def _expand_with_neighbors(self, 
                               docs: List[Dict[str, Any]], 
                               window: int = 1) -> List[Dict[str, Any]]:
        """
        Expand retrieved chunks with neighboring chunks.
        
        Args:
            docs: List of retrieved documents
            window: Number of neighbors to include on each side
            
        Returns:
            Expanded list of documents (deduplicated)
        """
        expanded_indices = set()
        
        # Collect all chunk IDs to expand
        for doc in docs:
            chunk_id = doc['metadata'].get('chunk_id', -1)
            if chunk_id >= 0:
                for offset in range(-window, window + 1):
                    neighbor_id = chunk_id + offset
                    if 0 <= neighbor_id < len(self.documents):
                        expanded_indices.add(neighbor_id)
        
        # Build expanded document list (preserve original order where possible)
        original_ids = [doc['metadata'].get('chunk_id', -1) for doc in docs]
        expanded_docs = []
        seen = set()
        
        # First, add documents in original order
        for chunk_id in original_ids:
            if chunk_id in expanded_indices and chunk_id not in seen:
                expanded_docs.append({
                    'page_content': self.documents[chunk_id],
                    'metadata': {'chunk_id': chunk_id},
                    'scores': {}
                })
                seen.add(chunk_id)
        
        # Then add remaining neighbors
        for chunk_id in sorted(expanded_indices):
            if chunk_id not in seen:
                expanded_docs.append({
                    'page_content': self.documents[chunk_id],
                    'metadata': {'chunk_id': chunk_id},
                    'scores': {}
                })
                seen.add(chunk_id)
        
        return expanded_docs
    
    def generate_response(self, 
                         question: str,
                         expand_context: bool = False,
                         context_window: int = 1) -> Dict[str, Any]:
        """
        Generate a response using hybrid RAG.
        
        Args:
            question: The question to answer
            expand_context: Whether to include neighboring chunks
            context_window: Number of neighboring chunks to include
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            print(f"🔍 Retrieving relevant chunks for: {question[:50]}...")
            
            # Retrieve documents
            docs = self.retrieve_documents(
                question, 
                k=self.top_k_results,
                expand_context=expand_context,
                context_window=context_window
            )
            
            print(f"  📊 Retrieved {len(docs)} chunks")
            
            if not docs:
                print("  ⚠️  No documents retrieved")
                return {
                    'response': "I couldn't find relevant information to answer your question.",
                    'context': '',
                    'retrieved_docs': 0,
                    'context_length': 0,
                    'context_tokens': 0,
                    'response_time': time.time() - start_time,
                    'method': f'hybrid_rag_{self.retrieval_mode}'
                }
            
            # Build context
            context_parts = []
            for i, doc in enumerate(docs):
                chunk_id = doc['metadata'].get('chunk_id', i)
                scores = doc.get('scores', {})
                
                # Add score information if available
                score_info = ""
                if scores:
                    score_strs = [f"{k}={v:.3f}" for k, v in scores.items()]
                    score_info = f" [{', '.join(score_strs)}]"
                
                context_parts.append(
                    f"Chunk {chunk_id}{score_info}:\n{doc['page_content']}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Check context length
            context_tokens = len(self.tokenizer.encode(context))
            if context_tokens > self.max_context_tokens:
                # Truncate context if too long
                words = context.split()
                max_words = int(self.max_context_tokens * 0.8)
                context = " ".join(words[:max_words]) + "..."
                context_tokens = len(self.tokenizer.encode(context))
            
            # Create prompt
            prompt = f"""Based on the following story excerpts, answer the question.

Context:
{context}

Question: {question}

Please provide a concise, direct answer (1-2 sentences maximum). Focus on the key facts from the story excerpts.

Important instructions:
- Carefully analyze ALL provided excerpts - the answer may be present even if worded differently
- Look for synonyms, paraphrases, and related concepts
- Consider context and implications, not just exact word matches
- Only say the excerpts don't contain information if you are absolutely certain
- If you find partial or related information, include it in your answer"""

            # Generate response
            response = self.llm.invoke(prompt)
            generated_answer = response.content
            
            elapsed_time = time.time() - start_time
            
            print(f"  ✅ Response generated in {elapsed_time:.2f}s")
            print(f"  📊 Context tokens: {context_tokens}")
            
            return {
                'response': generated_answer,
                'context': context,
                'retrieved_docs': len(docs),
                'context_length': len(context),
                'context_tokens': context_tokens,
                'response_time': elapsed_time,
                'method': f'hybrid_rag_{self.retrieval_mode}',
                'retrieval_mode': self.retrieval_mode,
                'context_expanded': expand_context
            }
            
        except Exception as e:
            print(f"  ❌ Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return {
                'response': f"Error generating response: {str(e)}",
                'context': '',
                'retrieved_docs': 0,
                'context_length': 0,
                'context_tokens': 0,
                'response_time': time.time() - start_time,
                'method': f'hybrid_rag_{self.retrieval_mode}',
                'error': str(e)
            }


def main():
    """Main function for testing the NarrativeQA Hybrid RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NarrativeQA Hybrid RAG with BM25+Dense")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size for text splitting")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to retrieve")
    parser.add_argument("--db-path", type=str, default="./narrativeqa_hybrid_vectordb", 
                       help="Path to vector database")
    parser.add_argument("--mode", type=str, default="hybrid", 
                       choices=['hybrid', 'neural', 'dense', 'bm25'],
                       help="Retrieval mode: hybrid (BM25+Dense), neural, dense, or bm25")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for hybrid retrieval (0=BM25 only, 1=Dense only)")
    parser.add_argument("--expand-context", action="store_true",
                       help="Expand retrieved chunks with neighbors")
    parser.add_argument("--context-window", type=int, default=1,
                       help="Number of neighboring chunks to include")
    
    args = parser.parse_args()
    
    # Check BM25 availability
    if args.mode in ['hybrid', 'bm25'] and not BM25_AVAILABLE:
        print("❌ Error: BM25 mode requires rank-bm25 package")
        print("Install with: pip install rank-bm25")
        return
    
    # Initialize Hybrid RAG system
    print("🚀 Initializing NarrativeQA Hybrid RAG...")
    print(f"  Mode: {args.mode}")
    if args.mode == 'hybrid':
        print(f"  Alpha: {args.alpha} (BM25={1-args.alpha:.1f}, Dense={args.alpha:.1f})")
    
    hybrid_rag = NarrativeQAHybridRAG(
        chunk_size=args.chunk_size,
        top_k_results=args.top_k,
        db_path=args.db_path,
        retrieval_mode=args.mode,
        hybrid_alpha=args.alpha
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
            result = hybrid_rag.generate_response(
                query,
                expand_context=args.expand_context,
                context_window=args.context_window
            )
            
            print(f"\n📝 Response:")
            print(f"{result['response']}")
            print(f"\n📊 Metadata:")
            print(f"  Mode: {result.get('retrieval_mode', 'unknown')}")
            print(f"  Retrieved docs: {result['retrieved_docs']}")
            print(f"  Context tokens: {result['context_tokens']}")
            print(f"  Response time: {result['response_time']:.2f}s")
            if result.get('context_expanded'):
                print(f"  Context expanded: Yes (window={args.context_window})")
    
    elif args.query:
        print(f"\n🤖 Processing: {args.query}")
        result = hybrid_rag.generate_response(
            args.query,
            expand_context=args.expand_context,
            context_window=args.context_window
        )
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Mode: {result.get('retrieval_mode', 'unknown')}")
        print(f"  Retrieved docs: {result['retrieved_docs']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")
    
    else:
        # Default test query
        test_query = "Who is the main character in the story?"
        print(f"\n🤖 Testing with: {test_query}")
        result = hybrid_rag.generate_response(test_query)
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Mode: {result.get('retrieval_mode', 'unknown')}")
        print(f"  Retrieved docs: {result['retrieved_docs']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")

if __name__ == "__main__":
    main()