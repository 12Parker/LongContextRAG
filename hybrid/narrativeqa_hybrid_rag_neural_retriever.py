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
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress indicator
    def tqdm(iterable, **kwargs):
        return iterable

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


class ImprovedNeuralRetriever(torch.nn.Module):
    """
    Improved neural retriever with better feature interactions.
    
    Architecture improvements:
    1. Element-wise product for query-document interaction
    2. Cross-attention mechanism
    3. Residual connections
    4. Batch normalization and dropout for regularization
    5. Deeper network with skip connections
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Feature interaction layers
        # 1. Element-wise product (captures multiplicative interactions)
        self.interaction_dim = embedding_dim
        
        # 2. Cross-attention for query-document alignment
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Feature combination: [query, doc, query*doc, attention_output]
        combined_dim = embedding_dim * 4  # query + doc + product + attention
        
        # 4. Deep network with residual connections
        # Use LayerNorm instead of BatchNorm1d to handle variable batch sizes
        self.feature_projection = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.hidden_layer1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.hidden_layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with improved feature interactions.
        
        Args:
            query_emb: Query embedding [batch_size, embedding_dim] or [embedding_dim]
            doc_emb: Document embeddings [num_docs, embedding_dim] or [batch_size, num_docs, embedding_dim]
            
        Returns:
            Relevance scores [num_docs] or [batch_size, num_docs]
        """
        # Handle single query case
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)  # [1, embedding_dim]
            single_query = True
        else:
            single_query = False
        
        if doc_emb.dim() == 2:
            doc_emb = doc_emb.unsqueeze(0)  # [1, num_docs, embedding_dim]
            single_batch = True
        else:
            single_batch = False
        
        batch_size, num_docs, emb_dim = doc_emb.shape
        
        # Expand query to match document batch
        query_expanded = query_emb.unsqueeze(1).expand(batch_size, num_docs, emb_dim)
        
        # 1. Element-wise product (multiplicative interaction)
        elementwise_product = query_expanded * doc_emb  # [batch_size, num_docs, embedding_dim]
        
        # 2. Cross-attention: query attends to documents
        query_for_attention = query_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        attended_output, _ = self.cross_attention(
            query_for_attention, doc_emb, doc_emb
        )  # [batch_size, 1, embedding_dim]
        attended_output = attended_output.expand(batch_size, num_docs, emb_dim)
        
        # 3. Concatenate all features
        combined_features = torch.cat([
            query_expanded,           # Original query
            doc_emb,                  # Original document
            elementwise_product,      # Multiplicative interaction
            attended_output           # Attention-aligned features
        ], dim=2)  # [batch_size, num_docs, embedding_dim * 4]
        
        # Reshape for batch processing
        combined_flat = combined_features.view(batch_size * num_docs, -1)
        
        # 4. Pass through deep network
        x = self.feature_projection(combined_flat)
        residual = x
        
        x = self.hidden_layer1(x)
        x = x + residual  # Residual connection
        
        x = self.hidden_layer2(x)
        
        # 5. Output relevance score
        scores = self.output_layer(x)  # [batch_size * num_docs, 1]
        scores = scores.view(batch_size, num_docs)
        
        # Handle single batch/query cases
        if single_batch:
            scores = scores.squeeze(0)  # [num_docs]
        if single_query and single_batch:
            scores = scores.squeeze(0)  # [num_docs] (already done)
        
        return scores

class NarrativeQAHybridRAG:
    """
    Hybrid attention RAG system specifically for NarrativeQA stories.
    
    This system combines:
    1. Neural retrieval for finding relevant story segments
    2. Attention mechanisms for focusing on important parts
    3. LLM generation for final answer synthesis
    """
    
    def __init__(self, 
                 max_context_tokens: int = 4000,
                 chunk_size: int = 600,
                 chunk_overlap: int = 0,
                 top_k_results: int = 5,
                 db_path: str = "./narrativeqa_hybrid_vectordb",
                 story_text: str = None,
                 use_hybrid_attention: bool = True,
                 retriever_checkpoint: str = None):
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
            retriever_checkpoint: Path to trained neural retriever checkpoint (optional)
        """
        self.max_context_tokens = max_context_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_results = top_k_results
        self.db_path = db_path
        self.story_text = story_text
        self.use_hybrid_attention = use_hybrid_attention
        self.retriever_checkpoint = retriever_checkpoint
        
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
        
        # Initialize vector store
        self.vectorstore = None
        self.document_embeddings = None
        self._initialize_vectordb()
        
        if use_hybrid_attention:
            self._initialize_attention_components()
            
            # Load trained checkpoint if provided
            if retriever_checkpoint:
                logger.info(f"Loading neural retriever checkpoint from: {retriever_checkpoint}")
                self.load_retriever_checkpoint(retriever_checkpoint)
            else:
                logger.warning("⚠️  Neural retriever using RANDOM WEIGHTS - performance may be poor!")
                logger.warning("⚠️  Train the neural retriever or load a checkpoint for better results.")
    
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
        
        # Initialize improved neural retriever with better architecture
        self.retriever_hidden_dim = 512
        self.retriever_dropout = 0.2
        
        # Improved neural retriever with feature interactions
        self.neural_retriever = ImprovedNeuralRetriever(
            embedding_dim=self.attention_dim,
            hidden_dim=self.retriever_hidden_dim,
            dropout=self.retriever_dropout
        )
        
        # Training state
        self.retriever_trained = False
        self.retriever_checkpoint_path = None
        
        logger.info(f"Hybrid attention components initialized with embedding dim: {self.attention_dim}")
        logger.info(f"Neural retriever architecture: {self.neural_retriever}")
    
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
            
            # Calculate similarity scores using improved neural retriever
            with torch.no_grad():
                scores = self.neural_retriever(query_embedding, self.document_embeddings)
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
            
            # Calculate similarity scores using improved neural retriever
            with torch.no_grad():
                scores = self.neural_retriever(query_embedding, doc_embeddings)
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
            # Retrieve more chunks initially to give neural retriever more options
            initial_k = min(self.top_k_results * 3, 60)  # Retrieve up to 60 chunks initially
            print(f"🔍 Retrieving relevant chunks for: {question[:50]}...")
            docs = self.vectorstore.similarity_search(
                question, 
                k=initial_k
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
            # Store similarity search results as fallback
            similarity_docs = docs[:self.top_k_results] if len(docs) > self.top_k_results else docs
            
            if self.use_hybrid_attention and self.document_embeddings is not None:
                # Get document embeddings for retrieved chunks
                doc_contents = [doc.page_content for doc in docs]
                doc_embeddings = torch.tensor(
                    self.embeddings.embed_documents(doc_contents),
                    dtype=torch.float32
                )
                
                # Apply neural retrieval to re-rank chunks (only on retrieved chunks)
                # Keep more chunks to ensure we don't filter out relevant information
                neural_indices = self._neural_retrieve_on_chunks(query_embedding, doc_embeddings, min(len(docs), self.top_k_results * 2))
                neural_retrieved_docs = [docs[i] for i in neural_indices if i < len(docs)]
                
                # Apply attention mechanism
                attended_embeddings = self._apply_attention(query_embedding, doc_embeddings)
                
                # Hybrid approach: Combine similarity search top results with neural retrieval top results
                # This ensures we don't lose information if neural retriever filters out relevant chunks
                # Take more from each to ensure we have enough after deduplication
                neural_top = neural_retrieved_docs[:min(len(neural_retrieved_docs), self.top_k_results * 2)]
                similarity_top = similarity_docs[:min(len(similarity_docs), self.top_k_results * 2)]
                
                # Combine and deduplicate (neural results first, then similarity results)
                combined_docs = []
                seen_content = set()
                
                # Add neural results first (they're ranked by neural retriever)
                for doc in neural_top:
                    content_hash = hash(doc.page_content[:200])  # Use first 200 chars for better uniqueness
                    if content_hash not in seen_content:
                        combined_docs.append(doc)
                        seen_content.add(content_hash)
                        if len(combined_docs) >= self.top_k_results:
                            break
                
                # Fill remaining slots with similarity results if needed
                if len(combined_docs) < self.top_k_results:
                    for doc in similarity_top:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            combined_docs.append(doc)
                            seen_content.add(content_hash)
                            if len(combined_docs) >= self.top_k_results:
                                break
                
                # Ensure we have at least some chunks (fallback to similarity if neural failed)
                if not combined_docs:
                    docs = similarity_top[:self.top_k_results] if similarity_top else docs[:self.top_k_results]
                else:
                    docs = combined_docs
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

Please provide a concise, direct answer (1-2 sentences maximum). Focus on the key facts from the story excerpts. 

Important instructions:
- Carefully analyze ALL provided excerpts - the answer may be present even if worded differently
- Look for synonyms, paraphrases, and related concepts (e.g., "takes his own life" = "commits suicide", "radio station" = "broadcast")
- Consider context and implications, not just exact word matches
- Only say the excerpts don't contain information if you have thoroughly checked all excerpts multiple times and are absolutely certain the information is not present
- If you find partial or related information, include it in your answer"""

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
    
    def train_neural_retriever(self,
                               training_data: List[Dict[str, Any]],
                               epochs: int = 10,
                               batch_size: int = 32,
                               learning_rate: float = 1e-4,
                               margin: float = 1.0,
                               checkpoint_dir: str = "./checkpoints",
                               save_best: bool = True,
                               validation_data: List[Dict[str, Any]] = None):
        """
        Train the neural retriever using contrastive ranking loss.
        
        Args:
            training_data: List of training examples with format:
                {
                    'question': str,
                    'positive_chunks': List[str],  # Relevant chunks
                    'negative_chunks': List[str],   # Irrelevant chunks (optional)
                    'story_id': str
                }
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            margin: Margin for contrastive loss
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model based on validation
            validation_data: Optional validation data for early stopping
        """
        if not self.use_hybrid_attention:
            logger.warning("Hybrid attention is disabled. Cannot train neural retriever.")
            return
        
        logger.info(f"Starting neural retriever training with {len(training_data)} examples")
        
        # Prepare training data
        train_loader = self._prepare_training_data(training_data, batch_size)
        total_batches = len(train_loader)
        logger.info(f"Total batches per epoch: {total_batches}")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.neural_retriever.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Training loop
        best_loss = float('inf')
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        self.neural_retriever.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()
            
            # Create progress bar for this epoch
            pbar = tqdm(
                enumerate(train_loader),
                total=total_batches,
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="batch",
                disable=not TQDM_AVAILABLE
            )
            
            for batch_idx, batch in pbar:
                optimizer.zero_grad()
                
                query_embs = batch['query_embeddings']
                pos_doc_embs_list = batch['positive_embeddings']
                neg_doc_embs_list = batch.get('negative_embeddings', None)
                
                # Process each example separately (handle variable lengths)
                batch_losses = []
                
                for i, query_emb in enumerate(query_embs):
                    pos_doc_embs = pos_doc_embs_list[i]
                    pos_scores = self.neural_retriever(query_emb, pos_doc_embs)
                    
                    # Check if negative embeddings exist for this example
                    has_negatives = (neg_doc_embs_list is not None and 
                                   i < len(neg_doc_embs_list) and 
                                   neg_doc_embs_list[i] is not None)
                    
                    if has_negatives:
                        # Contrastive loss: compare positive vs negative
                        neg_doc_embs = neg_doc_embs_list[i]
                        neg_scores = self.neural_retriever(query_emb, neg_doc_embs)
                        
                        # Compute loss for this example
                        # Get max positive and min negative
                        max_pos = pos_scores.max()
                        min_neg = neg_scores.min()
                        
                        # Contrastive loss: margin - (max_pos - min_neg)
                        example_loss = torch.clamp(margin - (max_pos - min_neg), min=0.0)
                        batch_losses.append(example_loss)
                    else:
                        # Pointwise loss: positive chunks should have high scores
                        target_scores = torch.ones_like(pos_scores)
                        example_loss = torch.nn.functional.mse_loss(pos_scores, target_scores)
                        batch_losses.append(example_loss)
                
                # Average loss across batch
                if batch_losses:
                    loss = torch.stack(batch_losses).mean()
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.neural_retriever.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    current_avg_loss = epoch_loss / num_batches
                    pbar.set_postfix({
                        'loss': f'{current_avg_loss:.4f}',
                        'batch': f'{batch_idx+1}/{total_batches}'
                    })
            
            pbar.close()
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            # Evaluate on validation set if provided
            val_loss = None
            if validation_data:
                logger.info(f"Evaluating on validation set ({len(validation_data)} examples)...")
                val_loss = self._evaluate_validation(validation_data, margin)
                logger.info(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.1f}s - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.1f}s - Average Loss: {avg_loss:.4f}")
            
            # Save best model (use validation loss if available, otherwise training loss)
            loss_to_compare = val_loss if val_loss is not None else avg_loss
            if save_best and loss_to_compare < best_loss:
                best_loss = loss_to_compare
                checkpoint_file = checkpoint_path / f"neural_retriever_best.pt"
                self.save_retriever_checkpoint(str(checkpoint_file))
                logger.info(f"Saved best model with {'validation' if val_loss else 'training'} loss: {best_loss:.4f}")
        
        self.neural_retriever.eval()
        self.retriever_trained = True
        logger.info("Neural retriever training completed")
    
    def _evaluate_validation(self, validation_data: List[Dict[str, Any]], margin: float) -> float:
        """Evaluate model on validation set."""
        self.neural_retriever.eval()
        val_losses = []
        
        # Add progress bar for validation
        val_pbar = tqdm(
            validation_data,
            desc="Validation",
            unit="example",
            disable=not TQDM_AVAILABLE
        )
        
        with torch.no_grad():
            for example in val_pbar:
                question = example['question']
                positive_chunks = example.get('positive_chunks', [])
                negative_chunks = example.get('negative_chunks', [])
                
                if not positive_chunks:
                    continue
                
                # Get embeddings
                query_emb = torch.tensor(
                    self.embeddings.embed_query(question),
                    dtype=torch.float32
                )
                
                pos_embs = torch.tensor(
                    self.embeddings.embed_documents(positive_chunks),
                    dtype=torch.float32
                )
                
                pos_scores = self.neural_retriever(query_emb, pos_embs)
                
                if negative_chunks:
                    neg_embs = torch.tensor(
                        self.embeddings.embed_documents(negative_chunks),
                        dtype=torch.float32
                    )
                    neg_scores = self.neural_retriever(query_emb, neg_embs)
                    
                    max_pos = pos_scores.max()
                    min_neg = neg_scores.min()
                    example_loss = torch.clamp(margin - (max_pos - min_neg), min=0.0)
                else:
                    target_scores = torch.ones_like(pos_scores)
                    example_loss = torch.nn.functional.mse_loss(pos_scores, target_scores)
                
                val_losses.append(example_loss.item())
                if len(val_losses) % 100 == 0:
                    current_val_loss = sum(val_losses) / len(val_losses)
                    val_pbar.set_postfix({'val_loss': f'{current_val_loss:.4f}'})
        
        val_pbar.close()
        self.neural_retriever.train()
        return sum(val_losses) / len(val_losses) if val_losses else float('inf')
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]], batch_size: int):
        """
        Prepare training data into batches.
        
        Returns:
            Generator of batches
        """
        batches = []
        current_batch = {
            'query_embeddings': [],
            'positive_embeddings': [],
            'negative_embeddings': []
        }
        
        for example in training_data:
            question = example['question']
            positive_chunks = example.get('positive_chunks', [])
            negative_chunks = example.get('negative_chunks', [])
            
            if not positive_chunks:
                continue
            
            # Get embeddings
            query_emb = torch.tensor(
                self.embeddings.embed_query(question),
                dtype=torch.float32
            )
            
            pos_embs = torch.tensor(
                self.embeddings.embed_documents(positive_chunks),
                dtype=torch.float32
            )
            
            neg_embs = None
            if negative_chunks:
                neg_embs = torch.tensor(
                    self.embeddings.embed_documents(negative_chunks),
                    dtype=torch.float32
                )
            
            current_batch['query_embeddings'].append(query_emb)
            current_batch['positive_embeddings'].append(pos_embs)
            # Always append to negative_embeddings (None if no negatives)
            if 'negative_embeddings' not in current_batch:
                current_batch['negative_embeddings'] = []
            current_batch['negative_embeddings'].append(neg_embs)  # Can be None
            
            if len(current_batch['query_embeddings']) >= batch_size:
                batches.append(self._format_batch(current_batch))
                current_batch = {
                    'query_embeddings': [],
                    'positive_embeddings': [],
                    'negative_embeddings': []  # Will contain None for examples without negatives
                }
        
        # Add remaining batch
        if current_batch['query_embeddings']:
            batches.append(self._format_batch(current_batch))
        
        return batches
    
    def _format_batch(self, batch: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Format batch for training."""
        # Stack query embeddings
        query_embs = torch.stack(batch['query_embeddings'])
        
        # For positive embeddings, we need to handle variable lengths
        # Use padding or process separately
        pos_embs = batch['positive_embeddings']
        
        neg_embs = None
        if batch.get('negative_embeddings') and batch['negative_embeddings']:
            neg_embs = batch['negative_embeddings']
        
        return {
            'query_embeddings': query_embs,
            'positive_embeddings': pos_embs,
            'negative_embeddings': neg_embs
        }
    
    def save_retriever_checkpoint(self, checkpoint_path: str):
        """Save neural retriever checkpoint."""
        checkpoint = {
            'model_state_dict': self.neural_retriever.state_dict(),
            'embedding_dim': self.attention_dim,
            'hidden_dim': self.retriever_hidden_dim,
            'dropout': self.retriever_dropout,
            'trained': self.retriever_trained
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_retriever_checkpoint(self, checkpoint_path: str):
        """Load neural retriever checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.neural_retriever.load_state_dict(checkpoint['model_state_dict'])
        self.retriever_trained = checkpoint.get('trained', False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.retriever_checkpoint_path = checkpoint_path


class ContrastiveRankingLoss(torch.nn.Module):
    """
    Contrastive ranking loss for training neural retriever.
    
    Maximizes the margin between positive and negative document scores.
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive ranking loss.
        
        Args:
            pos_scores: Scores for positive documents [batch_size, num_pos]
            neg_scores: Scores for negative documents [batch_size, num_neg]
            
        Returns:
            Loss value
        """
        # Get max positive score and min negative score for each query
        max_pos = pos_scores.max(dim=1)[0]  # [batch_size]
        min_neg = neg_scores.min(dim=1)[0]  # [batch_size]
        
        # Loss: margin - (max_pos - min_neg)
        # We want max_pos >> min_neg, so loss should be small when margin is satisfied
        loss = torch.clamp(self.margin - (max_pos - min_neg), min=0.0)
        
        return loss.mean()

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
