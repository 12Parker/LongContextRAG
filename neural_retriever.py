"""
Neural Retriever Components for End-to-End Training

This module implements learnable retrieval components that can be trained
end-to-end with the language model, including:
1. Neural retrievers with learnable embeddings
2. Dynamic query generation
3. End-to-end training framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class RetrieverConfig:
    """Configuration for neural retriever components."""
    # Embedding dimensions
    query_embed_dim: int = 768
    doc_embed_dim: int = 768
    hidden_dim: int = 512
    
    # Retrieval parameters
    num_candidates: int = 100
    top_k: int = 8
    retrieval_temperature: float = 1.0
    
    # Dynamic query generation
    query_expansion_layers: int = 2
    query_attention_heads: int = 8
    
    # Training parameters
    contrastive_margin: float = 0.2
    hard_negative_ratio: float = 0.5

class NeuralRetriever(nn.Module):
    """
    Neural retriever with learnable embeddings that can be trained end-to-end.
    """
    
    def __init__(self, config: RetrieverConfig):
        super().__init__()
        self.config = config
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(config.query_embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # Document encoder
        self.doc_encoder = nn.Sequential(
            nn.Linear(config.doc_embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # Similarity projection
        self.similarity_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Temperature parameter for softmax
        self.temperature = nn.Parameter(torch.tensor(config.retrieval_temperature))
        
    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor,
                doc_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for neural retrieval.
        
        Args:
            query_embeddings: [batch_size, query_len, embed_dim]
            doc_embeddings: [num_docs, doc_len, embed_dim]
            doc_ids: [batch_size, num_candidates] (optional)
            
        Returns:
            scores: [batch_size, num_candidates]
            retrieved_docs: [batch_size, top_k, doc_len, embed_dim]
        """
        batch_size, query_len, _ = query_embeddings.shape
        num_docs, doc_len, _ = doc_embeddings.shape
        
        # Encode queries
        query_repr = self.query_encoder(query_embeddings.mean(dim=1))  # [batch_size, hidden_dim]
        
        # Encode documents
        doc_repr = self.doc_encoder(doc_embeddings.mean(dim=1))  # [num_docs, hidden_dim]
        
        # Calculate similarity scores
        query_proj = self.similarity_proj(query_repr)  # [batch_size, hidden_dim]
        doc_proj = self.similarity_proj(doc_repr)  # [num_docs, hidden_dim]
        
        # Cosine similarity
        query_norm = F.normalize(query_proj, p=2, dim=-1)
        doc_norm = F.normalize(doc_proj, p=2, dim=-1)
        
        scores = torch.matmul(query_norm, doc_norm.t()) / self.temperature  # [batch_size, num_docs]
        
        # Select top-k documents
        top_k_scores, top_k_indices = torch.topk(scores, self.config.top_k, dim=-1)
        
        # Retrieve document embeddings
        retrieved_docs = doc_embeddings[top_k_indices]  # [batch_size, top_k, doc_len, embed_dim]
        
        return top_k_scores, retrieved_docs

class DynamicQueryGenerator(nn.Module):
    """
    Dynamic query generation that adaptively aligns retrieval with specific task requirements.
    """
    
    def __init__(self, config: RetrieverConfig):
        super().__init__()
        self.config = config
        
        # Query expansion network
        self.query_expansion = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.query_embed_dim,
                nhead=config.query_attention_heads,
                dim_feedforward=config.hidden_dim,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config.query_expansion_layers)
        ])
        
        # Task-specific query adaptation
        self.task_adaptation = nn.ModuleDict({
            'classification': nn.Linear(config.query_embed_dim, config.query_embed_dim),
            'generation': nn.Linear(config.query_embed_dim, config.query_embed_dim),
            'qa': nn.Linear(config.query_embed_dim, config.query_embed_dim),
            'summarization': nn.Linear(config.query_embed_dim, config.query_embed_dim)
        })
        
        # Processing state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.query_embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.query_embed_dim)
        )
        
        # Query fusion
        self.query_fusion = nn.MultiheadAttention(
            embed_dim=config.query_embed_dim,
            num_heads=config.query_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, original_query: torch.Tensor, processing_state: torch.Tensor,
                task_type: str = 'generation') -> torch.Tensor:
        """
        Generate dynamic query based on processing state and task requirements.
        
        Args:
            original_query: [batch_size, query_len, embed_dim]
            processing_state: [batch_size, embed_dim]
            task_type: Type of task for adaptation
            
        Returns:
            dynamic_query: [batch_size, query_len, embed_dim]
        """
        # Encode processing state
        state_encoded = self.state_encoder(processing_state)  # [batch_size, embed_dim]
        
        # Task-specific adaptation
        if task_type in self.task_adaptation:
            adapted_query = self.task_adaptation[task_type](original_query)
        else:
            adapted_query = original_query
        
        # Query expansion
        expanded_query = adapted_query
        for layer in self.query_expansion:
            expanded_query = layer(expanded_query)
        
        # Fuse with processing state
        state_expanded = state_encoded.unsqueeze(1).expand(-1, original_query.shape[1], -1)
        fused_query, _ = self.query_fusion(
            query=expanded_query,
            key=state_expanded,
            value=state_expanded
        )
        
        return fused_query

class EndToEndRAGTrainer(nn.Module):
    """
    End-to-end training framework for the complete RAG system.
    """
    
    def __init__(self, config: RetrieverConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.retriever = NeuralRetriever(config)
        self.query_generator = DynamicQueryGenerator(config)
        
        # Language model (placeholder - would be your actual LM)
        self.language_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.query_embed_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Loss functions
        self.retrieval_loss_fn = nn.CrossEntropyLoss()
        self.generation_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
                doc_embeddings: torch.Tensor, task_type: str = 'generation') -> Dict[str, torch.Tensor]:
        """
        Forward pass for end-to-end training.
        
        Args:
            input_ids: [batch_size, seq_len]
            target_ids: [batch_size, target_len]
            doc_embeddings: [num_docs, doc_len, embed_dim]
            task_type: Type of task
            
        Returns:
            Dictionary containing losses and outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Get input embeddings (placeholder - would use actual token embeddings)
        input_embeddings = torch.randn(batch_size, seq_len, self.config.query_embed_dim)
        
        # Dynamic query generation
        processing_state = input_embeddings.mean(dim=1)  # Simplified processing state
        dynamic_query = self.query_generator(input_embeddings, processing_state, task_type)
        
        # Neural retrieval
        retrieval_scores, retrieved_docs = self.retriever(dynamic_query, doc_embeddings)
        
        # Prepare context for language model
        context_embeddings = self._prepare_context(input_embeddings, retrieved_docs)
        
        # Language model generation
        lm_output = self.language_model(
            tgt=context_embeddings,
            memory=context_embeddings
        )
        
        # Calculate losses
        retrieval_loss = self._calculate_retrieval_loss(retrieval_scores, target_ids)
        generation_loss = self._calculate_generation_loss(lm_output, target_ids)
        
        total_loss = retrieval_loss + generation_loss
        
        return {
            'total_loss': total_loss,
            'retrieval_loss': retrieval_loss,
            'generation_loss': generation_loss,
            'retrieval_scores': retrieval_scores,
            'lm_output': lm_output
        }
    
    def _prepare_context(self, input_embeddings: torch.Tensor, retrieved_docs: torch.Tensor) -> torch.Tensor:
        """Prepare context by combining input and retrieved documents."""
        batch_size, seq_len, embed_dim = input_embeddings.shape
        _, top_k, doc_len, _ = retrieved_docs.shape
        
        # Flatten retrieved documents
        retrieved_flat = retrieved_docs.view(batch_size, top_k * doc_len, embed_dim)
        
        # Concatenate input and retrieved context
        context = torch.cat([input_embeddings, retrieved_flat], dim=1)
        
        return context
    
    def _calculate_retrieval_loss(self, retrieval_scores: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Calculate retrieval loss using contrastive learning."""
        # Simplified retrieval loss - in practice, you'd use actual relevance labels
        batch_size, num_candidates = retrieval_scores.shape
        
        # Create dummy positive/negative labels (in practice, these would be real)
        positive_indices = torch.randint(0, num_candidates, (batch_size,))
        labels = torch.zeros(batch_size, num_candidates)
        labels[torch.arange(batch_size), positive_indices] = 1
        
        # Contrastive loss
        loss = F.cross_entropy(retrieval_scores, positive_indices)
        
        return loss
    
    def _calculate_generation_loss(self, lm_output: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Calculate language model generation loss."""
        # Simplified generation loss
        batch_size, seq_len, vocab_size = lm_output.shape
        
        # Create dummy target embeddings (in practice, use actual token embeddings)
        target_embeddings = torch.randn(batch_size, target_ids.shape[1], self.config.query_embed_dim)
        
        # Calculate loss
        loss = F.mse_loss(lm_output[:, :target_embeddings.shape[1]], target_embeddings)
        
        return loss

class ContrastiveRetrievalLoss(nn.Module):
    """
    Contrastive loss for training the neural retriever.
    """
    
    def __init__(self, margin: float = 0.2, temperature: float = 0.05):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss.
        
        Args:
            query_embeddings: [batch_size, embed_dim]
            positive_embeddings: [batch_size, embed_dim]
            negative_embeddings: [batch_size, num_negatives, embed_dim]
            
        Returns:
            contrastive_loss: scalar tensor
        """
        batch_size = query_embeddings.shape[0]
        
        # Calculate positive similarities
        pos_sim = F.cosine_similarity(query_embeddings, positive_embeddings, dim=-1)
        
        # Calculate negative similarities
        neg_sim = F.cosine_similarity(
            query_embeddings.unsqueeze(1),
            negative_embeddings,
            dim=-1
        )  # [batch_size, num_negatives]
        
        # InfoNCE loss
        pos_exp = torch.exp(pos_sim / self.temperature)
        neg_exp = torch.exp(neg_sim / self.temperature).sum(dim=-1)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
        
        return loss

# Training utilities
class RAGTrainingConfig:
    """Configuration for RAG training."""
    def __init__(self):
        self.learning_rate = 1e-4
        self.batch_size = 8
        self.num_epochs = 10
        self.gradient_accumulation_steps = 4
        self.warmup_steps = 100
        self.max_grad_norm = 1.0
        
        # Retrieval training
        self.retrieval_weight = 0.3
        self.generation_weight = 0.7
        
        # Hard negative mining
        self.hard_negative_ratio = 0.5
        self.negative_sampling_strategy = 'random'

def create_training_data(batch_size: int = 4, seq_len: int = 512, num_docs: int = 1000) -> Dict[str, torch.Tensor]:
    """Create sample training data."""
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'target_ids': torch.randint(0, 1000, (batch_size, seq_len // 2)),
        'doc_embeddings': torch.randn(num_docs, 256, 768),
        'positive_doc_ids': torch.randint(0, num_docs, (batch_size,)),
        'negative_doc_ids': torch.randint(0, num_docs, (batch_size, 10))
    }

def test_neural_retriever():
    """Test the neural retriever components."""
    print("🧪 Testing Neural Retriever Components")
    print("=" * 50)
    
    # Create configuration
    config = RetrieverConfig(
        query_embed_dim=768,
        doc_embed_dim=768,
        hidden_dim=512,
        num_candidates=100,
        top_k=8
    )
    
    # Create components
    retriever = NeuralRetriever(config)
    query_generator = DynamicQueryGenerator(config)
    trainer = EndToEndRAGTrainer(config)
    
    # Create sample data
    batch_size, seq_len, embed_dim = 2, 128, 768
    num_docs, doc_len = 100, 256
    
    query_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    doc_embeddings = torch.randn(num_docs, doc_len, embed_dim)
    processing_state = torch.randn(batch_size, embed_dim)
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    
    # Test neural retriever
    with torch.no_grad():
        scores, retrieved_docs = retriever(query_embeddings, doc_embeddings)
        print(f"Retrieval scores shape: {scores.shape}")
        print(f"Retrieved docs shape: {retrieved_docs.shape}")
        
        # Test dynamic query generation
        dynamic_query = query_generator(query_embeddings, processing_state, 'generation')
        print(f"Dynamic query shape: {dynamic_query.shape}")
        
        # Test end-to-end training
        training_data = create_training_data(batch_size=2)
        outputs = trainer(
            training_data['input_ids'],
            training_data['target_ids'],
            training_data['doc_embeddings'],
            'generation'
        )
        print(f"Total loss: {outputs['total_loss'].item():.4f}")
        print(f"Retrieval loss: {outputs['retrieval_loss'].item():.4f}")
        print(f"Generation loss: {outputs['generation_loss'].item():.4f}")
    
    print("✅ Neural retriever components test completed successfully!")

if __name__ == "__main__":
    test_neural_retriever()
