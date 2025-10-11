"""
Training Script for Hybrid Attention RAG System

This script implements the complete training pipeline for the hybrid attention RAG system,
including end-to-end training of neural retrievers and the hierarchical processing framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm

from hybrid.hybrid_attention_rag import HybridAttentionRAG, AttentionConfig
from .neural_retriever import EndToEndRAGTrainer, RetrieverConfig, ContrastiveRetrievalLoss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training the hybrid RAG system."""
    # Model configuration
    attention_config: AttentionConfig
    retriever_config: RetrieverConfig
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 10
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Loss weights
    retrieval_weight: float = 0.3
    generation_weight: float = 0.7
    attention_weight: float = 0.1
    
    # Data parameters
    max_seq_length: int = 2048
    max_retrieved_docs: int = 8
    
    # Optimization
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    scheduler_type: str = 'linear'
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 250
    output_dir: str = './checkpoints'

class RAGDataset(Dataset):
    """
    Dataset for training the hybrid RAG system.
    """
    
    def __init__(self, data_path: str, config: TrainingConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from file."""
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                return json.load(f)
        else:
            # Create sample data for testing
            return self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """Create sample training data."""
        sample_data = []
        for i in range(100):  # 100 sample examples
            sample_data.append({
                'query': f"Sample query {i} about machine learning and AI",
                'context': f"Sample context {i} with relevant information about the topic",
                'answer': f"Sample answer {i} that addresses the query",
                'retrieved_docs': [
                    f"Retrieved document {i}-{j} with relevant information"
                    for j in range(3)
                ],
                'task_type': 'qa' if i % 2 == 0 else 'generation'
            })
        return sample_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        example = self.data[idx]
        
        # Tokenize (simplified - in practice, use actual tokenizer)
        query_tokens = self._tokenize(example['query'])
        context_tokens = self._tokenize(example['context'])
        answer_tokens = self._tokenize(example['answer'])
        
        # Create retrieved document embeddings
        retrieved_docs = []
        for doc in example['retrieved_docs']:
            doc_tokens = self._tokenize(doc)
            retrieved_docs.append(doc_tokens)
        
        return {
            'query_ids': query_tokens,
            'context_ids': context_tokens,
            'answer_ids': answer_tokens,
            'retrieved_docs': retrieved_docs,
            'task_type': example['task_type']
        }
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization (replace with actual tokenizer)."""
        # Convert text to token IDs (simplified)
        tokens = [hash(word) % 1000 for word in text.split()]
        return torch.tensor(tokens[:self.config.max_seq_length], dtype=torch.long)

class HybridRAGTrainer:
    """
    Trainer for the complete hybrid attention RAG system.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize models
        self.attention_model = HybridAttentionRAG(config.attention_config)
        self.retriever_trainer = EndToEndRAGTrainer(config.retriever_config)
        
        # Loss functions
        self.retrieval_loss_fn = ContrastiveRetrievalLoss()
        self.generation_loss_fn = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for all model parameters."""
        all_params = list(self.attention_model.parameters()) + list(self.retriever_trainer.parameters())
        
        return optim.AdamW(
            all_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_epsilon
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                return max(0.1, (self.config.num_epochs * 1000 - step) / (self.config.num_epochs * 1000 - self.config.warmup_steps))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Train the hybrid RAG system."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_loss = self._train_epoch(train_dataloader)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Evaluation
            if eval_dataloader:
                eval_loss = self._evaluate(eval_dataloader)
                logger.info(f"Evaluation loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self._save_checkpoint(epoch, eval_loss)
            
            # Update learning rate
            self.scheduler.step()
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.attention_model.train()
        self.retriever_trainer.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss = self._training_step(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.attention_model.parameters()) + list(self.retriever_trainer.parameters()),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(epoch=0, loss=loss.item())
        
        return total_loss / num_batches
    
    def _training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Single training step."""
        # Prepare inputs
        query_embeddings = self._get_embeddings(batch['query_ids'])
        context_embeddings = self._get_embeddings(batch['context_ids'])
        answer_embeddings = self._get_embeddings(batch['answer_ids'])
        
        # Prepare retrieved documents
        retrieved_docs = [self._get_embeddings(doc) for doc in batch['retrieved_docs']]
        
        # Hybrid attention processing
        attention_output = self.attention_model(
            query_embeddings,
            retrieved_docs
        )
        
        # End-to-end retrieval and generation
        retriever_outputs = self.retriever_trainer(
            batch['query_ids'],
            batch['answer_ids'],
            torch.cat(retrieved_docs, dim=0),
            batch['task_type'][0] if isinstance(batch['task_type'], list) else batch['task_type']
        )
        
        # Calculate losses
        attention_loss = F.mse_loss(attention_output, context_embeddings)
        retrieval_loss = retriever_outputs['retrieval_loss']
        generation_loss = retriever_outputs['generation_loss']
        
        # Combined loss
        total_loss = (
            self.config.attention_weight * attention_loss +
            self.config.retrieval_weight * retrieval_loss +
            self.config.generation_weight * generation_loss
        )
        
        return total_loss
    
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model."""
        self.attention_model.eval()
        self.retriever_trainer.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                loss = self._training_step(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings (simplified)."""
        # In practice, use actual embedding layer
        batch_size, seq_len = token_ids.shape
        return torch.randn(batch_size, seq_len, self.config.attention_config.hidden_size)
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'loss': loss,
            'attention_model_state_dict': self.attention_model.state_dict(),
            'retriever_trainer_state_dict': self.retriever_trainer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f'checkpoint-epoch-{epoch}-step-{self.global_step}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

def create_training_config() -> TrainingConfig:
    """Create training configuration."""
    attention_config = AttentionConfig(
        window_size=512,
        num_landmark_tokens=32,
        max_retrieved_segments=8,
        hidden_size=768,
        num_attention_heads=12
    )
    
    retriever_config = RetrieverConfig(
        query_embed_dim=768,
        doc_embed_dim=768,
        hidden_dim=512,
        num_candidates=100,
        top_k=8
    )
    
    return TrainingConfig(
        attention_config=attention_config,
        retriever_config=retriever_config,
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=5,
        output_dir='./checkpoints'
    )

def main():
    """Main training function."""
    logger.info("Initializing Hybrid Attention RAG Training")
    
    # Create configuration
    config = create_training_config()
    
    # Create datasets
    train_dataset = RAGDataset('data/train_data.json', config)
    eval_dataset = RAGDataset('data/eval_data.json', config)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create trainer
    trainer = HybridRAGTrainer(config)
    
    # Train
    trainer.train(train_dataloader, eval_dataloader)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
