#!/usr/bin/env python3
"""
Training Script for Neural Retriever

This script prepares training data from NarrativeQA and trains the neural retriever
to improve retrieval performance.

Usage:
    python scripts/train_neural_retriever.py --num-examples 1000 --epochs 10
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any
import numpy as np
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from langchain.embeddings import HuggingFaceEmbeddings
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from hybrid.narrativeqa_hybrid_rag_neural_retriever import NarrativeQAHybridRAG
from core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_training_data_from_narrativeqa(
    num_examples: int = 1000,
    chunk_size: int = 2000,
    chunk_overlap: int = 400,
    subset: str = "train"
) -> List[Dict[str, Any]]:
    """
    Prepare training data from NarrativeQA dataset.
    
    For each question-answer pair, we:
    1. Find chunks that contain the answer (positive examples)
    2. Find chunks that don't contain the answer (negative examples)
    
    Args:
        num_examples: Number of examples to prepare (-1 for all examples)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        subset: Dataset subset to use ('train', 'validation', 'test')
        
    Returns:
        List of training examples
    """
    logger.info(f"Loading NarrativeQA dataset ({subset} split)...")
    dataset = load_dataset("narrativeqa", split=subset)
    
    training_data = []
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name="text-embedding-3-large"
    )
    
    # Initialize embeddings for semantic similarity matching
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    total_examples = len(dataset)
    if num_examples == -1:
        num_examples = total_examples
        logger.info(f"Using full dataset: {total_examples} examples")
    else:
        num_examples = min(num_examples, total_examples)
        logger.info(f"Preparing {num_examples} training examples (from {total_examples} total)...")
    
    # Create progress bar for data preparation
    data_pbar = tqdm(
        enumerate(dataset),
        total=num_examples,
        desc="Preparing training data",
        unit="example",
        disable=not TQDM_AVAILABLE
    )
    
    for i, example in data_pbar:
        if i >= num_examples:
            break
        
        # Get story text
        story_data = example.get('document', '')
        if isinstance(story_data, dict):
            story_text = story_data.get('text', '')
        else:
            story_text = str(story_data)
        
        if not story_text:
            continue
        
        # Get question and answers
        question = example.get('question', {}).get('text', '')
        answers = example.get('answers', [])
        if not question or not answers:
            continue
        
        # Split story into chunks
        chunks = text_splitter.split_text(story_text)
        
        # Find positive chunks using multiple methods:
        # 1. Exact text matching (strict)
        # 2. Semantic similarity (embeddings)
        # 3. Key term matching (fuzzy)
        positive_chunks = []
        negative_chunks = []
        
        answer_texts = [ans.get('text', '').strip() for ans in answers if ans.get('text')]
        if not answer_texts:
            continue
        
        # Get embeddings for all answers (for semantic similarity)
        try:
            answer_embeddings = embeddings.embed_documents(answer_texts)
            answer_emb_matrix = np.array(answer_embeddings)
        except Exception as e:
            logger.warning(f"Failed to get answer embeddings: {e}, falling back to text matching")
            answer_emb_matrix = None
        
        # Get embeddings for all chunks
        try:
            chunk_embeddings = embeddings.embed_documents(chunks)
            chunk_emb_matrix = np.array(chunk_embeddings)
        except Exception as e:
            logger.warning(f"Failed to get chunk embeddings: {e}, falling back to text matching")
            chunk_emb_matrix = None
        
        # Extract key terms from answers (for fuzzy matching)
        def extract_key_terms(text: str) -> set:
            """Extract important terms from answer text."""
            import re
            # Remove common stop words and get meaningful terms
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'is', 'are', 'was', 'were'}
            words = re.findall(r'\b\w+\b', text.lower())
            return {w for w in words if len(w) > 2 and w not in stop_words}
        
        answer_key_terms = set()
        for answer_text in answer_texts:
            answer_key_terms.update(extract_key_terms(answer_text))
        
        # Score each chunk
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = 0.0
            is_positive = False
            
            # Method 1: Exact text matching (highest confidence)
            for answer_text in answer_texts:
                if answer_text.lower() in chunk_lower:
                    score = 1.0
                    is_positive = True
                    break
            
            # Method 2: Semantic similarity (if embeddings available)
            if not is_positive and answer_emb_matrix is not None and chunk_emb_matrix is not None:
                chunk_emb = chunk_emb_matrix[i]
                # Calculate cosine similarity with all answer embeddings
                similarities = []
                for answer_emb in answer_emb_matrix:
                    # Cosine similarity
                    dot_product = np.dot(chunk_emb, answer_emb)
                    norm_chunk = np.linalg.norm(chunk_emb)
                    norm_answer = np.linalg.norm(answer_emb)
                    if norm_chunk > 0 and norm_answer > 0:
                        similarity = dot_product / (norm_chunk * norm_answer)
                        similarities.append(similarity)
                
                if similarities:
                    max_similarity = max(similarities)
                    # Threshold for semantic similarity (tuned for all-MiniLM-L6-v2)
                    if max_similarity > 0.65:  # Adjustable threshold
                        score = max(score, max_similarity)
                        is_positive = True
            
            # Method 3: Key term matching (fuzzy)
            if not is_positive:
                chunk_key_terms = extract_key_terms(chunk)
                if answer_key_terms:
                    term_overlap = len(chunk_key_terms.intersection(answer_key_terms))
                    term_ratio = term_overlap / len(answer_key_terms)
                    # If chunk contains most key terms, it's likely relevant
                    if term_ratio > 0.4:  # At least 40% of key terms present
                        score = max(score, term_ratio * 0.8)  # Lower confidence than exact match
                        is_positive = True
            
            chunk_scores.append((chunk, score, is_positive))
        
        # Separate positive and negative chunks
        for chunk, score, is_pos in chunk_scores:
            if is_pos:
                positive_chunks.append((chunk, score))
            else:
                negative_chunks.append(chunk)
        
        # Sort positive chunks by score (highest first)
        positive_chunks.sort(key=lambda x: x[1], reverse=True)
        positive_chunks = [chunk for chunk, score in positive_chunks]  # Remove scores
        
        # Skip if no positive chunks found
        if not positive_chunks:
            continue
        
        # Limit number of chunks for efficiency
        positive_chunks = positive_chunks[:5]  # Top 5 positive chunks
        negative_chunks = negative_chunks[:10]  # Top 10 negative chunks
        
        training_data.append({
            'question': question,
            'positive_chunks': positive_chunks,
            'negative_chunks': negative_chunks,
            'story_id': example.get('document', {}).get('id', f'story_{i}'),
            'question_id': example.get('question', {}).get('id', f'q_{i}')
        })
        
        # Update progress bar
        data_pbar.set_postfix({
            'prepared': len(training_data),
            'pos_chunks': len(positive_chunks),
            'neg_chunks': len(negative_chunks)
        })
    
    data_pbar.close()
    logger.info(f"Prepared {len(training_data)} training examples")
    return training_data


def main():
    parser = argparse.ArgumentParser(description="Train Neural Retriever")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=500,
        help="Number of training examples to prepare (-1 for full dataset)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset subset to use for training (default: train)"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation (0.0-1.0, default: 0.1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load (optional)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size for text splitting"
    )
    
    args = parser.parse_args()
    
    # Prepare training data
    logger.info("Preparing training data from NarrativeQA...")
    all_training_data = prepare_training_data_from_narrativeqa(
        num_examples=args.num_examples,
        chunk_size=args.chunk_size,
        subset=args.subset
    )
    
    if not all_training_data:
        logger.error("No training data prepared. Exiting.")
        return
    
    # Split into train and validation
    if args.validation_split > 0.0 and len(all_training_data) > 10:
        split_idx = int(len(all_training_data) * (1 - args.validation_split))
        training_data = all_training_data[:split_idx]
        validation_data = all_training_data[split_idx:]
        logger.info(f"Split data: {len(training_data)} train, {len(validation_data)} validation")
    else:
        training_data = all_training_data
        validation_data = []
        logger.info(f"Using all {len(training_data)} examples for training (no validation split)")
    
    # Initialize RAG system
    logger.info("Initializing Hybrid RAG system...")
    hybrid_rag = NarrativeQAHybridRAG(
        chunk_size=args.chunk_size,
        use_hybrid_attention=True
    )
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        hybrid_rag.load_retriever_checkpoint(args.load_checkpoint)
    
    # Train neural retriever
    logger.info("Starting training...")
    hybrid_rag.train_neural_retriever(
        training_data=training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        save_best=True,
        validation_data=validation_data if validation_data else None
    )
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to {args.checkpoint_dir}/neural_retriever_best.pt")


if __name__ == "__main__":
    main()

