#!/usr/bin/env python3
"""
QA-Specific Evaluation Metrics

This module provides comprehensive evaluation metrics specifically designed
for question-answering tasks, including Exact Match, F1 Score, BERTScore,
and METEOR.
"""

import re
import string
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class QAMetrics:
    """
    Comprehensive QA evaluation metrics including EM, F1, BERTScore, and METEOR.
    """
    
    def __init__(self):
        """Initialize the QA metrics evaluator."""
        self._bert_score_available = self._check_bert_score_availability()
        self._meteor_available = self._check_meteor_availability()
    
    def _check_bert_score_availability(self) -> bool:
        """Check if BERTScore is available."""
        try:
            from bert_score import score
            return True
        except ImportError:
            return False
    
    def _check_meteor_availability(self) -> bool:
        """Check if METEOR is available."""
        try:
            from nltk.translate.meteor_score import meteor_score
            return True
        except ImportError:
            return False
    
    def evaluate_answer(self, generated_answer: str, reference_answers: List[str]) -> Dict[str, float]:
        """
        Evaluate a generated answer against reference answers using QA-specific metrics.
        
        Args:
            generated_answer: The generated answer to evaluate
            reference_answers: List of reference answers
            
        Returns:
            Dictionary containing various QA evaluation metrics
        """
        if not reference_answers:
            return {
                'exact_match': 0.0,
                'f1_score': 0.0,
                'bert_score': 0.0,
                'meteor_score': 0.0,
                'token_precision': 0.0,
                'token_recall': 0.0
            }
        
        metrics = {}
        
        # Exact Match
        metrics['exact_match'] = self._calculate_exact_match(generated_answer, reference_answers)
        
        # F1 Score
        f1_scores = []
        for ref_answer in reference_answers:
            f1 = self._calculate_f1_score(generated_answer, ref_answer)
            f1_scores.append(f1)
        metrics['f1_score'] = max(f1_scores) if f1_scores else 0.0
        
        # BERTScore
        if self._bert_score_available:
            metrics['bert_score'] = self._calculate_bert_score(generated_answer, reference_answers)
        else:
            metrics['bert_score'] = 0.0
        
        # METEOR Score
        if self._meteor_available:
            metrics['meteor_score'] = self._calculate_meteor_score(generated_answer, reference_answers)
        else:
            metrics['meteor_score'] = 0.0
        
        # Token-level metrics
        token_metrics = self._calculate_token_metrics(generated_answer, reference_answers)
        metrics.update(token_metrics)
        
        return metrics
    
    def _calculate_exact_match(self, generated_answer: str, reference_answers: List[str]) -> float:
        """Calculate exact match score."""
        generated_clean = self._normalize_text(generated_answer)
        
        for ref_answer in reference_answers:
            ref_clean = self._normalize_text(ref_answer)
            if generated_clean == ref_clean:
                return 1.0
        
        return 0.0
    
    def _calculate_f1_score(self, generated_answer: str, reference_answer: str) -> float:
        """Calculate F1 score between generated and reference answers."""
        gen_tokens = self._get_tokens(generated_answer)
        ref_tokens = self._get_tokens(reference_answer)
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Calculate precision and recall
        gen_tokens_set = set(gen_tokens)
        ref_tokens_set = set(ref_tokens)
        
        # True positives: tokens that appear in both
        true_positives = len(gen_tokens_set.intersection(ref_tokens_set))
        
        # Precision: true_positives / total_generated_tokens
        precision = true_positives / len(gen_tokens_set) if gen_tokens_set else 0.0
        
        # Recall: true_positives / total_reference_tokens
        recall = true_positives / len(ref_tokens_set) if ref_tokens_set else 0.0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _calculate_bert_score(self, generated_answer: str, reference_answers: List[str]) -> float:
        """Calculate BERTScore for semantic similarity."""
        if not self._bert_score_available:
            return 0.0
        
        try:
            from bert_score import score
            
            # BERTScore expects lists of strings
            generated_list = [generated_answer]
            reference_list = [reference_answers[0]]  # Use first reference for BERTScore
            
            # Calculate BERTScore
            P, R, F1 = score(generated_list, reference_list, lang="en", verbose=False)
            
            # Return F1 score (harmonic mean of precision and recall)
            return F1.item()
            
        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            return 0.0
    
    def _calculate_meteor_score(self, generated_answer: str, reference_answers: List[str]) -> float:
        """Calculate METEOR score for alignment."""
        if not self._meteor_available:
            return 0.0
        
        try:
            from nltk.translate.meteor_score import meteor_score
            
            # METEOR expects tokenized text
            generated_tokens = self._get_tokens(generated_answer)
            reference_tokens_list = [self._get_tokens(ref) for ref in reference_answers]
            
            # Calculate METEOR score
            meteor = meteor_score(reference_tokens_list, generated_tokens)
            return meteor
            
        except Exception as e:
            logger.error(f"METEOR calculation failed: {e}")
            return 0.0
    
    def _calculate_token_metrics(self, generated_answer: str, reference_answers: List[str]) -> Dict[str, float]:
        """Calculate token-level precision and recall."""
        gen_tokens = self._get_tokens(generated_answer)
        
        if not gen_tokens:
            return {'token_precision': 0.0, 'token_recall': 0.0}
        
        # Calculate metrics against all references
        max_precision = 0.0
        max_recall = 0.0
        
        for ref_answer in reference_answers:
            ref_tokens = self._get_tokens(ref_answer)
            if not ref_tokens:
                continue
            
            gen_tokens_set = set(gen_tokens)
            ref_tokens_set = set(ref_tokens)
            
            # True positives: tokens that appear in both
            true_positives = len(gen_tokens_set.intersection(ref_tokens_set))
            
            # Precision: true_positives / total_generated_tokens
            precision = true_positives / len(gen_tokens_set) if gen_tokens_set else 0.0
            
            # Recall: true_positives / total_reference_tokens
            recall = true_positives / len(ref_tokens_set) if ref_tokens_set else 0.0
            
            max_precision = max(max_precision, precision)
            max_recall = max(max_recall, recall)
        
        return {
            'token_precision': max_precision,
            'token_recall': max_recall
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for exact match comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _get_tokens(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        # Simple tokenization (can be improved with proper tokenizers)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def get_metrics_summary(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate summary statistics for a list of evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary containing summary statistics
        """
        if not results:
            return {}
        
        summary = {}
        
        # Calculate averages for each metric
        metrics = ['exact_match', 'f1_score', 'bert_score', 'meteor_score', 
                 'token_precision', 'token_recall']
        
        for metric in metrics:
            values = [result.get(metric, 0.0) for result in results]
            summary[f'avg_{metric}'] = sum(values) / len(values)
            summary[f'max_{metric}'] = max(values)
            summary[f'min_{metric}'] = min(values)
        
        return summary

def main():
    """Test the QA metrics evaluator."""
    evaluator = QAMetrics()
    
    # Test with sample data
    generated_answer = "Mark Hunter is a high school student in Phoenix who runs a pirate radio station."
    reference_answers = [
        "He is a high school student in Phoenix.",
        "Mark Hunter is a student who operates a radio station."
    ]
    
    print("🧪 Testing QA Metrics Evaluator")
    print("=" * 50)
    
    # Check availability
    print(f"BERTScore available: {evaluator._bert_score_available}")
    print(f"METEOR available: {evaluator._meteor_available}")
    
    # Evaluate
    metrics = evaluator.evaluate_answer(generated_answer, reference_answers)
    
    print(f"\nGenerated Answer: {generated_answer}")
    print(f"Reference Answers: {reference_answers}")
    print(f"\nQA Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
