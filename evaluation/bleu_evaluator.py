#!/usr/bin/env python3
"""
BLEU Score Evaluator for NarrativeQA

This module provides comprehensive evaluation metrics including BLEU score,
ROUGE score, and other text similarity measures for comparing generated
answers against reference answers.
"""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BLEUEvaluator:
    """
    Comprehensive evaluator for NarrativeQA answers using multiple metrics.
    """
    
    def __init__(self):
        """Initialize the BLEU evaluator."""
        self._nltk_available = self._check_nltk_availability()
        self._rouge_available = self._check_rouge_availability()
    
    def _check_nltk_availability(self) -> bool:
        """Check if NLTK is available."""
        try:
            import nltk
            return True
        except ImportError:
            return False
    
    def _check_rouge_availability(self) -> bool:
        """Check if ROUGE is available."""
        try:
            from rouge_score import rouge_scorer
            return True
        except ImportError:
            return False
    
    def evaluate_answer(self, generated_answer: str, reference_answers: List[str]) -> Dict[str, float]:
        """
        Evaluate a generated answer against reference answers using multiple metrics.
        
        Args:
            generated_answer: The generated answer to evaluate
            reference_answers: List of reference answers
            
        Returns:
            Dictionary containing various evaluation metrics
        """
        if not reference_answers:
            return {
                'bleu_score': 0.0,
                'rouge_l': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'word_overlap': 0.0,
                'exact_match': 0.0
            }
        
        metrics = {}
        
        # BLEU Score
        metrics['bleu_score'] = self._calculate_bleu_score(generated_answer, reference_answers)
        
        # ROUGE Scores
        if self._rouge_available:
            rouge_scores = self._calculate_rouge_scores(generated_answer, reference_answers)
            metrics.update(rouge_scores)
        else:
            metrics.update({
                'rouge_l': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0
            })
        
        # Word Overlap (fallback metric)
        metrics['word_overlap'] = self._calculate_word_overlap(generated_answer, reference_answers)
        
        # Exact Match
        metrics['exact_match'] = self._calculate_exact_match(generated_answer, reference_answers)
        
        return metrics
    
    def _calculate_bleu_score(self, generated_answer: str, reference_answers: List[str]) -> float:
        """Calculate BLEU score for the generated answer."""
        if not self._nltk_available:
            logger.warning("NLTK not available, BLEU score will be 0.0")
            return 0.0
        
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Prepare reference answers for BLEU calculation
            references = [ref.split() for ref in reference_answers]
            candidate = generated_answer.split()
            
            # Use smoothing function to handle cases with no n-gram matches
            smoothing = SmoothingFunction().method1
            
            # Calculate BLEU score
            bleu_score = sentence_bleu(
                references, 
                candidate, 
                smoothing_function=smoothing
            )
            
            return bleu_score
            
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return 0.0
    
    def _calculate_rouge_scores(self, generated_answer: str, reference_answers: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores for the generated answer."""
        if not self._rouge_available:
            return {
                'rouge_l': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0
            }
        
        try:
            from rouge_score import rouge_scorer
            
            # Initialize ROUGE scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # Calculate ROUGE scores against all reference answers
            rouge_scores = []
            for ref_answer in reference_answers:
                scores = scorer.score(ref_answer, generated_answer)
                rouge_scores.append({
                    'rouge_1': scores['rouge1'].fmeasure,
                    'rouge_2': scores['rouge2'].fmeasure,
                    'rouge_l': scores['rougeL'].fmeasure
                })
            
            # Return the maximum scores across all references
            return {
                'rouge_1': max(score['rouge_1'] for score in rouge_scores),
                'rouge_2': max(score['rouge_2'] for score in rouge_scores),
                'rouge_l': max(score['rouge_l'] for score in rouge_scores)
            }
            
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return {
                'rouge_l': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0
            }
    
    def _calculate_word_overlap(self, generated_answer: str, reference_answers: List[str]) -> float:
        """Calculate word overlap score (fallback metric)."""
        max_overlap = 0.0
        
        for ref_answer in reference_answers:
            ref_words = set(ref_answer.lower().split())
            gen_words = set(generated_answer.lower().split())
            overlap = len(ref_words.intersection(gen_words))
            
            if len(ref_words) > 0:
                overlap_score = overlap / len(ref_words)
                max_overlap = max(max_overlap, overlap_score)
        
        return max_overlap
    
    def _calculate_exact_match(self, generated_answer: str, reference_answers: List[str]) -> float:
        """Calculate exact match score."""
        generated_clean = self._clean_text(generated_answer)
        
        for ref_answer in reference_answers:
            ref_clean = self._clean_text(ref_answer)
            if generated_clean == ref_clean:
                return 1.0
        
        return 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean text for exact match comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
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
        metrics = ['bleu_score', 'rouge_l', 'rouge_1', 'rouge_2', 'word_overlap', 'exact_match']
        
        for metric in metrics:
            values = [result.get(metric, 0.0) for result in results]
            summary[f'avg_{metric}'] = sum(values) / len(values)
            summary[f'max_{metric}'] = max(values)
            summary[f'min_{metric}'] = min(values)
        
        return summary

def main():
    """Test the BLEU evaluator."""
    evaluator = BLEUEvaluator()
    
    # Test with sample data
    generated_answer = "Mark Hunter is a high school student in Phoenix who runs a pirate radio station."
    reference_answers = [
        "He is a high school student in Phoenix.",
        "Mark Hunter is a student who operates a radio station."
    ]
    
    print("🧪 Testing BLEU Evaluator")
    print("=" * 50)
    
    # Check availability
    print(f"NLTK available: {evaluator._nltk_available}")
    print(f"ROUGE available: {evaluator._rouge_available}")
    
    # Evaluate
    metrics = evaluator.evaluate_answer(generated_answer, reference_answers)
    
    print(f"\nGenerated Answer: {generated_answer}")
    print(f"Reference Answers: {reference_answers}")
    print(f"\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
