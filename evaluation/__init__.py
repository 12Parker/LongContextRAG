"""
Evaluation Module for Long Context RAG

This module provides comprehensive evaluation tools for comparing RAG systems
against base LLMs across multiple dimensions.
"""

from .rag_evaluation import RAGEvaluator, EvaluationMetrics, EvaluationResult

# Handle missing modules gracefully
try:
    from .quick_rag_evaluation import quick_evaluation
    __all__ = [
        'RAGEvaluator',
        'EvaluationMetrics', 
        'EvaluationResult',
        'quick_evaluation'
    ]
except ImportError:
    __all__ = [
        'RAGEvaluator',
        'EvaluationMetrics', 
        'EvaluationResult'
    ]
