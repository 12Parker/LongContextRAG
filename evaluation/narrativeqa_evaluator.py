#!/usr/bin/env python3
"""
NarrativeQA Benchmark Integration

This module integrates the NarrativeQA benchmark for evaluating RAG systems
on complex, long-form question-answering tasks using narrative text.

NarrativeQA Dataset:
- 1,567 stories (books and movie scripts)
- 46,765 question-answer pairs
- Focuses on deep reading comprehension
- Requires integrative reasoning over long narratives

Usage:
    python evaluation/narrativeqa_evaluator.py
    python evaluation/narrativeqa_evaluator.py --subset test --num-questions 100
"""

import sys
import os
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import RAG systems - handle missing modules gracefully
WorkingHybridRAG = None
ContextSize = None
StandardRAGBaseline = None
BookCorpusRawLLMBaseline = None

try:
    from hybrid.working_hybrid_rag import WorkingHybridRAG
    from core.long_context_config import LongContextManager, ContextSize
except ImportError:
    logger.warning("Hybrid RAG not available, using fallback")

try:
    from examples.standard_rag_baseline import StandardRAGBaseline
except ImportError:
    logger.warning("Standard RAG baseline not available")

try:
    from examples.bookcorpus_raw_llm_baseline import BookCorpusRawLLMBaseline
except ImportError:
    logger.warning("Raw LLM baseline not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NarrativeQAMetrics:
    """Metrics for evaluating NarrativeQA performance."""
    
    # Question-Answer metrics
    question_id: str
    story_id: str
    question: str
    reference_answers: List[str]
    generated_answer: str
    
    # Evaluation scores
    bleu_score: float = 0.0
    rouge_l_score: float = 0.0
    meteor_score: float = 0.0
    exact_match: bool = False
    
    # System metrics
    response_time: float = 0.0
    context_length: int = 0
    retrieved_chunks: int = 0
    method: str = ""
    
    # Quality metrics
    answer_length: int = 0
    answer_quality_score: float = 0.0

@dataclass
class NarrativeQAResult:
    """Complete evaluation result for NarrativeQA."""
    
    question_id: str
    story_id: str
    question: str
    reference_answers: List[str]
    generated_answer: str
    
    # System comparisons
    raw_llm_metrics: NarrativeQAMetrics
    standard_rag_metrics: NarrativeQAMetrics
    hybrid_rag_metrics: NarrativeQAMetrics
    
    # Overall scores
    best_system: str = ""
    improvement_over_baseline: float = 0.0
    timestamp: str = ""

class NarrativeQAEvaluator:
    """Evaluator for NarrativeQA benchmark integration."""
    
    def __init__(self, 
                 db_path: str = "./full_bookcorpus_db",
                 max_questions: int = 100,
                 subset: str = "test"):
        """
        Initialize NarrativeQA evaluator.
        
        Args:
            db_path: Path to the vector database
            max_questions: Maximum number of questions to evaluate
            subset: Dataset subset ('train', 'test', 'validation')
        """
        self.db_path = db_path
        self.max_questions = max_questions
        self.subset = subset
        
        # Initialize systems
        self.raw_llm = None
        self.standard_rag = None
        self.hybrid_rag = None
        
        # Results storage
        self.results = []
        
        # Evaluation metrics
        self.bleu_scores = []
        self.rouge_scores = []
        self.meteor_scores = []
        
        logger.info(f"Initialized NarrativeQA evaluator for {subset} subset")
        logger.info(f"Max questions: {max_questions}")
    
    def load_narrativeqa_dataset(self) -> List[Dict[str, Any]]:
        """Load NarrativeQA dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            logger.info("Loading NarrativeQA dataset...")
            
            # Load the dataset
            dataset = load_dataset("narrativeqa", split=self.subset)
            
            # Convert to list and limit size
            questions = []
            for i, example in enumerate(dataset):
                if i >= self.max_questions:
                    break
                
                # Handle different dataset structures
                question_text = example.get('question', {})
                if isinstance(question_text, dict):
                    question_text = question_text.get('text', '')
                
                answers = example.get('answers', [])
                if isinstance(answers, list) and len(answers) > 0:
                    if isinstance(answers[0], dict):
                        answers = [ans.get('text', '') for ans in answers]
                
                questions.append({
                    'question_id': example.get('question_id', f"q_{i}"),
                    'story_id': example.get('story_id', f"story_{i}"),
                    'question': question_text,
                    'answers': answers,
                    'story': example.get('story', ''),
                    'summary': example.get('summary', '')
                })
            
            logger.info(f"Loaded {len(questions)} questions from NarrativeQA {self.subset} subset")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to load NarrativeQA dataset: {e}")
            return []
    
    def initialize_systems(self):
        """Initialize all RAG systems for evaluation."""
        logger.info("Initializing RAG systems...")
        
        # Initialize Raw LLM baseline
        if BookCorpusRawLLMBaseline is not None:
            try:
                self.raw_llm = BookCorpusRawLLMBaseline(
                    max_context_tokens=50000,
                    num_books=10
                )
                logger.info("✅ Raw LLM baseline initialized")
            except Exception as e:
                logger.warning(f"Raw LLM baseline failed to initialize: {e}")
                self.raw_llm = None
        else:
            logger.warning("Raw LLM baseline not available")
            self.raw_llm = None
        
        # Initialize Standard RAG baseline
        if StandardRAGBaseline is not None:
            try:
                self.standard_rag = StandardRAGBaseline(
                    db_path=self.db_path,
                    top_k_results=10
                )
                logger.info("✅ Standard RAG baseline initialized")
            except Exception as e:
                logger.warning(f"Standard RAG baseline failed to initialize: {e}")
                self.standard_rag = None
        else:
            logger.warning("Standard RAG baseline not available")
            self.standard_rag = None
        
        # Initialize Hybrid RAG
        if WorkingHybridRAG is not None and ContextSize is not None:
            try:
                self.hybrid_rag = WorkingHybridRAG(
                    db_path=self.db_path,
                    context_size=ContextSize.LARGE
                )
                logger.info("✅ Hybrid RAG initialized")
            except Exception as e:
                logger.warning(f"Hybrid RAG failed to initialize: {e}")
                self.hybrid_rag = None
        else:
            logger.warning("Hybrid RAG not available")
            self.hybrid_rag = None
    
    def evaluate_question(self, question_data: Dict[str, Any]) -> NarrativeQAResult:
        """Evaluate a single NarrativeQA question across all systems."""
        question_id = question_data['question_id']
        question = question_data['question']
        reference_answers = question_data['answers']
        
        logger.info(f"Evaluating question {question_id}: {question[:60]}...")
        
        # Evaluate Raw LLM
        raw_llm_metrics = self._evaluate_raw_llm(question_data) if self.raw_llm else self._create_error_metrics(question_data, 'raw_llm')
        
        # Evaluate Standard RAG
        standard_rag_metrics = self._evaluate_standard_rag(question_data) if self.standard_rag else self._create_error_metrics(question_data, 'standard_rag')
        
        # Evaluate Hybrid RAG
        hybrid_rag_metrics = self._evaluate_hybrid_rag(question_data) if self.hybrid_rag else self._create_error_metrics(question_data, 'hybrid_rag')
        
        # Determine best system
        best_system = self._determine_best_system(
            raw_llm_metrics, standard_rag_metrics, hybrid_rag_metrics
        )
        
        # Calculate improvement
        improvement = self._calculate_improvement(
            raw_llm_metrics, standard_rag_metrics, hybrid_rag_metrics
        )
        
        result = NarrativeQAResult(
            question_id=question_id,
            story_id=question_data['story_id'],
            question=question,
            reference_answers=reference_answers,
            generated_answer=hybrid_rag_metrics.generated_answer if self.hybrid_rag else "System not available",
            raw_llm_metrics=raw_llm_metrics,
            standard_rag_metrics=standard_rag_metrics,
            hybrid_rag_metrics=hybrid_rag_metrics,
            best_system=best_system,
            improvement_over_baseline=improvement,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def _evaluate_raw_llm(self, question_data: Dict[str, Any]) -> NarrativeQAMetrics:
        """Evaluate using Raw LLM baseline."""
        start_time = time.time()
        
        try:
            response = self.raw_llm.generate_response(question_data['question'])
            generated_answer = response['response']
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            bleu_score = self._calculate_bleu(generated_answer, question_data['answers'])
            rouge_score = self._calculate_rouge(generated_answer, question_data['answers'])
            meteor_score = self._calculate_meteor(generated_answer, question_data['answers'])
            exact_match = self._check_exact_match(generated_answer, question_data['answers'])
            
            return NarrativeQAMetrics(
                question_id=question_data['question_id'],
                story_id=question_data['story_id'],
                question=question_data['question'],
                reference_answers=question_data['answers'],
                generated_answer=generated_answer,
                bleu_score=bleu_score,
                rouge_l_score=rouge_score,
                meteor_score=meteor_score,
                exact_match=exact_match,
                response_time=elapsed_time,
                context_length=response.get('context_length', 0),
                retrieved_chunks=0,
                method='raw_llm',
                answer_length=len(generated_answer),
                answer_quality_score=(bleu_score + rouge_score + meteor_score) / 3
            )
            
        except Exception as e:
            logger.error(f"Raw LLM evaluation failed: {e}")
            return self._create_error_metrics(question_data, 'raw_llm')
    
    def _evaluate_standard_rag(self, question_data: Dict[str, Any]) -> NarrativeQAMetrics:
        """Evaluate using Standard RAG baseline."""
        start_time = time.time()
        
        try:
            response = self.standard_rag.generate_response(question_data['question'])
            generated_answer = response['response']
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            bleu_score = self._calculate_bleu(generated_answer, question_data['answers'])
            rouge_score = self._calculate_rouge(generated_answer, question_data['answers'])
            meteor_score = self._calculate_meteor(generated_answer, question_data['answers'])
            exact_match = self._check_exact_match(generated_answer, question_data['answers'])
            
            return NarrativeQAMetrics(
                question_id=question_data['question_id'],
                story_id=question_data['story_id'],
                question=question_data['question'],
                reference_answers=question_data['answers'],
                generated_answer=generated_answer,
                bleu_score=bleu_score,
                rouge_l_score=rouge_score,
                meteor_score=meteor_score,
                exact_match=exact_match,
                response_time=elapsed_time,
                context_length=response.get('context_length', 0),
                retrieved_chunks=response.get('chunks_retrieved', 0),
                method='standard_rag',
                answer_length=len(generated_answer),
                answer_quality_score=(bleu_score + rouge_score + meteor_score) / 3
            )
            
        except Exception as e:
            logger.error(f"Standard RAG evaluation failed: {e}")
            return self._create_error_metrics(question_data, 'standard_rag')
    
    def _evaluate_hybrid_rag(self, question_data: Dict[str, Any]) -> NarrativeQAMetrics:
        """Evaluate using Hybrid RAG."""
        start_time = time.time()
        
        try:
            response = self.hybrid_rag.generate_response(question_data['question'])
            generated_answer = response['response']
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            bleu_score = self._calculate_bleu(generated_answer, question_data['answers'])
            rouge_score = self._calculate_rouge(generated_answer, question_data['answers'])
            meteor_score = self._calculate_meteor(generated_answer, question_data['answers'])
            exact_match = self._check_exact_match(generated_answer, question_data['answers'])
            
            return NarrativeQAMetrics(
                question_id=question_data['question_id'],
                story_id=question_data['story_id'],
                question=question_data['question'],
                reference_answers=question_data['answers'],
                generated_answer=generated_answer,
                bleu_score=bleu_score,
                rouge_l_score=rouge_score,
                meteor_score=meteor_score,
                exact_match=exact_match,
                response_time=elapsed_time,
                context_length=response.get('context_length', 0),
                retrieved_chunks=response.get('retrieved_docs', 0),
                method='hybrid_rag',
                answer_length=len(generated_answer),
                answer_quality_score=(bleu_score + rouge_score + meteor_score) / 3
            )
            
        except Exception as e:
            logger.error(f"Hybrid RAG evaluation failed: {e}")
            return self._create_error_metrics(question_data, 'hybrid_rag')
    
    def _calculate_bleu(self, generated: str, references: List[str]) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            # Tokenize references
            ref_tokens = [word_tokenize(ref.lower()) for ref in references]
            gen_tokens = word_tokenize(generated.lower())
            
            # Calculate BLEU score
            bleu = sentence_bleu(ref_tokens, gen_tokens)
            return float(bleu)
            
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return 0.0
    
    def _calculate_rouge(self, generated: str, references: List[str]) -> float:
        """Calculate ROUGE-L score."""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            # Calculate ROUGE-L for each reference and take the maximum
            rouge_scores = []
            for ref in references:
                scores = scorer.score(ref, generated)
                rouge_scores.append(scores['rougeL'].fmeasure)
            
            return max(rouge_scores) if rouge_scores else 0.0
            
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return 0.0
    
    def _calculate_meteor(self, generated: str, references: List[str]) -> float:
        """Calculate METEOR score."""
        try:
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize
            
            # Tokenize references
            ref_tokens = [word_tokenize(ref.lower()) for ref in references]
            gen_tokens = word_tokenize(generated.lower())
            
            # Calculate METEOR score
            meteor = meteor_score(ref_tokens, gen_tokens)
            return float(meteor)
            
        except Exception as e:
            logger.warning(f"METEOR calculation failed: {e}")
            return 0.0
    
    def _check_exact_match(self, generated: str, references: List[str]) -> bool:
        """Check for exact match with any reference answer."""
        generated_lower = generated.lower().strip()
        for ref in references:
            if generated_lower == ref.lower().strip():
                return True
        return False
    
    def _create_error_metrics(self, question_data: Dict[str, Any], method: str) -> NarrativeQAMetrics:
        """Create error metrics for failed evaluations."""
        return NarrativeQAMetrics(
            question_id=question_data['question_id'],
            story_id=question_data['story_id'],
            question=question_data['question'],
            reference_answers=question_data['answers'],
            generated_answer="",
            method=method,
            response_time=0.0,
            context_length=0,
            retrieved_chunks=0,
            answer_length=0,
            answer_quality_score=0.0
        )
    
    def _determine_best_system(self, raw_llm: NarrativeQAMetrics, 
                              standard_rag: NarrativeQAMetrics, 
                              hybrid_rag: NarrativeQAMetrics) -> str:
        """Determine which system performed best."""
        scores = {
            'raw_llm': raw_llm.answer_quality_score,
            'standard_rag': standard_rag.answer_quality_score,
            'hybrid_rag': hybrid_rag.answer_quality_score
        }
        
        return max(scores, key=scores.get)
    
    def _calculate_improvement(self, raw_llm: NarrativeQAMetrics, 
                             standard_rag: NarrativeQAMetrics, 
                             hybrid_rag: NarrativeQAMetrics) -> float:
        """Calculate improvement over baseline."""
        baseline_score = raw_llm.answer_quality_score
        best_score = max(
            standard_rag.answer_quality_score,
            hybrid_rag.answer_quality_score
        )
        
        if baseline_score > 0:
            return (best_score - baseline_score) / baseline_score
        return 0.0
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete NarrativeQA evaluation."""
        logger.info("🚀 Starting NarrativeQA evaluation...")
        
        # Load dataset
        questions = self.load_narrativeqa_dataset()
        if not questions:
            logger.error("No questions loaded from NarrativeQA dataset")
            return {'error': 'No questions loaded'}
        
        # Initialize systems
        self.initialize_systems()
        
        # Evaluate questions
        logger.info(f"Evaluating {len(questions)} questions...")
        
        for i, question_data in enumerate(questions):
            logger.info(f"Progress: {i+1}/{len(questions)}")
            self.evaluate_question(question_data)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics()
        
        # Save results
        self._save_results()
        
        logger.info("✅ NarrativeQA evaluation completed")
        return overall_metrics
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall evaluation metrics."""
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Calculate averages
        raw_llm_scores = [r.raw_llm_metrics.answer_quality_score for r in self.results]
        standard_rag_scores = [r.standard_rag_metrics.answer_quality_score for r in self.results]
        hybrid_rag_scores = [r.hybrid_rag_metrics.answer_quality_score for r in self.results]
        
        # Calculate BLEU, ROUGE, METEOR averages
        raw_llm_bleu = np.mean([r.raw_llm_metrics.bleu_score for r in self.results])
        standard_rag_bleu = np.mean([r.standard_rag_metrics.bleu_score for r in self.results])
        hybrid_rag_bleu = np.mean([r.hybrid_rag_metrics.bleu_score for r in self.results])
        
        raw_llm_rouge = np.mean([r.raw_llm_metrics.rouge_l_score for r in self.results])
        standard_rag_rouge = np.mean([r.standard_rag_metrics.rouge_l_score for r in self.results])
        hybrid_rag_rouge = np.mean([r.hybrid_rag_metrics.rouge_l_score for r in self.results])
        
        # Calculate exact match rates
        raw_llm_exact = np.mean([r.raw_llm_metrics.exact_match for r in self.results])
        standard_rag_exact = np.mean([r.standard_rag_metrics.exact_match for r in self.results])
        hybrid_rag_exact = np.mean([r.hybrid_rag_metrics.exact_match for r in self.results])
        
        return {
            'total_questions': len(self.results),
            'raw_llm': {
                'avg_quality_score': np.mean(raw_llm_scores),
                'avg_bleu': raw_llm_bleu,
                'avg_rouge': raw_llm_rouge,
                'exact_match_rate': raw_llm_exact
            },
            'standard_rag': {
                'avg_quality_score': np.mean(standard_rag_scores),
                'avg_bleu': standard_rag_bleu,
                'avg_rouge': standard_rag_rouge,
                'exact_match_rate': standard_rag_exact
            },
            'hybrid_rag': {
                'avg_quality_score': np.mean(hybrid_rag_scores),
                'avg_bleu': hybrid_rag_bleu,
                'avg_rouge': hybrid_rag_rouge,
                'exact_match_rate': hybrid_rag_exact
            },
            'best_system': max(['raw_llm', 'standard_rag', 'hybrid_rag'], 
                             key=lambda x: np.mean([r.raw_llm_metrics.answer_quality_score if x == 'raw_llm' 
                                                 else r.standard_rag_metrics.answer_quality_score if x == 'standard_rag'
                                                 else r.hybrid_rag_metrics.answer_quality_score for r in self.results]))
        }
    
    def _save_results(self):
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narrativeqa_results_{timestamp}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            results_data.append({
                'question_id': result.question_id,
                'story_id': result.story_id,
                'question': result.question,
                'reference_answers': result.reference_answers,
                'generated_answer': result.generated_answer,
                'raw_llm_metrics': asdict(result.raw_llm_metrics),
                'standard_rag_metrics': asdict(result.standard_rag_metrics),
                'hybrid_rag_metrics': asdict(result.hybrid_rag_metrics),
                'best_system': result.best_system,
                'improvement_over_baseline': result.improvement_over_baseline,
                'timestamp': result.timestamp
            })
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Main function for NarrativeQA evaluation."""
    parser = argparse.ArgumentParser(
        description="NarrativeQA Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings
  python evaluation/narrativeqa_evaluator.py
  
  # Run with custom parameters
  python evaluation/narrativeqa_evaluator.py --subset test --num-questions 50 --db-path ./my_db
  
  # Run on validation set
  python evaluation/narrativeqa_evaluator.py --subset validation --num-questions 100
        """
    )
    
    parser.add_argument("--subset", type=str, default="test", 
                       choices=["train", "test", "validation"],
                       help="Dataset subset to evaluate")
    parser.add_argument("--num-questions", type=int, default=100,
                       help="Maximum number of questions to evaluate")
    parser.add_argument("--db-path", type=str, default="./full_bookcorpus_db",
                       help="Path to the vector database")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = NarrativeQAEvaluator(
        db_path=args.db_path,
        max_questions=args.num_questions,
        subset=args.subset
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print results
    print("\n📊 NARRATIVEQA EVALUATION RESULTS")
    print("=" * 50)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"Total questions evaluated: {results['total_questions']}")
        print(f"Best performing system: {results['best_system']}")
        
        print("\n📈 SYSTEM COMPARISON:")
        for system, metrics in results.items():
            if system not in ['total_questions', 'best_system']:
                print(f"\n{system.upper()}:")
                print(f"  Quality Score: {metrics['avg_quality_score']:.3f}")
                print(f"  BLEU Score: {metrics['avg_bleu']:.3f}")
                print(f"  ROUGE Score: {metrics['avg_rouge']:.3f}")
                print(f"  Exact Match Rate: {metrics['exact_match_rate']:.3f}")


if __name__ == "__main__":
    main()
