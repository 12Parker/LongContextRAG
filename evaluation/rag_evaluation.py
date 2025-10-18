#!/usr/bin/env python3
"""
RAG Evaluation Framework

This module provides comprehensive evaluation tools to compare RAG systems
against base LLMs across multiple dimensions including accuracy, relevance,
context utilization, and response quality.
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# from hybrid.working_hybrid_rag import WorkingHybridRAG  # Removed - using NarrativeQA hybrid RAG instead
from core.long_context_config import LongContextManager, ContextSize
import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Metrics for evaluating RAG vs Base LLM performance."""
    
    # Response metrics
    response_length: int
    response_time: float
    token_count: int
    
    # Context metrics
    context_length: int
    retrieved_docs: int
    context_utilization: float  # How much of available context was used
    
    # Quality metrics (to be filled by human evaluation or automated scoring)
    relevance_score: float = 0.0
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    coherence_score: float = 0.0
    
    # Technical metrics
    method: str = ""
    error_count: int = 0
    fallback_used: bool = False

@dataclass
class EvaluationResult:
    """Complete evaluation result for a single query."""
    query: str
    base_llm_metrics: EvaluationMetrics
    rag_metrics: EvaluationMetrics
    hybrid_rag_metrics: EvaluationMetrics
    improvement_score: float = 0.0
    timestamp: str = ""

class RAGEvaluator:
    """Comprehensive evaluator for RAG systems vs Base LLM."""
    
    def __init__(self, context_size: ContextSize = ContextSize.LARGE):
        self.context_size = context_size
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.results = []
        
        # Initialize systems
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all systems for evaluation."""
        print("🔧 Initializing evaluation systems...")
        
        # Initialize long context manager
        self.manager = LongContextManager()
        self.manager.config = self.manager.config.__class__.for_context_size(self.context_size)
        
        # Create hybrid RAG system
        self.hybrid_rag = self.manager.create_hybrid_rag()
        
        # Create vector store
        print(f"📚 Creating vector store with {self.manager.config.num_documents:,} documents...")
        self.hybrid_rag.create_vectorstore(
            use_vectordb=True,
            num_documents=self.manager.config.num_documents
        )
        
        print("✅ Systems initialized successfully!")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def evaluate_query(self, query: str) -> EvaluationResult:
        """Evaluate a single query across all systems."""
        print(f"\n🔍 Evaluating query: {query[:60]}...")
        
        # Test Base LLM (no RAG)
        print("   Testing Base LLM...")
        base_metrics = self._evaluate_base_llm(query)
        
        # Test RAG (base RAG system)
        print("   Testing Base RAG...")
        rag_metrics = self._evaluate_base_rag(query)
        
        # Test Hybrid RAG
        print("   Testing Hybrid RAG...")
        hybrid_metrics = self._evaluate_hybrid_rag(query)
        
        # Calculate improvement score
        improvement_score = self._calculate_improvement_score(
            base_metrics, rag_metrics, hybrid_metrics
        )
        
        result = EvaluationResult(
            query=query,
            base_llm_metrics=base_metrics,
            rag_metrics=rag_metrics,
            hybrid_rag_metrics=hybrid_metrics,
            improvement_score=improvement_score,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def _evaluate_base_llm(self, query: str) -> EvaluationMetrics:
        """Evaluate base LLM without RAG."""
        start_time = time.time()
        
        try:
            # Use the base RAG system but disable RAG
            response = self.hybrid_rag.base_rag.generate_response(query, use_rag=False)
            
            elapsed_time = time.time() - start_time
            response_text = response.get('response', '')
            
            return EvaluationMetrics(
                response_length=len(response_text),
                response_time=elapsed_time,
                token_count=self.count_tokens(response_text),
                context_length=0,
                retrieved_docs=0,
                context_utilization=0.0,
                method="base_llm",
                error_count=0
            )
            
        except Exception as e:
            logger.error(f"Base LLM evaluation failed: {e}")
            return EvaluationMetrics(
                response_length=0,
                response_time=time.time() - start_time,
                token_count=0,
                context_length=0,
                retrieved_docs=0,
                context_utilization=0.0,
                method="base_llm",
                error_count=1
            )
    
    def _evaluate_base_rag(self, query: str) -> EvaluationMetrics:
        """Evaluate base RAG system."""
        start_time = time.time()
        
        try:
            # Use base RAG with retrieval
            response = self.hybrid_rag.base_rag.generate_response(query, use_rag=True)
            
            elapsed_time = time.time() - start_time
            response_text = response.get('response', '')
            context_length = response.get('context_length', 0)
            retrieved_docs = response.get('retrieved_docs', 0)
            
            # Calculate context utilization
            max_context = self.manager.config.max_context_tokens * 4  # Rough char estimate
            context_utilization = min(context_length / max_context, 1.0) if max_context > 0 else 0
            
            return EvaluationMetrics(
                response_length=len(response_text),
                response_time=elapsed_time,
                token_count=self.count_tokens(response_text),
                context_length=context_length,
                retrieved_docs=retrieved_docs,
                context_utilization=context_utilization,
                method=response.get('method', 'base_rag'),
                error_count=0
            )
            
        except Exception as e:
            logger.error(f"Base RAG evaluation failed: {e}")
            return EvaluationMetrics(
                response_length=0,
                response_time=time.time() - start_time,
                token_count=0,
                context_length=0,
                retrieved_docs=0,
                context_utilization=0.0,
                method="base_rag",
                error_count=1
            )
    
    def _evaluate_hybrid_rag(self, query: str) -> EvaluationMetrics:
        """Evaluate hybrid RAG system."""
        start_time = time.time()
        
        try:
            # Use hybrid RAG
            response = self.hybrid_rag.generate_response(query, task_type='qa')
            
            elapsed_time = time.time() - start_time
            response_text = response.get('response', '')
            context_length = response.get('context_length', 0)
            retrieved_docs = response.get('retrieved_docs', 0)
            
            # Calculate context utilization
            max_context = self.manager.config.max_context_tokens * 4  # Rough char estimate
            context_utilization = min(context_length / max_context, 1.0) if max_context > 0 else 0
            
            return EvaluationMetrics(
                response_length=len(response_text),
                response_time=elapsed_time,
                token_count=self.count_tokens(response_text),
                context_length=context_length,
                retrieved_docs=retrieved_docs,
                context_utilization=context_utilization,
                method=response.get('method', 'hybrid_rag'),
                error_count=0
            )
            
        except Exception as e:
            logger.error(f"Hybrid RAG evaluation failed: {e}")
            return EvaluationMetrics(
                response_length=0,
                response_time=time.time() - start_time,
                token_count=0,
                context_length=0,
                retrieved_docs=0,
                context_utilization=0.0,
                method="hybrid_rag",
                error_count=1
            )
    
    def _calculate_improvement_score(self, base_metrics: EvaluationMetrics, 
                                   rag_metrics: EvaluationMetrics, 
                                   hybrid_metrics: EvaluationMetrics) -> float:
        """Calculate overall improvement score."""
        # Weighted scoring based on multiple factors
        scores = []
        
        # Context utilization improvement
        if base_metrics.context_utilization == 0:
            context_improvement = (rag_metrics.context_utilization + hybrid_metrics.context_utilization) / 2
        else:
            context_improvement = ((rag_metrics.context_utilization / base_metrics.context_utilization) + 
                                 (hybrid_metrics.context_utilization / base_metrics.context_utilization)) / 2
        scores.append(context_improvement)
        
        # Response quality (length as proxy for completeness)
        if base_metrics.response_length > 0:
            rag_quality = rag_metrics.response_length / base_metrics.response_length
            hybrid_quality = hybrid_metrics.response_length / base_metrics.response_length
            scores.append((rag_quality + hybrid_quality) / 2)
        
        # Information density (context per response)
        if base_metrics.response_length > 0:
            base_density = 0  # No context
            rag_density = rag_metrics.context_length / rag_metrics.response_length if rag_metrics.response_length > 0 else 0
            hybrid_density = hybrid_metrics.context_length / hybrid_metrics.response_length if hybrid_metrics.response_length > 0 else 0
            scores.append((rag_density + hybrid_density) / 2)
        
        return np.mean(scores) if scores else 0.0
    
    def run_comprehensive_evaluation(self, test_queries: Optional[List[str]] = None) -> List[EvaluationResult]:
        """Run comprehensive evaluation across multiple queries."""
        if test_queries is None:
            test_queries = self._get_default_test_queries()
        
        print(f"🚀 Starting comprehensive evaluation with {len(test_queries)} queries")
        print("=" * 70)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}/{len(test_queries)}")
            result = self.evaluate_query(query)
            self._print_query_summary(result)
        
        self._save_results()
        self._print_comprehensive_summary()
        
        return self.results
    
    def _get_default_test_queries(self) -> List[str]:
        """Get default test queries for evaluation."""
        return [
            "What are the main themes and literary devices used in the stories?",
            "Analyze the character development patterns across different narratives",
            "Compare the dialogue styles and narrative techniques used",
            "Identify recurring motifs and their symbolic significance",
            "Evaluate the pacing and structure of the storytelling",
            "What conflicts and resolutions are present in the narratives?",
            "How do the authors develop atmosphere and setting?",
            "What narrative perspectives and point of view techniques are used?",
            "Analyze the relationship between plot and character development",
            "What literary genres and styles are represented in the collection?"
        ]
    
    def _print_query_summary(self, result: EvaluationResult):
        """Print summary for a single query evaluation."""
        print(f"   📊 Results Summary:")
        print(f"      Base LLM:    {result.base_llm_metrics.response_length:>4} chars, "
              f"{result.base_llm_metrics.response_time:>5.2f}s, "
              f"{result.base_llm_metrics.context_length:>4} context")
        print(f"      Base RAG:    {result.rag_metrics.response_length:>4} chars, "
              f"{result.rag_metrics.response_time:>5.2f}s, "
              f"{result.rag_metrics.context_length:>4} context, "
              f"{result.rag_metrics.retrieved_docs} docs")
        print(f"      Hybrid RAG:  {result.hybrid_rag_metrics.response_length:>4} chars, "
              f"{result.hybrid_rag_metrics.response_time:>5.2f}s, "
              f"{result.hybrid_rag_metrics.context_length:>4} context, "
              f"{result.hybrid_rag_metrics.retrieved_docs} docs")
        print(f"      Improvement: {result.improvement_score:>5.2f}x")
    
    def _print_comprehensive_summary(self):
        """Print comprehensive evaluation summary."""
        if not self.results:
            return
        
        print(f"\n{'='*70}")
        print("📈 COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        # Calculate averages
        base_avg_length = np.mean([r.base_llm_metrics.response_length for r in self.results])
        rag_avg_length = np.mean([r.rag_metrics.response_length for r in self.results])
        hybrid_avg_length = np.mean([r.hybrid_rag_metrics.response_length for r in self.results])
        
        base_avg_time = np.mean([r.base_llm_metrics.response_time for r in self.results])
        rag_avg_time = np.mean([r.rag_metrics.response_time for r in self.results])
        hybrid_avg_time = np.mean([r.hybrid_rag_metrics.response_time for r in self.results])
        
        base_avg_context = np.mean([r.base_llm_metrics.context_length for r in self.results])
        rag_avg_context = np.mean([r.rag_metrics.context_length for r in self.results])
        hybrid_avg_context = np.mean([r.hybrid_rag_metrics.context_length for r in self.results])
        
        avg_improvement = np.mean([r.improvement_score for r in self.results])
        
        print(f"📊 Average Response Length:")
        print(f"   Base LLM:    {base_avg_length:>6.0f} characters")
        print(f"   Base RAG:    {rag_avg_length:>6.0f} characters ({rag_avg_length/base_avg_length:>4.2f}x)")
        print(f"   Hybrid RAG:  {hybrid_avg_length:>6.0f} characters ({hybrid_avg_length/base_avg_length:>4.2f}x)")
        
        print(f"\n⏱️  Average Response Time:")
        print(f"   Base LLM:    {base_avg_time:>6.2f} seconds")
        print(f"   Base RAG:    {rag_avg_time:>6.2f} seconds ({rag_avg_time/base_avg_time:>4.2f}x)")
        print(f"   Hybrid RAG:  {hybrid_avg_time:>6.2f} seconds ({hybrid_avg_time/base_avg_time:>4.2f}x)")
        
        print(f"\n📄 Average Context Usage:")
        print(f"   Base LLM:    {base_avg_context:>6.0f} characters")
        print(f"   Base RAG:    {rag_avg_context:>6.0f} characters")
        print(f"   Hybrid RAG:  {hybrid_avg_context:>6.0f} characters")
        
        print(f"\n🎯 Overall Improvement Score: {avg_improvement:>5.2f}x")
        
        # Success rates
        base_errors = sum(1 for r in self.results if r.base_llm_metrics.error_count > 0)
        rag_errors = sum(1 for r in self.results if r.rag_metrics.error_count > 0)
        hybrid_errors = sum(1 for r in self.results if r.hybrid_rag_metrics.error_count > 0)
        
        print(f"\n✅ Success Rates:")
        print(f"   Base LLM:    {(len(self.results) - base_errors) / len(self.results) * 100:>5.1f}%")
        print(f"   Base RAG:    {(len(self.results) - rag_errors) / len(self.results) * 100:>5.1f}%")
        print(f"   Hybrid RAG:  {(len(self.results) - hybrid_errors) / len(self.results) * 100:>5.1f}%")
        
        # Recommendations
        print(f"\n💡 Key Insights:")
        if hybrid_avg_length > base_avg_length * 1.5:
            print(f"   ✅ Hybrid RAG provides significantly more detailed responses")
        if hybrid_avg_context > 0:
            print(f"   ✅ Hybrid RAG effectively utilizes retrieved context")
        if avg_improvement > 1.5:
            print(f"   ✅ Overall improvement is substantial ({avg_improvement:.2f}x)")
        if hybrid_avg_time < base_avg_time * 2:
            print(f"   ✅ Performance overhead is reasonable")
    
    def _save_results(self):
        """Save evaluation results to file."""
        os.makedirs("results", exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'query': result.query,
                'base_llm_metrics': asdict(result.base_llm_metrics),
                'rag_metrics': asdict(result.rag_metrics),
                'hybrid_rag_metrics': asdict(result.hybrid_rag_metrics),
                'improvement_score': result.improvement_score,
                'timestamp': result.timestamp
            })
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/rag_evaluation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'evaluation_config': {
                    'context_size': self.context_size.value,
                    'max_context_tokens': self.manager.config.max_context_tokens,
                    'num_documents': self.manager.config.num_documents,
                    'timestamp': timestamp
                },
                'results': serializable_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main function to run RAG evaluation."""
    print("🧪 RAG vs Base LLM Evaluation Framework")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(context_size=ContextSize.MEDIUM)  # Start with medium for faster testing
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 Evaluated {len(results)} queries across 3 systems")
    print(f"💡 Use the saved results to analyze performance differences")

if __name__ == "__main__":
    main()
