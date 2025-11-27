#!/usr/bin/env python3
"""
LLM as Judge Analysis

This script uses an LLM to evaluate the quality of generated answers
compared to reference answers from NarrativeQA system comparisons.
It provides scores and overall averages for each system.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.config import config
from langchain_openai import ChatOpenAI

def load_latest_results() -> Dict[str, Any]:
    """Load the latest comparison results."""
    # Find the most recent results file
    result_files = glob.glob("results/system_comparisons/system_comparison_narrativeqa_*.json")
    if not result_files:
        print("❌ No results files found")
        return {}
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"📁 Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def load_specific_results(filepath: str) -> Dict[str, Any]:
    """Load results from a specific file."""
    print(f"📁 Loading results from: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

class LLMJudge:
    """LLM-based judge for evaluating answer quality."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        """
        Initialize the LLM judge.
        
        Args:
            model: OpenAI model to use (defaults to config.OPENAI_MODEL)
            temperature: Temperature for LLM (0.0 for consistent scoring)
        """
        self.model = model or config.OPENAI_MODEL
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=temperature
        )
    
    def evaluate_answer(self, 
                       question: str,
                       reference_answers: List[str],
                       generated_answer: str) -> Dict[str, Any]:
        """
        Evaluate a generated answer against reference answers.
        
        Args:
            question: The question being answered
            reference_answers: List of reference answers
            generated_answer: The generated answer to evaluate
            
        Returns:
            Dictionary with score and reasoning
        """
        # Format reference answers
        ref_text = "\n".join([f"- {ref}" for ref in reference_answers])
        
        prompt = f"""You are an expert evaluator for question-answering systems. Your task is to evaluate how well a generated answer addresses a question compared to reference answers.

Question: {question}

Reference Answers (one or more acceptable answers):
{ref_text}

Generated Answer:
{generated_answer}

Please evaluate the generated answer on the following criteria:
1. **Correctness**: Does the answer correctly address the question? (0-10)
2. **Completeness**: Does it cover the key information from the reference answers? (0-10)
3. **Relevance**: Is the answer relevant to the question? (0-10)
4. **Clarity**: Is the answer clear and well-structured? (0-10)

Provide your evaluation in the following JSON format:
{{
    "correctness_score": <0-10>,
    "completeness_score": <0-10>,
    "relevance_score": <0-10>,
    "clarity_score": <0-10>,
    "overall_score": <0-10>,
    "reasoning": "<brief explanation of your scores>"
}}

The overall_score should be a weighted average considering all criteria, with correctness being most important.
Return ONLY valid JSON, no additional text."""

        response_text = ""
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            evaluation = json.loads(response_text)
            
            # Validate scores are in range
            for key in ['correctness_score', 'completeness_score', 'relevance_score', 
                       'clarity_score', 'overall_score']:
                if key in evaluation:
                    evaluation[key] = max(0, min(10, float(evaluation[key])))
            
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            if response_text:
                print(f"Response was: {response_text[:200]}...")
            # Return default scores
            return {
                "correctness_score": 5.0,
                "completeness_score": 5.0,
                "relevance_score": 5.0,
                "clarity_score": 5.0,
                "overall_score": 5.0,
                "reasoning": f"Error parsing response: {str(e)}"
            }
        except Exception as e:
            print(f"⚠️  Error evaluating answer: {e}")
            return {
                "correctness_score": 5.0,
                "completeness_score": 5.0,
                "relevance_score": 5.0,
                "clarity_score": 5.0,
                "overall_score": 5.0,
                "reasoning": f"Error: {str(e)}"
            }

def evaluate_all_results(results: Dict[str, Any], 
                         max_questions: Optional[int] = None,
                         judge: LLMJudge = None) -> Dict[str, Any]:
    """
    Evaluate all results using LLM judge.
    
    Args:
        results: Results dictionary from comparison script
        max_questions: Maximum number of questions to evaluate per system (None for all)
        judge: LLMJudge instance (creates new one if None)
        
    Returns:
        Dictionary with evaluated results and statistics
    """
    if judge is None:
        judge = LLMJudge()
    
    evaluated_results = {}
    system_stats = defaultdict(lambda: {
        'total_questions': 0,
        'scores': [],
        'correctness_scores': [],
        'completeness_scores': [],
        'relevance_scores': [],
        'clarity_scores': [],
        'evaluations': []
    })
    
    print("\n🔍 Evaluating answers with LLM judge...")
    print("=" * 60)
    
    total_evaluations = 0
    for system_name, system_data in results.items():
        if not isinstance(system_data, list):
            continue
        
        print(f"\n📊 Evaluating {system_name}...")
        evaluated_results[system_name] = []
        
        questions_to_evaluate = system_data
        if max_questions:
            questions_to_evaluate = system_data[:max_questions]
        
        for i, result in enumerate(questions_to_evaluate):
            if 'error' in result:
                # Skip failed results but include them
                evaluated_results[system_name].append({
                    **result,
                    'llm_judge': {
                        'overall_score': 0.0,
                        'correctness_score': 0.0,
                        'completeness_score': 0.0,
                        'relevance_score': 0.0,
                        'clarity_score': 0.0,
                        'reasoning': 'Error in original result'
                    }
                })
                continue
            
            question = result.get('question', '')
            reference_answers = result.get('reference_answers', [])
            generated_answer = result.get('generated_answer', '')
            
            if not question or not reference_answers or not generated_answer:
                print(f"  ⚠️  Skipping question {i+1}: missing data")
                continue
            
            print(f"  📋 Question {i+1}/{len(questions_to_evaluate)}: {question[:60]}...")
            
            # Evaluate with LLM judge
            evaluation = judge.evaluate_answer(
                question=question,
                reference_answers=reference_answers,
                generated_answer=generated_answer
            )
            
            # Add evaluation to result
            evaluated_result = {
                **result,
                'llm_judge': evaluation
            }
            evaluated_results[system_name].append(evaluated_result)
            
            # Update statistics
            if 'error' not in result:
                system_stats[system_name]['total_questions'] += 1
                system_stats[system_name]['scores'].append(evaluation.get('overall_score', 0.0))
                system_stats[system_name]['correctness_scores'].append(evaluation.get('correctness_score', 0.0))
                system_stats[system_name]['completeness_scores'].append(evaluation.get('completeness_score', 0.0))
                system_stats[system_name]['relevance_scores'].append(evaluation.get('relevance_score', 0.0))
                system_stats[system_name]['clarity_scores'].append(evaluation.get('clarity_score', 0.0))
                system_stats[system_name]['evaluations'].append(evaluation)
            
            total_evaluations += 1
            
            # Small delay to avoid rate limiting
            import time
            time.sleep(0.1)
    
    print(f"\n✅ Evaluated {total_evaluations} answers")
    
    # Calculate averages
    system_averages = {}
    for system_name, stats in system_stats.items():
        if stats['total_questions'] > 0:
            system_averages[system_name] = {
                'total_questions': stats['total_questions'],
                'avg_overall_score': sum(stats['scores']) / len(stats['scores']),
                'avg_correctness': sum(stats['correctness_scores']) / len(stats['correctness_scores']),
                'avg_completeness': sum(stats['completeness_scores']) / len(stats['completeness_scores']),
                'avg_relevance': sum(stats['relevance_scores']) / len(stats['relevance_scores']),
                'avg_clarity': sum(stats['clarity_scores']) / len(stats['clarity_scores']),
                'min_score': min(stats['scores']),
                'max_score': max(stats['scores']),
                'median_score': sorted(stats['scores'])[len(stats['scores']) // 2] if stats['scores'] else 0.0
            }
    
    return {
        'evaluated_results': evaluated_results,
        'system_averages': system_averages,
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'model_used': judge.model,
            'total_evaluations': total_evaluations
        }
    }

def print_summary(evaluation_data: Dict[str, Any]):
    """Print summary of LLM judge evaluation."""
    system_averages = evaluation_data.get('system_averages', {})
    metadata = evaluation_data.get('metadata', {})
    
    print("\n" + "=" * 80)
    print("📊 LLM JUDGE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model used: {metadata.get('model_used', 'unknown')}")
    print(f"Total evaluations: {metadata.get('total_evaluations', 0)}")
    print(f"Evaluation date: {metadata.get('evaluation_date', 'unknown')}")
    
    print("\n" + "-" * 80)
    print(f"{'System':<25} {'Questions':<12} {'Overall':<10} {'Correct':<10} {'Complete':<10} {'Relevant':<10} {'Clear':<10}")
    print("-" * 80)
    
    # Sort by overall score
    sorted_systems = sorted(
        system_averages.items(),
        key=lambda x: x[1].get('avg_overall_score', 0.0),
        reverse=True
    )
    
    for system_name, stats in sorted_systems:
        print(f"{system_name:<25} "
              f"{stats['total_questions']:<12} "
              f"{stats['avg_overall_score']:<10.2f} "
              f"{stats['avg_correctness']:<10.2f} "
              f"{stats['avg_completeness']:<10.2f} "
              f"{stats['avg_relevance']:<10.2f} "
              f"{stats['avg_clarity']:<10.2f}")
    
    print("\n" + "=" * 80)
    print("📈 DETAILED STATISTICS")
    print("=" * 80)
    
    for system_name, stats in sorted_systems:
        print(f"\n{system_name.upper()}:")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Average overall score: {stats['avg_overall_score']:.2f}/10")
        print(f"  Average correctness: {stats['avg_correctness']:.2f}/10")
        print(f"  Average completeness: {stats['avg_completeness']:.2f}/10")
        print(f"  Average relevance: {stats['avg_relevance']:.2f}/10")
        print(f"  Average clarity: {stats['avg_clarity']:.2f}/10")
        print(f"  Score range: {stats['min_score']:.2f} - {stats['max_score']:.2f}")
        print(f"  Median score: {stats['median_score']:.2f}/10")
    
    # Find best system
    if sorted_systems:
        best_system = sorted_systems[0]
        print(f"\n🏆 Best performing system: {best_system[0]}")
        print(f"   Overall score: {best_system[1]['avg_overall_score']:.2f}/10")

def save_results(evaluation_data: Dict[str, Any], output_path: Optional[str] = None):
    """Save evaluation results to JSON file."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/llm_judge_evaluations")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"llm_judge_evaluation_{timestamp}.json"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    return output_path

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM as Judge Evaluation for NarrativeQA Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate latest results (all questions)
  python scripts/analysis/analyze_llm_judge.py
  
  # Evaluate latest results (first 10 questions per system)
  python scripts/analysis/analyze_llm_judge.py --max-questions 10
  
  # Evaluate specific results file
  python scripts/analysis/analyze_llm_judge.py --input-file results/system_comparisons/system_comparison_narrativeqa_20251126_210117.json
  
  # Use specific model
  python scripts/analysis/analyze_llm_judge.py --model gpt-4o
        """
    )
    
    parser.add_argument("--input-file", type=str, default=None,
                       help="Path to specific results file (default: latest)")
    parser.add_argument("--max-questions", type=int, default=None,
                       help="Maximum number of questions to evaluate per system (default: all)")
    parser.add_argument("--model", type=str, default=None,
                       help="OpenAI model to use for judging (default: from config)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    print("🔍 LLM as Judge Evaluation")
    print("=" * 60)
    
    # Load results
    if args.input_file:
        results = load_specific_results(args.input_file)
    else:
        results = load_latest_results()
    
    if not results:
        print("❌ No results to evaluate")
        return
    
    # Initialize judge
    judge = LLMJudge(model=args.model)
    
    # Evaluate results
    evaluation_data = evaluate_all_results(
        results=results,
        max_questions=args.max_questions,
        judge=judge
    )
    
    # Print summary
    print_summary(evaluation_data)
    
    # Save results
    save_results(evaluation_data, args.output_file)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()

