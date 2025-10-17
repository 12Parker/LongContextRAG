#!/usr/bin/env python3
"""
Summarize Base LLM Results Against NarrativeQA

This script analyzes the results from your base LLM testing against NarrativeQA
and provides insights into performance and areas for improvement.

Usage:
    python summarize_base_llm_results.py
    python summarize_base_llm_results.py --results-file base_llm_narrativeqa_results_20251016_194237.json
"""

import sys
import os
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def find_latest_results_file() -> str:
    """Find the most recent results file."""
    results_files = list(Path('.').glob('base_llm_narrativeqa_results_*.json'))
    if not results_files:
        return None
    
    # Sort by modification time, newest first
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(results_files[0])

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the results and provide insights."""
    if not results:
        return {'error': 'No results to analyze'}
    
    # Basic statistics
    total_questions = len(results)
    successful_questions = len([r for r in results if 'error' not in r])
    success_rate = successful_questions / total_questions * 100 if total_questions > 0 else 0
    
    if successful_questions == 0:
        return {
            'error': 'No successful responses',
            'total_questions': total_questions,
            'success_rate': 0
        }
    
    # Calculate averages
    avg_response_time = sum(r['response_time'] for r in results if 'error' not in r) / successful_questions
    avg_answer_length = sum(r['answer_length'] for r in results if 'error' not in r) / successful_questions
    avg_context_tokens = sum(r['context_tokens'] for r in results if 'error' not in r) / successful_questions
    avg_relevance_score = sum(r['relevance_score'] for r in results if 'error' not in r) / successful_questions
    
    # Find best and worst performing questions
    successful_results = [r for r in results if 'error' not in r]
    best_question = max(successful_results, key=lambda x: x['relevance_score'])
    worst_question = min(successful_results, key=lambda x: x['relevance_score'])
    
    # Analyze answer quality
    high_relevance = len([r for r in successful_results if r['relevance_score'] > 0.5])
    medium_relevance = len([r for r in successful_results if 0.2 <= r['relevance_score'] <= 0.5])
    low_relevance = len([r for r in successful_results if r['relevance_score'] < 0.2])
    
    return {
        'total_questions': total_questions,
        'successful_questions': successful_questions,
        'success_rate': success_rate,
        'avg_response_time': avg_response_time,
        'avg_answer_length': avg_answer_length,
        'avg_context_tokens': avg_context_tokens,
        'avg_relevance_score': avg_relevance_score,
        'best_question': best_question,
        'worst_question': worst_question,
        'relevance_distribution': {
            'high_relevance': high_relevance,
            'medium_relevance': medium_relevance,
            'low_relevance': low_relevance
        }
    }

def print_analysis(analysis: Dict[str, Any]):
    """Print the analysis results."""
    print("📊 BASE LLM NARRATIVEQA PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    # Basic metrics
    print(f"Total questions tested: {analysis['total_questions']}")
    print(f"Successful responses: {analysis['successful_questions']}")
    print(f"Success rate: {analysis['success_rate']:.1f}%")
    print()
    
    # Performance metrics
    print("📈 PERFORMANCE METRICS:")
    print(f"Average response time: {analysis['avg_response_time']:.2f}s")
    print(f"Average answer length: {analysis['avg_answer_length']:.0f} characters")
    print(f"Average context tokens: {analysis['avg_context_tokens']:.0f}")
    print(f"Average relevance score: {analysis['avg_relevance_score']:.3f}")
    print()
    
    # Relevance distribution
    print("🎯 ANSWER QUALITY DISTRIBUTION:")
    rel_dist = analysis['relevance_distribution']
    total_successful = analysis['successful_questions']
    
    if total_successful > 0:
        print(f"High relevance (>0.5): {rel_dist['high_relevance']} ({rel_dist['high_relevance']/total_successful*100:.1f}%)")
        print(f"Medium relevance (0.2-0.5): {rel_dist['medium_relevance']} ({rel_dist['medium_relevance']/total_successful*100:.1f}%)")
        print(f"Low relevance (<0.2): {rel_dist['low_relevance']} ({rel_dist['low_relevance']/total_successful*100:.1f}%)")
    print()
    
    # Best and worst questions
    print("🏆 BEST PERFORMING QUESTION:")
    best = analysis['best_question']
    print(f"Question: {best['question'][:80]}...")
    print(f"Relevance score: {best['relevance_score']:.3f}")
    print(f"Answer length: {best['answer_length']} chars")
    print(f"Response time: {best['response_time']:.2f}s")
    print()
    
    print("⚠️  WORST PERFORMING QUESTION:")
    worst = analysis['worst_question']
    print(f"Question: {worst['question'][:80]}...")
    print(f"Relevance score: {worst['relevance_score']:.3f}")
    print(f"Answer length: {worst['answer_length']} chars")
    print(f"Response time: {worst['response_time']:.2f}s")
    print()

def provide_insights(analysis: Dict[str, Any]):
    """Provide insights and recommendations."""
    print("💡 INSIGHTS AND RECOMMENDATIONS:")
    print("=" * 60)
    
    if 'error' in analysis:
        print("❌ Cannot provide insights due to errors in results")
        return
    
    # Success rate insights
    success_rate = analysis['success_rate']
    if success_rate == 100:
        print("✅ Excellent: 100% success rate - all questions were answered")
    elif success_rate >= 80:
        print("✅ Good: High success rate - most questions were answered")
    elif success_rate >= 60:
        print("⚠️  Moderate: Some questions failed - check error handling")
    else:
        print("❌ Poor: Many questions failed - check system configuration")
    
    # Response time insights
    avg_time = analysis['avg_response_time']
    if avg_time < 3:
        print("✅ Fast: Response times are good")
    elif avg_time < 10:
        print("⚠️  Moderate: Response times are acceptable")
    else:
        print("❌ Slow: Response times are too high - consider optimization")
    
    # Relevance insights
    avg_relevance = analysis['avg_relevance_score']
    if avg_relevance > 0.5:
        print("✅ High relevance: Answers are well-aligned with reference answers")
    elif avg_relevance > 0.3:
        print("⚠️  Moderate relevance: Some answers are relevant, room for improvement")
    else:
        print("❌ Low relevance: Answers don't match reference answers well")
    
    # Context usage insights
    avg_tokens = analysis['avg_context_tokens']
    if avg_tokens < 500:
        print("⚠️  Low context usage: Consider using more context for better answers")
    elif avg_tokens < 2000:
        print("✅ Good context usage: Appropriate amount of context")
    else:
        print("⚠️  High context usage: Consider optimizing context length")
    
    print()
    print("🔧 RECOMMENDATIONS:")
    print("=" * 60)
    
    # Specific recommendations
    if analysis['avg_relevance_score'] < 0.4:
        print("1. Improve answer relevance:")
        print("   - Use more specific context from BookCorpus")
        print("   - Consider using RAG systems for better retrieval")
        print("   - Fine-tune prompt engineering")
    
    if analysis['avg_response_time'] > 5:
        print("2. Optimize response time:")
        print("   - Reduce context length")
        print("   - Use faster models")
        print("   - Implement caching")
    
    if analysis['avg_context_tokens'] < 1000:
        print("3. Increase context usage:")
        print("   - Load more books from BookCorpus")
        print("   - Use longer context windows")
        print("   - Implement better context selection")
    
    print("4. Consider RAG systems:")
    print("   - Standard RAG for better retrieval")
    print("   - Hybrid RAG for advanced capabilities")
    print("   - Compare performance with current baseline")
    
    print()
    print("📚 NEXT STEPS:")
    print("=" * 60)
    print("1. Run comparison with RAG systems:")
    print("   python compare_systems_narrativeqa.py --systems base_llm,standard_rag")
    print()
    print("2. Test with more questions:")
    print("   python test_base_llm_narrativeqa.py --num-questions 10")
    print()
    print("3. Analyze specific question types:")
    print("   - Character questions")
    print("   - Plot questions") 
    print("   - Theme questions")
    print()
    print("4. Optimize your base LLM setup:")
    print("   - Adjust context length")
    print("   - Improve prompt engineering")
    print("   - Use better book selection")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Summarize Base LLM Results Against NarrativeQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest results
  python summarize_base_llm_results.py
  
  # Analyze specific results file
  python summarize_base_llm_results.py --results-file base_llm_narrativeqa_results_20251016_194237.json
        """
    )
    
    parser.add_argument("--results-file", type=str, default=None,
                       help="Path to results file (default: find latest)")
    
    args = parser.parse_args()
    
    # Find results file
    if args.results_file:
        results_file = args.results_file
    else:
        results_file = find_latest_results_file()
    
    if not results_file or not os.path.exists(results_file):
        print("❌ No results file found")
        print("Run: python test_base_llm_narrativeqa.py --num-questions 5")
        return
    
    print(f"📁 Loading results from: {results_file}")
    
    # Load and analyze results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        analysis = analyze_results(results)
        print_analysis(analysis)
        provide_insights(analysis)
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")

if __name__ == "__main__":
    main()
