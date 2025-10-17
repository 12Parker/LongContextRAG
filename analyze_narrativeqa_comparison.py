#!/usr/bin/env python3
"""
Analyze NarrativeQA Comparison Results

This script analyzes the results from comparing base LLM vs NarrativeQA RAG
systems against NarrativeQA questions and provides insights.

Usage:
    python analyze_narrativeqa_comparison.py
    python analyze_narrativeqa_comparison.py --results-file system_comparison_narrativeqa_20251016_200324.json
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

def find_latest_comparison_file() -> str:
    """Find the most recent comparison results file."""
    results_files = list(Path('.').glob('system_comparison_narrativeqa_*.json'))
    if not results_files:
        return None
    
    # Sort by modification time, newest first
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(results_files[0])

def analyze_comparison_results(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze the comparison results and provide insights."""
    if not results:
        return {'error': 'No results to analyze'}
    
    analysis = {}
    
    for system_name, system_results in results.items():
        if not system_results:
            analysis[system_name] = {'error': 'No results for this system'}
            continue
        
        # Basic statistics
        total_questions = len(system_results)
        successful_questions = len([r for r in system_results if 'error' not in r])
        success_rate = successful_questions / total_questions * 100 if total_questions > 0 else 0
        
        if successful_questions == 0:
            analysis[system_name] = {
                'error': 'No successful responses',
                'total_questions': total_questions,
                'success_rate': 0
            }
            continue
        
        # Calculate averages
        successful_results = [r for r in system_results if 'error' not in r]
        avg_response_time = sum(r['response_time'] for r in successful_results) / successful_questions
        avg_answer_length = sum(r['answer_length'] for r in successful_results) / successful_questions
        avg_context_tokens = sum(r['context_tokens'] for r in successful_results) / successful_questions
        avg_relevance_score = sum(r['relevance_score'] for r in successful_results) / successful_questions
        avg_retrieved_docs = sum(r.get('retrieved_docs', 0) for r in successful_results) / successful_questions
        
        # Find best and worst performing questions
        best_question = max(successful_results, key=lambda x: x['relevance_score'])
        worst_question = min(successful_results, key=lambda x: x['relevance_score'])
        
        # Analyze answer quality
        high_relevance = len([r for r in successful_results if r['relevance_score'] > 0.5])
        medium_relevance = len([r for r in successful_results if 0.2 <= r['relevance_score'] <= 0.5])
        low_relevance = len([r for r in successful_results if r['relevance_score'] < 0.2])
        
        analysis[system_name] = {
            'total_questions': total_questions,
            'successful_questions': successful_questions,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'avg_answer_length': avg_answer_length,
            'avg_context_tokens': avg_context_tokens,
            'avg_relevance_score': avg_relevance_score,
            'avg_retrieved_docs': avg_retrieved_docs,
            'best_question': best_question,
            'worst_question': worst_question,
            'relevance_distribution': {
                'high_relevance': high_relevance,
                'medium_relevance': medium_relevance,
                'low_relevance': low_relevance
            }
        }
    
    return analysis

def print_comparison_analysis(analysis: Dict[str, Any]):
    """Print the comparison analysis results."""
    print("📊 NARRATIVEQA SYSTEM COMPARISON ANALYSIS")
    print("=" * 60)
    
    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    # Print results for each system
    for system_name, system_analysis in analysis.items():
        if 'error' in system_analysis:
            print(f"\n{system_name.upper()}: ❌ {system_analysis['error']}")
            continue
        
        print(f"\n{system_name.upper()}:")
        print(f"  Success rate: {system_analysis['success_rate']:.1f}%")
        print(f"  Avg response time: {system_analysis['avg_response_time']:.2f}s")
        print(f"  Avg answer length: {system_analysis['avg_answer_length']:.0f} chars")
        print(f"  Avg context tokens: {system_analysis['avg_context_tokens']:.0f}")
        print(f"  Avg relevance score: {system_analysis['avg_relevance_score']:.3f}")
        if system_analysis['avg_retrieved_docs'] > 0:
            print(f"  Avg retrieved docs: {system_analysis['avg_retrieved_docs']:.1f}")
        
        # Relevance distribution
        rel_dist = system_analysis['relevance_distribution']
        total_successful = system_analysis['successful_questions']
        if total_successful > 0:
            print(f"  Relevance distribution:")
            print(f"    High relevance (>0.5): {rel_dist['high_relevance']} ({rel_dist['high_relevance']/total_successful*100:.1f}%)")
            print(f"    Medium relevance (0.2-0.5): {rel_dist['medium_relevance']} ({rel_dist['medium_relevance']/total_successful*100:.1f}%)")
            print(f"    Low relevance (<0.2): {rel_dist['low_relevance']} ({rel_dist['low_relevance']/total_successful*100:.1f}%)")
    
    # Compare systems
    if len(analysis) >= 2:
        print(f"\n🏆 SYSTEM COMPARISON:")
        print("=" * 60)
        
        # Find best system for each metric
        systems = list(analysis.keys())
        successful_systems = [s for s in systems if 'error' not in analysis[s]]
        
        if len(successful_systems) >= 2:
            # Compare relevance scores
            relevance_scores = {s: analysis[s]['avg_relevance_score'] for s in successful_systems}
            best_relevance = max(relevance_scores, key=relevance_scores.get)
            print(f"  Best relevance score: {best_relevance} ({relevance_scores[best_relevance]:.3f})")
            
            # Compare response times
            response_times = {s: analysis[s]['avg_response_time'] for s in successful_systems}
            fastest = min(response_times, key=response_times.get)
            print(f"  Fastest response: {fastest} ({response_times[fastest]:.2f}s)")
            
            # Compare context efficiency
            context_efficiency = {}
            for s in successful_systems:
                if analysis[s]['avg_context_tokens'] > 0:
                    efficiency = analysis[s]['avg_relevance_score'] / (analysis[s]['avg_context_tokens'] / 1000)
                    context_efficiency[s] = efficiency
            
            if context_efficiency:
                most_efficient = max(context_efficiency, key=context_efficiency.get)
                print(f"  Most context efficient: {most_efficient} ({context_efficiency[most_efficient]:.3f})")

def provide_insights(analysis: Dict[str, Any]):
    """Provide insights and recommendations."""
    print(f"\n💡 INSIGHTS AND RECOMMENDATIONS:")
    print("=" * 60)
    
    if 'error' in analysis:
        print("❌ Cannot provide insights due to errors in results")
        return
    
    successful_systems = [s for s in analysis.keys() if 'error' not in analysis[s]]
    
    if len(successful_systems) < 2:
        print("⚠️  Need at least 2 systems for meaningful comparison")
        return
    
    # Analyze performance differences
    print("🔍 KEY FINDINGS:")
    
    # Relevance analysis
    relevance_scores = {s: analysis[s]['avg_relevance_score'] for s in successful_systems}
    best_relevance = max(relevance_scores, key=relevance_scores.get)
    worst_relevance = min(relevance_scores, key=relevance_scores.get)
    
    print(f"  • {best_relevance} has the highest relevance score ({relevance_scores[best_relevance]:.3f})")
    print(f"  • {worst_relevance} has the lowest relevance score ({relevance_scores[worst_relevance]:.3f})")
    
    # Speed analysis
    response_times = {s: analysis[s]['avg_response_time'] for s in successful_systems}
    fastest = min(response_times, key=response_times.get)
    slowest = max(response_times, key=response_times.get)
    
    print(f"  • {fastest} is the fastest ({response_times[fastest]:.2f}s)")
    print(f"  • {slowest} is the slowest ({response_times[slowest]:.2f}s)")
    
    # Context usage analysis
    context_usage = {s: analysis[s]['avg_context_tokens'] for s in successful_systems}
    most_context = max(context_usage, key=context_usage.get)
    least_context = min(context_usage, key=context_usage.get)
    
    print(f"  • {most_context} uses the most context ({context_usage[most_context]:.0f} tokens)")
    print(f"  • {least_context} uses the least context ({context_usage[least_context]:.0f} tokens)")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    print("=" * 60)
    
    # System-specific recommendations
    for system_name in successful_systems:
        system_analysis = analysis[system_name]
        
        print(f"\n{system_name.upper()}:")
        
        if system_analysis['avg_relevance_score'] < 0.4:
            print("  • Improve answer relevance - consider better context selection")
        
        if system_analysis['avg_response_time'] > 8:
            print("  • Optimize response time - consider caching or faster models")
        
        if system_analysis['avg_context_tokens'] > 1000:
            print("  • Reduce context usage - consider more targeted retrieval")
        elif system_analysis['avg_context_tokens'] < 500:
            print("  • Increase context usage - consider retrieving more relevant chunks")
    
    print(f"\n📚 NEXT STEPS:")
    print("=" * 60)
    print("1. **Improve the winning system:**")
    print(f"   - Focus on {best_relevance} for best performance")
    print("   - Optimize its weaknesses")
    
    print("2. **Hybrid approach:**")
    print("   - Combine strengths of both systems")
    print("   - Use RAG for retrieval, base LLM for generation")
    
    print("3. **Further testing:**")
    print("   - Test with more questions")
    print("   - Test on different question types")
    print("   - Test on different story types")
    
    print("4. **System optimization:**")
    print("   - Fine-tune retrieval parameters")
    print("   - Improve prompt engineering")
    print("   - Consider different embedding models")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze NarrativeQA Comparison Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest results
  python analyze_narrativeqa_comparison.py
  
  # Analyze specific results file
  python analyze_narrativeqa_comparison.py --results-file system_comparison_narrativeqa_20251016_200324.json
        """
    )
    
    parser.add_argument("--results-file", type=str, default=None,
                       help="Path to results file (default: find latest)")
    
    args = parser.parse_args()
    
    # Find results file
    if args.results_file:
        results_file = args.results_file
    else:
        results_file = find_latest_comparison_file()
    
    if not results_file or not os.path.exists(results_file):
        print("❌ No comparison results file found")
        print("Run: python compare_systems_narrativeqa.py --systems base_llm,narrativeqa_rag")
        return
    
    print(f"📁 Loading results from: {results_file}")
    
    # Load and analyze results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        analysis = analyze_comparison_results(results)
        print_comparison_analysis(analysis)
        provide_insights(analysis)
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")

if __name__ == "__main__":
    main()
