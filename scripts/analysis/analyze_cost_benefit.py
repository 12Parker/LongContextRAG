#!/usr/bin/env python3
"""
Cost-Benefit Analysis for RAG Systems

This script analyzes the trade-offs between different RAG systems, including:
- Cost efficiency (tokens, API costs)
- Quality-adjusted latency
- Scalability projections
- Performance by question type
- Break-even analysis
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# OpenAI pricing (as of 2024, adjust as needed)
# GPT-4o-mini pricing
INPUT_PRICE_PER_1K_TOKENS = 0.15 / 1000  # $0.15 per 1M tokens
OUTPUT_PRICE_PER_1K_TOKENS = 0.60 / 1000  # $0.60 per 1M tokens

# Embedding pricing (text-embedding-3-large)
EMBEDDING_PRICE_PER_1K_TOKENS = 0.13 / 1000  # $0.13 per 1M tokens

def load_latest_llm_judge_results() -> Dict[str, Any]:
    """Load the latest LLM judge evaluation results."""
    result_files = glob.glob("results/llm_judge_evaluations/llm_judge_evaluation_*.json")
    if not result_files:
        print("❌ No LLM judge evaluation files found")
        return {}
    
    latest_file = max(result_files, key=os.path.getmtime)
    print(f"📁 Loading LLM judge results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def load_latest_comparison_results() -> Dict[str, Any]:
    """Load the latest system comparison results."""
    result_files = glob.glob("results/system_comparisons/system_comparison_narrativeqa_*.json")
    if not result_files:
        print("❌ No comparison results files found")
        return {}
    
    latest_file = max(result_files, key=os.path.getmtime)
    print(f"📁 Loading comparison results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def calculate_token_costs(input_tokens: int, output_tokens: int) -> float:
    """Calculate API cost for tokens."""
    input_cost = input_tokens * INPUT_PRICE_PER_1K_TOKENS / 1000
    output_cost = output_tokens * OUTPUT_PRICE_PER_1K_TOKENS / 1000
    return input_cost + output_cost

def estimate_output_tokens(answer_length: int) -> int:
    """Estimate output tokens from answer length (rough estimate: 1 token ≈ 4 chars)."""
    return answer_length // 4

def calculate_query_embedding_cost(query_text: str) -> float:
    """Calculate embedding cost for a single query (per-query cost)."""
    # Estimate tokens for query (rough estimate: 1 token ≈ 4 chars)
    estimated_tokens = len(query_text) // 4
    return estimated_tokens * EMBEDDING_PRICE_PER_1K_TOKENS / 1000

def calculate_one_time_setup_cost(num_chunks: int, avg_chunk_size: int = 1500) -> float:
    """Calculate one-time embedding cost for document indexing (setup cost, not per-query)."""
    # Estimate tokens for all document embeddings (chunk_size chars ≈ chunk_size/4 tokens)
    estimated_tokens = (num_chunks * avg_chunk_size) // 4
    return estimated_tokens * EMBEDDING_PRICE_PER_1K_TOKENS / 1000

def analyze_system_metrics(llm_judge_results: Dict[str, Any], 
                          comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze comprehensive metrics for each system."""
    
    evaluated_results = llm_judge_results.get('evaluated_results', {})
    system_metrics = {}
    
    for system_name, evaluated_data in evaluated_results.items():
        if not isinstance(evaluated_data, list):
            continue
        
        # Get corresponding comparison data
        comparison_data = comparison_results.get(system_name, [])
        if not isinstance(comparison_data, list):
            comparison_data = []
        
        # Create mapping by question_id for efficient lookup
        comparison_map = {item.get('question_id'): item for item in comparison_data}
        
        metrics = {
            'total_questions': 0,
            'avg_llm_score': 0.0,
            'avg_response_time': 0.0,
            'avg_input_tokens': 0.0,
            'avg_output_tokens': 0.0,
            'avg_cost_per_query': 0.0,
            'avg_retrieved_docs': 0.0,
            'quality_adjusted_latency': 0.0,
            'cost_efficiency_score': 0.0,  # Score per dollar
            'latency_efficiency_score': 0.0,  # Score per second
            'total_cost_1000_queries': 0.0,
            'total_cost_10000_queries': 0.0,
            'total_cost_100000_queries': 0.0,
            'total_latency_1000_queries': 0.0,
            'total_latency_10000_queries': 0.0,
            'total_latency_100000_queries': 0.0,
            'one_time_setup_cost': 0.0,  # One-time cost for indexing (embeddings, vector store)
            'estimated_total_chunks': 0,  # Estimated total chunks in vector store
        }
        
        scores = []
        response_times = []
        input_tokens_list = []
        output_tokens_list = []
        costs = []
        retrieved_docs_list = []
        
        for eval_item in evaluated_data:
            if 'error' in eval_item:
                continue
            
            question_id = eval_item.get('question_id')
            llm_score = eval_item.get('llm_judge', {}).get('overall_score', 0.0)
            
            # Get comparison data
            comp_item = comparison_map.get(question_id, {})
            response_time = comp_item.get('response_time', eval_item.get('response_time', 0.0))
            context_tokens = comp_item.get('context_tokens', eval_item.get('context_tokens', 0))
            answer_length = comp_item.get('answer_length', eval_item.get('answer_length', 0))
            retrieved_docs = comp_item.get('retrieved_docs', eval_item.get('retrieved_docs', 0))
            
            # Calculate tokens and costs
            input_tokens = context_tokens
            output_tokens = estimate_output_tokens(answer_length)
            
            # Calculate cost
            query_cost = calculate_token_costs(input_tokens, output_tokens)
            
            # Add query embedding cost for retrieval systems (only the query needs embedding per request)
            # Note: Document embeddings are created once during indexing (one-time setup cost)
            if retrieved_docs > 0:
                question = comp_item.get('question', eval_item.get('question', ''))
                query_embedding_cost = calculate_query_embedding_cost(question)
                query_cost += query_embedding_cost
            
            scores.append(llm_score)
            response_times.append(response_time)
            input_tokens_list.append(input_tokens)
            output_tokens_list.append(output_tokens)
            costs.append(query_cost)
            retrieved_docs_list.append(retrieved_docs)
            
            metrics['total_questions'] += 1
        
        if metrics['total_questions'] > 0:
            metrics['avg_llm_score'] = sum(scores) / len(scores)
            metrics['avg_response_time'] = sum(response_times) / len(response_times)
            metrics['avg_input_tokens'] = sum(input_tokens_list) / len(input_tokens_list)
            metrics['avg_output_tokens'] = sum(output_tokens_list) / len(output_tokens_list)
            metrics['avg_cost_per_query'] = sum(costs) / len(costs)
            metrics['avg_retrieved_docs'] = sum(retrieved_docs_list) / len(retrieved_docs_list)
            
            # Quality-adjusted metrics
            if metrics['avg_response_time'] > 0:
                metrics['quality_adjusted_latency'] = metrics['avg_response_time'] / (metrics['avg_llm_score'] / 10)
            
            if metrics['avg_cost_per_query'] > 0:
                metrics['cost_efficiency_score'] = metrics['avg_llm_score'] / metrics['avg_cost_per_query']
            
            if metrics['avg_response_time'] > 0:
                metrics['latency_efficiency_score'] = metrics['avg_llm_score'] / metrics['avg_response_time']
            
            # Project costs at scale
            metrics['total_cost_1000_queries'] = metrics['avg_cost_per_query'] * 1000
            metrics['total_cost_10000_queries'] = metrics['avg_cost_per_query'] * 10000
            metrics['total_cost_100000_queries'] = metrics['avg_cost_per_query'] * 100000
            
            # Project latency at scale
            metrics['total_latency_1000_queries'] = metrics['avg_response_time'] * 1000
            metrics['total_latency_10000_queries'] = metrics['avg_response_time'] * 10000
            metrics['total_latency_100000_queries'] = metrics['avg_response_time'] * 100000
            
            # Calculate one-time setup cost for retrieval systems
            # Estimate based on average retrieved docs and typical vector store size
            if metrics['avg_retrieved_docs'] > 0:
                # Estimate total chunks: if we retrieve 10 docs on average, 
                # assume vector store has ~100-1000x more chunks (rough estimate)
                # This is a heuristic - actual size depends on document corpus
                estimated_total_chunks = max(100, metrics['avg_retrieved_docs'] * 100)
                metrics['estimated_total_chunks'] = estimated_total_chunks
                metrics['one_time_setup_cost'] = calculate_one_time_setup_cost(estimated_total_chunks)
        
        system_metrics[system_name] = metrics
    
    return system_metrics

def analyze_question_types(evaluated_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Analyze performance by question type (simple heuristic based on question words)."""
    
    question_categories = {
        'factual': ['who', 'what', 'where', 'when'],
        'inferential': ['why', 'how', 'what does', 'what is'],
        'complex': ['describe', 'explain', 'analyze']
    }
    
    category_metrics = defaultdict(lambda: defaultdict(list))
    
    for system_name, system_data in evaluated_results.items():
        if not isinstance(system_data, list):
            continue
        
        for item in system_data:
            if 'error' in item:
                continue
            
            question = item.get('question', '').lower()
            llm_score = item.get('llm_judge', {}).get('overall_score', 0.0)
            response_time = item.get('response_time', 0.0)
            
            # Categorize question
            category = 'other'
            for cat, keywords in question_categories.items():
                if any(keyword in question for keyword in keywords):
                    category = cat
                    break
            
            category_metrics[category][system_name].append({
                'score': llm_score,
                'time': response_time
            })
    
    # Calculate averages per category
    category_summary = {}
    for category, systems in category_metrics.items():
        category_summary[category] = {}
        for system_name, results in systems.items():
            if results:
                category_summary[category][system_name] = {
                    'avg_score': sum(r['score'] for r in results) / len(results),
                    'avg_time': sum(r['time'] for r in results) / len(results),
                    'count': len(results)
                }
    
    return category_summary

def find_break_even_points(system_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Find break-even points for different scenarios."""
    
    base_llm = system_metrics.get('base_llm', {})
    hybrid_bm25_dense = system_metrics.get('hybrid_bm25_dense', {})
    
    if not base_llm or not hybrid_bm25_dense:
        return {}
    
    break_even = {}
    
    # Cost break-even (when HYBRID becomes cheaper)
    base_cost = base_llm.get('avg_cost_per_query', 0)
    hybrid_cost = hybrid_bm25_dense.get('avg_cost_per_query', 0)
    
    if hybrid_cost < base_cost:
        cost_savings_per_query = base_cost - hybrid_cost
        break_even['cost_savings_per_query'] = cost_savings_per_query
        break_even['cost_savings_1000'] = cost_savings_per_query * 1000
        break_even['cost_savings_10000'] = cost_savings_per_query * 10000
        break_even['cost_savings_100000'] = cost_savings_per_query * 100000
    
    # Quality-adjusted cost break-even
    base_score = base_llm.get('avg_llm_score', 0)
    hybrid_score = hybrid_bm25_dense.get('avg_llm_score', 0)
    
    if base_cost > 0 and hybrid_cost > 0:
        base_cost_per_score = base_cost / base_score if base_score > 0 else float('inf')
        hybrid_cost_per_score = hybrid_cost / hybrid_score if hybrid_score > 0 else float('inf')
        
        break_even['base_cost_per_score'] = base_cost_per_score
        break_even['hybrid_cost_per_score'] = hybrid_cost_per_score
        # Positive means hybrid is cheaper, negative means hybrid is more expensive
        if base_cost_per_score > 0:
            cost_diff = (base_cost_per_score - hybrid_cost_per_score) / base_cost_per_score
            break_even['hybrid_cost_advantage'] = cost_diff
            break_even['hybrid_is_cheaper'] = hybrid_cost_per_score < base_cost_per_score
        else:
            break_even['hybrid_cost_advantage'] = 0
            break_even['hybrid_is_cheaper'] = False
    
    # Latency trade-off
    base_time = base_llm.get('avg_response_time', 0)
    hybrid_time = hybrid_bm25_dense.get('avg_response_time', 0)
    
    if base_time > 0 and hybrid_time > 0:
        break_even['latency_penalty'] = (hybrid_time - base_time) / base_time
        break_even['quality_gain'] = (hybrid_score - base_score) / base_score if base_score > 0 else 0
    
    return break_even

def print_comprehensive_analysis(system_metrics: Dict[str, Any], 
                                 category_metrics: Dict[str, Dict[str, Any]],
                                 break_even: Dict[str, Any]):
    """Print comprehensive cost-benefit analysis."""
    
    print("\n" + "=" * 100)
    print("💰 COST-BENEFIT ANALYSIS")
    print("=" * 100)
    
    # Main metrics table
    print("\n📊 SYSTEM METRICS COMPARISON")
    print("-" * 100)
    print(f"{'System':<25} {'Score':<8} {'Time(s)':<10} {'Cost/Query':<12} {'Input Tokens':<14} {'Efficiency':<12}")
    print("-" * 100)
    
    # Sort by score
    sorted_systems = sorted(
        system_metrics.items(),
        key=lambda x: x[1].get('avg_llm_score', 0),
        reverse=True
    )
    
    for system_name, metrics in sorted_systems:
        print(f"{system_name:<25} "
              f"{metrics['avg_llm_score']:<8.2f} "
              f"{metrics['avg_response_time']:<10.2f} "
              f"${metrics['avg_cost_per_query']:<11.4f} "
              f"{metrics['avg_input_tokens']:<14.0f} "
              f"{metrics['cost_efficiency_score']:<12.2f}")
    
    # One-time setup costs
    print("\n🔧 ONE-TIME SETUP COSTS (Indexing/Embeddings)")
    print("-" * 100)
    print(f"{'System':<25} {'Setup Cost':<15} {'Est. Chunks':<15}")
    print("-" * 100)
    
    for system_name, metrics in sorted_systems:
        setup_cost = metrics.get('one_time_setup_cost', 0.0)
        chunks = metrics.get('estimated_total_chunks', 0)
        if setup_cost > 0:
            print(f"{system_name:<25} "
                  f"${setup_cost:<14.2f} "
                  f"{chunks:<15.0f}")
        else:
            print(f"{system_name:<25} "
                  f"$0.00 (N/A)      "
                  f"{'N/A':<15}")
    
    # Cost projections (per-query costs only, excluding setup)
    print("\n💵 PER-QUERY COST PROJECTIONS AT SCALE")
    print("-" * 100)
    print(f"{'System':<25} {'1K Queries':<15} {'10K Queries':<15} {'100K Queries':<15}")
    print("-" * 100)
    
    for system_name, metrics in sorted_systems:
        print(f"{system_name:<25} "
              f"${metrics['total_cost_1000_queries']:<14.2f} "
              f"${metrics['total_cost_10000_queries']:<14.2f} "
              f"${metrics['total_cost_100000_queries']:<14.2f}")
    
    # Total cost including setup (for first-time deployment)
    print("\n💵 TOTAL COST (Including One-Time Setup)")
    print("-" * 100)
    print(f"{'System':<25} {'1K Queries':<15} {'10K Queries':<15} {'100K Queries':<15}")
    print("-" * 100)
    
    for system_name, metrics in sorted_systems:
        setup_cost = metrics.get('one_time_setup_cost', 0.0)
        print(f"{system_name:<25} "
              f"${metrics['total_cost_1000_queries'] + setup_cost:<14.2f} "
              f"${metrics['total_cost_10000_queries'] + setup_cost:<14.2f} "
              f"${metrics['total_cost_100000_queries'] + setup_cost:<14.2f}")
    
    # Latency projections
    print("\n⏱️  LATENCY PROJECTIONS AT SCALE")
    print("-" * 100)
    print(f"{'System':<25} {'1K Queries (hrs)':<18} {'10K Queries (hrs)':<18} {'100K Queries (hrs)':<18}")
    print("-" * 100)
    
    for system_name, metrics in sorted_systems:
        print(f"{system_name:<25} "
              f"{metrics['total_latency_1000_queries']/3600:<18.2f} "
              f"{metrics['total_latency_10000_queries']/3600:<18.2f} "
              f"{metrics['total_latency_100000_queries']/3600:<18.2f}")
    
    # Quality-adjusted metrics
    print("\n🎯 QUALITY-ADJUSTED METRICS")
    print("-" * 100)
    print(f"{'System':<25} {'Score/Time':<12} {'Score/Cost':<12} {'Quality-Adj Latency':<20}")
    print("-" * 100)
    
    for system_name, metrics in sorted_systems:
        print(f"{system_name:<25} "
              f"{metrics['latency_efficiency_score']:<12.3f} "
              f"{metrics['cost_efficiency_score']:<12.2f} "
              f"{metrics['quality_adjusted_latency']:<20.2f}")
    
    # Break-even analysis
    if break_even:
        print("\n⚖️  BREAK-EVEN ANALYSIS (BASE_LLM vs HYBRID_BM25_DENSE)")
        print("-" * 100)
        
        if 'cost_savings_per_query' in break_even:
            print(f"💰 Cost savings per query: ${break_even['cost_savings_per_query']:.4f}")
            print(f"   At 1K queries: ${break_even['cost_savings_1000']:.2f}")
            print(f"   At 10K queries: ${break_even['cost_savings_10000']:.2f}")
            print(f"   At 100K queries: ${break_even['cost_savings_100000']:.2f}")
        
        if 'hybrid_cost_advantage' in break_even:
            print(f"\n📊 Cost per quality point:")
            print(f"   BASE_LLM: ${break_even['base_cost_per_score']:.4f} per point")
            print(f"   HYBRID_BM25_DENSE: ${break_even['hybrid_cost_per_score']:.4f} per point")
            if break_even.get('hybrid_is_cheaper', False):
                print(f"   HYBRID is {abs(break_even['hybrid_cost_advantage']*100):.1f}% cheaper per quality point")
            else:
                print(f"   HYBRID is {abs(break_even['hybrid_cost_advantage']*100):.1f}% more expensive per quality point")
        
        if 'latency_penalty' in break_even:
            print(f"\n⏱️  Latency trade-off:")
            print(f"   HYBRID is {break_even['latency_penalty']*100:.1f}% slower")
            print(f"   Quality difference: {break_even['quality_gain']*100:.1f}%")
    
    # Question type analysis
    if category_metrics:
        print("\n📝 PERFORMANCE BY QUESTION TYPE")
        print("-" * 100)
        
        for category, systems in category_metrics.items():
            print(f"\n{category.upper()} Questions:")
            print(f"{'System':<25} {'Avg Score':<12} {'Avg Time':<12} {'Count':<8}")
            print("-" * 60)
            
            sorted_sys = sorted(
                systems.items(),
                key=lambda x: x[1].get('avg_score', 0),
                reverse=True
            )
            
            for system_name, metrics in sorted_sys:
                print(f"{system_name:<25} "
                      f"{metrics['avg_score']:<12.2f} "
                      f"{metrics['avg_time']:<12.2f} "
                      f"{metrics['count']:<8}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 100)
    
    base_llm = system_metrics.get('base_llm', {})
    hybrid_bm25_dense = system_metrics.get('hybrid_bm25_dense', {})
    
    if base_llm and hybrid_bm25_dense:
        base_score = base_llm.get('avg_llm_score', 0)
        hybrid_score = hybrid_bm25_dense.get('avg_llm_score', 0)
        base_cost = base_llm.get('avg_cost_per_query', 0)
        hybrid_cost = hybrid_bm25_dense.get('avg_cost_per_query', 0)
        base_time = base_llm.get('avg_response_time', 0)
        hybrid_time = hybrid_bm25_dense.get('avg_response_time', 0)
        
        print("✅ Use BASE_LLM when:")
        print("   - Documents fit within context window (<50k tokens)")
        print("   - Latency is critical (<2s required)")
        print("   - Best quality is needed")
        print("   - Low query volume (<1k queries/month)")
        
        print("\n✅ Use HYBRID_BM25_DENSE when:")
        print("   - Documents exceed context window")
        print("   - High query volume (>10k queries/month)")
        print("   - Cost optimization is important")
        print("   - Handling variable document sizes")
        
        if hybrid_cost < base_cost:
            savings = ((base_cost - hybrid_cost) / base_cost) * 100
            print(f"\n💰 HYBRID saves {savings:.1f}% on cost per query")
        
        if hybrid_time > base_time:
            penalty = ((hybrid_time - base_time) / base_time) * 100
            print(f"⏱️  HYBRID is {penalty:.1f}% slower per query")
        
        if hybrid_score < base_score:
            quality_diff = ((base_score - hybrid_score) / base_score) * 100
            print(f"📊 HYBRID scores {quality_diff:.1f}% lower on quality")

def save_analysis_results(system_metrics: Dict[str, Any],
                         category_metrics: Dict[str, Dict[str, Any]],
                         break_even: Dict[str, Any],
                         output_path: Optional[str] = None):
    """Save analysis results to JSON file."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/cost_benefit_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"cost_benefit_analysis_{timestamp}.json"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'system_metrics': system_metrics,
        'category_metrics': category_metrics,
        'break_even_analysis': break_even,
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'pricing': {
                'input_price_per_1k': INPUT_PRICE_PER_1K_TOKENS * 1000,
                'output_price_per_1k': OUTPUT_PRICE_PER_1K_TOKENS * 1000,
                'embedding_price_per_1k': EMBEDDING_PRICE_PER_1K_TOKENS * 1000
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_path}")
    return output_path

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cost-Benefit Analysis for RAG Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest results
  python scripts/analysis/analyze_cost_benefit.py
  
  # Specify custom files
  python scripts/analysis/analyze_cost_benefit.py \\
    --llm-judge-file results/llm_judge_evaluations/llm_judge_evaluation_20251126_213034.json \\
    --comparison-file results/system_comparisons/system_comparison_narrativeqa_20251126_210117.json
        """
    )
    
    parser.add_argument("--llm-judge-file", type=str, default=None,
                       help="Path to LLM judge evaluation file (default: latest)")
    parser.add_argument("--comparison-file", type=str, default=None,
                       help="Path to system comparison file (default: latest)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    print("💰 Cost-Benefit Analysis for RAG Systems")
    print("=" * 100)
    
    # Load results
    if args.llm_judge_file:
        print(f"📁 Loading LLM judge results from: {args.llm_judge_file}")
        with open(args.llm_judge_file, 'r') as f:
            llm_judge_results = json.load(f)
    else:
        llm_judge_results = load_latest_llm_judge_results()
    
    if args.comparison_file:
        print(f"📁 Loading comparison results from: {args.comparison_file}")
        with open(args.comparison_file, 'r') as f:
            comparison_results = json.load(f)
    else:
        comparison_results = load_latest_comparison_results()
    
    if not llm_judge_results or not comparison_results:
        print("❌ Missing required results files")
        return
    
    # Analyze metrics
    print("\n🔍 Analyzing system metrics...")
    system_metrics = analyze_system_metrics(llm_judge_results, comparison_results)
    
    # Analyze question types
    print("🔍 Analyzing performance by question type...")
    evaluated_results = llm_judge_results.get('evaluated_results', {})
    category_metrics = analyze_question_types(evaluated_results)
    
    # Find break-even points
    print("🔍 Calculating break-even points...")
    break_even = find_break_even_points(system_metrics)
    
    # Print analysis
    print_comprehensive_analysis(system_metrics, category_metrics, break_even)
    
    # Save results
    save_analysis_results(system_metrics, category_metrics, break_even, args.output_file)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()

