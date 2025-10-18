#!/usr/bin/env python3
"""
BLEU Results Analyzer

This script analyzes the results from NarrativeQA system comparisons
using BLEU scoring and other evaluation metrics.
"""

import json
import os
import glob
from typing import Dict, List, Any
from collections import defaultdict

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

def analyze_bleu_metrics(results: Dict[str, Any]) -> None:
    """Analyze BLEU and other evaluation metrics."""
    print("📊 BLEU METRICS ANALYSIS")
    print("=" * 60)
    
    system_metrics = defaultdict(list)
    
    # Collect metrics for each system
    for system_name, system_data in results.items():
        if isinstance(system_data, list):
            for result in system_data:
                evaluation_metrics = result.get('evaluation_metrics', {})
                if evaluation_metrics:
                    system_metrics[system_name].append(evaluation_metrics)
    
    # Analyze each system
    for system, metrics_list in system_metrics.items():
        if not metrics_list:
            continue
            
        print(f"\n🔍 {system.upper()}:")
        print("-" * 40)
        
        # Calculate averages for each metric
        metric_names = ['bleu_score', 'rouge_1', 'rouge_2', 'rouge_l', 'word_overlap', 'exact_match']
        
        for metric in metric_names:
            values = [m.get(metric, 0.0) for m in metrics_list]
            if values:
                avg_val = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                print(f"  {metric:12}: {avg_val:.4f} (max: {max_val:.4f}, min: {min_val:.4f})")

def analyze_system_comparison(results: Dict[str, Any]) -> None:
    """Analyze system comparison results."""
    print("\n📊 SYSTEM COMPARISON ANALYSIS")
    print("=" * 60)
    
    system_stats = defaultdict(lambda: {
        'total_questions': 0,
        'avg_bleu': 0.0,
        'avg_rouge_1': 0.0,
        'avg_rouge_2': 0.0,
        'avg_rouge_l': 0.0,
        'avg_word_overlap': 0.0,
        'exact_matches': 0,
        'avg_response_time': 0.0,
        'avg_answer_length': 0.0
    })
    
    # Collect statistics for each system
    for system_name, system_data in results.items():
        if isinstance(system_data, list):
            for result in system_data:
                evaluation_metrics = result.get('evaluation_metrics', {})
                
                system_stats[system_name]['total_questions'] += 1
                system_stats[system_name]['avg_response_time'] += result.get('response_time', 0)
                system_stats[system_name]['avg_answer_length'] += result.get('answer_length', 0)
                
                if evaluation_metrics:
                    system_stats[system_name]['avg_bleu'] += evaluation_metrics.get('bleu_score', 0.0)
                    system_stats[system_name]['avg_rouge_1'] += evaluation_metrics.get('rouge_1', 0.0)
                    system_stats[system_name]['avg_rouge_2'] += evaluation_metrics.get('rouge_2', 0.0)
                    system_stats[system_name]['avg_rouge_l'] += evaluation_metrics.get('rouge_l', 0.0)
                    system_stats[system_name]['avg_word_overlap'] += evaluation_metrics.get('word_overlap', 0.0)
                    
                    if evaluation_metrics.get('exact_match', 0.0) > 0:
                        system_stats[system_name]['exact_matches'] += 1
    
    # Calculate averages
    for system, stats in system_stats.items():
        if stats['total_questions'] > 0:
            stats['avg_bleu'] /= stats['total_questions']
            stats['avg_rouge_1'] /= stats['total_questions']
            stats['avg_rouge_2'] /= stats['total_questions']
            stats['avg_rouge_l'] /= stats['total_questions']
            stats['avg_word_overlap'] /= stats['total_questions']
            stats['avg_response_time'] /= stats['total_questions']
            stats['avg_answer_length'] /= stats['total_questions']
    
    # Display results
    print(f"{'System':<20} {'Questions':<10} {'BLEU':<8} {'ROUGE-1':<8} {'ROUGE-2':<8} {'ROUGE-L':<8} {'Exact':<8} {'Time':<8}")
    print("-" * 80)
    
    for system, stats in system_stats.items():
        print(f"{system:<20} {stats['total_questions']:<10} "
              f"{stats['avg_bleu']:<8.4f} {stats['avg_rouge_1']:<8.4f} "
              f"{stats['avg_rouge_2']:<8.4f} {stats['avg_rouge_l']:<8.4f} "
              f"{stats['exact_matches']:<8} {stats['avg_response_time']:<8.2f}")

def analyze_metric_correlations(results: Dict[str, Any]) -> None:
    """Analyze correlations between different metrics."""
    print("\n📊 METRIC CORRELATIONS")
    print("=" * 60)
    
    # Collect all metrics
    all_metrics = []
    for system_name, system_data in results.items():
        if isinstance(system_data, list):
            for result in system_data:
                evaluation_metrics = result.get('evaluation_metrics', {})
                if evaluation_metrics:
                    all_metrics.append(evaluation_metrics)
    
    if len(all_metrics) < 2:
        print("❌ Not enough data for correlation analysis")
        return
    
    # Calculate correlations between metrics
    metric_pairs = [
        ('bleu_score', 'rouge_1'),
        ('bleu_score', 'rouge_2'),
        ('bleu_score', 'rouge_l'),
        ('rouge_1', 'rouge_2'),
        ('rouge_1', 'rouge_l'),
        ('rouge_2', 'rouge_l')
    ]
    
    for metric1, metric2 in metric_pairs:
        values1 = [m.get(metric1, 0.0) for m in all_metrics]
        values2 = [m.get(metric2, 0.0) for m in all_metrics]
        
        if values1 and values2:
            # Simple correlation calculation
            n = len(values1)
            sum1 = sum(values1)
            sum2 = sum(values2)
            sum1_sq = sum(x*x for x in values1)
            sum2_sq = sum(x*x for x in values2)
            sum12 = sum(x*y for x, y in zip(values1, values2))
            
            if n > 1:
                correlation = (n * sum12 - sum1 * sum2) / \
                            ((n * sum1_sq - sum1**2) * (n * sum2_sq - sum2**2))**0.5
                print(f"  {metric1} vs {metric2}: {correlation:.4f}")

def main():
    """Main analysis function."""
    print("🔍 BLEU Results Analysis")
    print("=" * 60)
    
    # Load results
    results = load_latest_results()
    if not results:
        return
    
    # Analyze BLEU metrics
    analyze_bleu_metrics(results)
    
    # Analyze system comparison
    analyze_system_comparison(results)
    
    # Analyze metric correlations
    analyze_metric_correlations(results)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
