#!/usr/bin/env python3
"""
QA Metrics Analyzer

This script analyzes the results from NarrativeQA system comparisons
using QA-specific evaluation metrics including EM, F1, BERTScore, and METEOR.
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

def analyze_qa_metrics(results: Dict[str, Any]) -> None:
    """Analyze QA-specific evaluation metrics."""
    print("📊 QA METRICS ANALYSIS")
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
        
        # QA-specific metrics
        qa_metric_names = ['exact_match', 'f1_score', 'bert_score', 'meteor_score', 
                          'token_precision', 'token_recall']
        
        for metric in qa_metric_names:
            values = [m.get(metric, 0.0) for m in metrics_list]
            if values:
                avg_val = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                print(f"  {metric:15}: {avg_val:.4f} (max: {max_val:.4f}, min: {min_val:.4f})")
        
        # Traditional metrics
        print(f"\n  Traditional Metrics:")
        traditional_metrics = ['bleu_score', 'rouge_1', 'rouge_2', 'rouge_l', 'word_overlap']
        
        for metric in traditional_metrics:
            values = [m.get(metric, 0.0) for m in metrics_list]
            if values:
                avg_val = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                print(f"  {metric:15}: {avg_val:.4f} (max: {max_val:.4f}, min: {min_val:.4f})")

def analyze_system_comparison(results: Dict[str, Any]) -> None:
    """Analyze system comparison results with QA metrics."""
    print("\n📊 QA SYSTEM COMPARISON")
    print("=" * 60)
    
    system_stats = defaultdict(lambda: {
        'total_questions': 0,
        'avg_exact_match': 0.0,
        'avg_f1_score': 0.0,
        'avg_bert_score': 0.0,
        'avg_meteor_score': 0.0,
        'avg_token_precision': 0.0,
        'avg_token_recall': 0.0,
        'avg_bleu_score': 0.0,
        'avg_rouge_1': 0.0,
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
                    # QA-specific metrics
                    system_stats[system_name]['avg_exact_match'] += evaluation_metrics.get('exact_match', 0.0)
                    system_stats[system_name]['avg_f1_score'] += evaluation_metrics.get('f1_score', 0.0)
                    system_stats[system_name]['avg_bert_score'] += evaluation_metrics.get('bert_score', 0.0)
                    system_stats[system_name]['avg_meteor_score'] += evaluation_metrics.get('meteor_score', 0.0)
                    system_stats[system_name]['avg_token_precision'] += evaluation_metrics.get('token_precision', 0.0)
                    system_stats[system_name]['avg_token_recall'] += evaluation_metrics.get('token_recall', 0.0)
                    
                    # Traditional metrics
                    system_stats[system_name]['avg_bleu_score'] += evaluation_metrics.get('bleu_score', 0.0)
                    system_stats[system_name]['avg_rouge_1'] += evaluation_metrics.get('rouge_1', 0.0)
                    
                    if evaluation_metrics.get('exact_match', 0.0) > 0:
                        system_stats[system_name]['exact_matches'] += 1
    
    # Calculate averages
    for system, stats in system_stats.items():
        if stats['total_questions'] > 0:
            stats['avg_exact_match'] /= stats['total_questions']
            stats['avg_f1_score'] /= stats['total_questions']
            stats['avg_bert_score'] /= stats['total_questions']
            stats['avg_meteor_score'] /= stats['total_questions']
            stats['avg_token_precision'] /= stats['total_questions']
            stats['avg_token_recall'] /= stats['total_questions']
            stats['avg_bleu_score'] /= stats['total_questions']
            stats['avg_rouge_1'] /= stats['total_questions']
            stats['avg_response_time'] /= stats['total_questions']
            stats['avg_answer_length'] /= stats['total_questions']
    
    # Display results
    print(f"{'System':<20} {'Questions':<10} {'EM':<6} {'F1':<6} {'BERT':<6} {'METEOR':<8} {'BLEU':<6} {'ROUGE-1':<8} {'Time':<8}")
    print("-" * 90)
    
    for system, stats in system_stats.items():
        print(f"{system:<20} {stats['total_questions']:<10} "
              f"{stats['avg_exact_match']:<6.3f} {stats['avg_f1_score']:<6.3f} "
              f"{stats['avg_bert_score']:<6.3f} {stats['avg_meteor_score']:<8.3f} "
              f"{stats['avg_bleu_score']:<6.3f} {stats['avg_rouge_1']:<8.3f} "
              f"{stats['avg_response_time']:<8.2f}")

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
        ('f1_score', 'exact_match'),
        ('f1_score', 'bert_score'),
        ('f1_score', 'meteor_score'),
        ('f1_score', 'bleu_score'),
        ('f1_score', 'rouge_1'),
        ('bert_score', 'meteor_score'),
        ('bleu_score', 'rouge_1'),
        ('token_precision', 'token_recall')
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
                denominator = ((n * sum1_sq - sum1**2) * (n * sum2_sq - sum2**2))**0.5
                if denominator != 0:
                    correlation = (n * sum12 - sum1 * sum2) / denominator
                    print(f"  {metric1} vs {metric2}: {correlation:.4f}")
                else:
                    print(f"  {metric1} vs {metric2}: N/A (no variance)")

def analyze_qa_insights(results: Dict[str, Any]) -> None:
    """Provide insights about QA performance."""
    print("\n📊 QA PERFORMANCE INSIGHTS")
    print("=" * 60)
    
    # Collect all metrics
    all_metrics = []
    for system_name, system_data in results.items():
        if isinstance(system_data, list):
            for result in system_data:
                evaluation_metrics = result.get('evaluation_metrics', {})
                if evaluation_metrics:
                    all_metrics.append(evaluation_metrics)
    
    if not all_metrics:
        print("❌ No metrics found for analysis")
        return
    
    # Calculate overall statistics
    exact_matches = sum(1 for m in all_metrics if m.get('exact_match', 0.0) > 0)
    total_questions = len(all_metrics)
    exact_match_rate = exact_matches / total_questions if total_questions > 0 else 0.0
    
    avg_f1 = sum(m.get('f1_score', 0.0) for m in all_metrics) / len(all_metrics)
    avg_bert = sum(m.get('bert_score', 0.0) for m in all_metrics) / len(all_metrics)
    avg_meteor = sum(m.get('meteor_score', 0.0) for m in all_metrics) / len(all_metrics)
    
    print(f"📈 Overall Performance:")
    print(f"  Exact Match Rate: {exact_match_rate:.1%} ({exact_matches}/{total_questions})")
    print(f"  Average F1 Score: {avg_f1:.3f}")
    print(f"  Average BERTScore: {avg_bert:.3f}")
    print(f"  Average METEOR: {avg_meteor:.3f}")
    
    # Performance interpretation
    print(f"\n🎯 Performance Interpretation:")
    if exact_match_rate > 0.5:
        print("  ✅ High exact match rate - answers are very close to references")
    elif exact_match_rate > 0.2:
        print("  ⚠️  Moderate exact match rate - some answers match references")
    else:
        print("  ❌ Low exact match rate - answers differ significantly from references")
    
    if avg_f1 > 0.5:
        print("  ✅ High F1 score - good token-level overlap")
    elif avg_f1 > 0.3:
        print("  ⚠️  Moderate F1 score - some token overlap")
    else:
        print("  ❌ Low F1 score - limited token overlap")
    
    if avg_bert > 0.7:
        print("  ✅ High BERTScore - strong semantic similarity")
    elif avg_bert > 0.5:
        print("  ⚠️  Moderate BERTScore - some semantic similarity")
    else:
        print("  ❌ Low BERTScore - limited semantic similarity")

def main():
    """Main analysis function."""
    print("🔍 QA Metrics Analysis")
    print("=" * 60)
    
    # Load results
    results = load_latest_results()
    if not results:
        return
    
    # Analyze QA metrics
    analyze_qa_metrics(results)
    
    # Analyze system comparison
    analyze_system_comparison(results)
    
    # Analyze metric correlations
    analyze_metric_correlations(results)
    
    # Provide insights
    analyze_qa_insights(results)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
