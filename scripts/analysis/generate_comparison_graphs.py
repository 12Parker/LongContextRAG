#!/usr/bin/env python3
"""
Generate Comparison Graphs for Paper

This script creates publication-quality comparison graphs from evaluation results.
"""

import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_latest_comparison_results() -> Dict[str, Any]:
    """Load the latest comparison results."""
    result_files = glob.glob("results/system_comparisons/system_comparison_narrativeqa_*.json")
    if not result_files:
        raise FileNotFoundError("No comparison results found")
    
    latest_file = max(result_files, key=lambda f: os.path.getmtime(f))
    print(f"📁 Loading comparison results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def load_specific_comparison_results(filepath: str) -> Dict[str, Any]:
    """Load comparison results from a specific file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_latest_llm_judge_results() -> Dict[str, Any]:
    """Load the latest LLM judge results."""
    result_files = glob.glob("results/llm_judge_evaluations/llm_judge_evaluation_*.json")
    if not result_files:
        raise FileNotFoundError("No LLM judge results found")
    
    latest_file = max(result_files, key=lambda f: os.path.getmtime(f))
    print(f"📁 Loading LLM judge results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def load_multiple_llm_judge_results() -> List[Dict[str, Any]]:
    """Load multiple LLM judge results for chunk size comparison."""
    result_files = sorted(glob.glob("results/llm_judge_evaluations/llm_judge_evaluation_*.json"), 
                         key=lambda f: os.path.getmtime(f), reverse=True)
    
    results = []
    for filepath in result_files[:5]:  # Get last 5 results
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Try to infer chunk size from filename or metadata
                results.append(data)
        except Exception as e:
            print(f"⚠️  Skipping {filepath}: {e}")
    
    return results

def calculate_system_metrics(comparison_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Calculate aggregate metrics for each system."""
    metrics = {}
    
    for system_name, results in comparison_results.items():
        if not isinstance(results, list):
            continue
        
        successful = [r for r in results if 'error' not in r]
        if not successful:
            continue
        
        metrics[system_name] = {
            'avg_time': np.mean([r.get('response_time', 0) for r in successful]),
            'avg_tokens': np.mean([r.get('context_tokens', 0) for r in successful]),
            'avg_relevance': np.mean([r.get('relevance_score', 0) for r in successful]),
            'success_rate': len(successful) / len(results) * 100,
            'num_questions': len(successful)
        }
    
    return metrics

def plot_performance_comparison(comparison_results: Dict[str, Any], output_dir: Path):
    """Create performance comparison bar charts."""
    metrics = calculate_system_metrics(comparison_results)
    
    if not metrics:
        print("⚠️  No metrics to plot")
        return
    
    systems = list(metrics.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Average Response Time
    times = [metrics[s]['avg_time'] for s in systems]
    colors = sns.color_palette("husl", len(systems))
    
    bars1 = axes[0].bar(systems, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Average Response Time (seconds)', fontweight='bold')
    axes[0].set_title('(a) Latency Comparison', fontweight='bold', pad=10)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim(0, max(times) * 1.2)
    
    # Add value labels on bars
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Average Context Tokens
    tokens = [metrics[s]['avg_tokens'] for s in systems]
    bars2 = axes[1].bar(systems, tokens, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Average Context Tokens', fontweight='bold')
    axes[1].set_title('(b) Token Usage Comparison', fontweight='bold', pad=10)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim(0, max(tokens) * 1.2)
    
    for bar, token in zip(bars2, tokens):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(token)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Average Relevance Score
    relevance = [metrics[s]['avg_relevance'] for s in systems]
    bars3 = axes[2].bar(systems, relevance, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Average Relevance Score (F1)', fontweight='bold')
    axes[2].set_title('(c) Relevance Comparison', fontweight='bold', pad=10)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    axes[2].set_ylim(0, max(relevance) * 1.3 if max(relevance) > 0 else 0.3)
    
    for bar, rel in zip(bars3, relevance):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{rel:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Rotate x-axis labels
    for ax in axes:
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in systems], 
                          rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_quality_scores(llm_judge_results: Dict[str, Any], output_dir: Path):
    """Create quality score comparison from LLM judge results."""
    system_averages = llm_judge_results.get('system_averages', {})
    
    if not system_averages:
        print("⚠️  No LLM judge averages to plot")
        return
    
    systems = list(system_averages.keys())
    
    # Extract metrics
    overall = [system_averages[s]['avg_overall_score'] for s in systems]
    correctness = [system_averages[s]['avg_correctness'] for s in systems]
    completeness = [system_averages[s]['avg_completeness'] for s in systems]
    relevance = [system_averages[s]['avg_relevance'] for s in systems]
    clarity = [system_averages[s]['avg_clarity'] for s in systems]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(systems))
    width = 0.15
    
    bars1 = ax.bar(x - 2*width, overall, width, label='Overall', alpha=0.9, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x - width, correctness, width, label='Correctness', alpha=0.9, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x, completeness, width, label='Completeness', alpha=0.9, edgecolor='black', linewidth=1)
    bars4 = ax.bar(x + width, relevance, width, label='Relevance', alpha=0.9, edgecolor='black', linewidth=1)
    bars5 = ax.bar(x + 2*width, clarity, width, label='Clarity', alpha=0.9, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Score (0-10)', fontweight='bold')
    ax.set_xlabel('System', fontweight='bold')
    ax.set_title('LLM Judge Quality Scores by System', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in systems], rotation=15, ha='right')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    output_path = output_dir / 'quality_scores.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_chunk_size_comparison(output_dir: Path):
    """Compare different chunk sizes."""
    # Load multiple LLM judge results
    results = load_multiple_llm_judge_results()
    
    # We'll need to manually map chunk sizes or use file timestamps
    # For now, let's use the known chunk sizes from our tests
    chunk_sizes = [200, 600, 1200]
    
    # Try to find results for each chunk size
    # This is a simplified approach - in practice, you'd tag results with chunk size
    comparison_files = [
        "results/system_comparisons/system_comparison_narrativeqa_20251130_164419.json",  # chunk 200
        "results/system_comparisons/system_comparison_narrativeqa_20251130_170539.json",  # chunk 600
        "results/system_comparisons/system_comparison_narrativeqa_20251126_231015.json",  # chunk 1200
    ]
    
    llm_judge_files = [
        "results/llm_judge_evaluations/llm_judge_evaluation_20251130_164718.json",  # chunk 200
        "results/llm_judge_evaluations/llm_judge_evaluation_20251130_170836.json",  # chunk 600
        "results/llm_judge_evaluations/llm_judge_evaluation_20251126_231843.json",  # chunk 1200
    ]
    
    data = []
    for chunk_size, comp_file, judge_file in zip(chunk_sizes, comparison_files, llm_judge_files):
        try:
            comp_data = load_specific_comparison_results(comp_file)
            judge_data = json.load(open(judge_file))
            
            # Calculate metrics
            system_results = comp_data.get('hybrid_bm25_optimized', [])
            if system_results:
                successful = [r for r in system_results if 'error' not in r]
                avg_time = np.mean([r.get('response_time', 0) for r in successful])
                avg_tokens = np.mean([r.get('context_tokens', 0) for r in successful])
                
                judge_avg = judge_data.get('system_averages', {}).get('hybrid_bm25_optimized', {})
                quality = judge_avg.get('avg_overall_score', 0)
                
                data.append({
                    'chunk_size': chunk_size,
                    'time': avg_time,
                    'tokens': avg_tokens,
                    'quality': quality
                })
        except Exception as e:
            print(f"⚠️  Skipping chunk size {chunk_size}: {e}")
    
    if not data:
        print("⚠️  No chunk size comparison data found")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    chunk_sizes = [d['chunk_size'] for d in data]
    times = [d['time'] for d in data]
    tokens = [d['tokens'] for d in data]
    qualities = [d['quality'] for d in data]
    
    # 1. Latency vs Chunk Size
    axes[0].plot(chunk_sizes, times, marker='o', markersize=10, linewidth=2.5, 
                color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='black', 
                markeredgewidth=1.5)
    axes[0].set_xlabel('Chunk Size', fontweight='bold')
    axes[0].set_ylabel('Average Response Time (s)', fontweight='bold')
    axes[0].set_title('(a) Latency vs Chunk Size', fontweight='bold', pad=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xticks(chunk_sizes)
    
    # 2. Token Usage vs Chunk Size
    axes[1].plot(chunk_sizes, tokens, marker='s', markersize=10, linewidth=2.5,
                color='#F18F01', markerfacecolor='#C73E1D', markeredgecolor='black',
                markeredgewidth=1.5)
    axes[1].set_xlabel('Chunk Size', fontweight='bold')
    axes[1].set_ylabel('Average Context Tokens', fontweight='bold')
    axes[1].set_title('(b) Token Usage vs Chunk Size', fontweight='bold', pad=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticks(chunk_sizes)
    
    # 3. Quality vs Chunk Size
    axes[2].plot(chunk_sizes, qualities, marker='^', markersize=10, linewidth=2.5,
                color='#6A994E', markerfacecolor='#A7C957', markeredgecolor='black',
                markeredgewidth=1.5)
    axes[2].set_xlabel('Chunk Size', fontweight='bold')
    axes[2].set_ylabel('LLM Judge Quality Score', fontweight='bold')
    axes[2].set_title('(c) Quality vs Chunk Size', fontweight='bold', pad=10)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xticks(chunk_sizes)
    axes[2].set_ylim(0, 10)
    
    plt.tight_layout()
    output_path = output_dir / 'chunk_size_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_cost_benefit_tradeoff(comparison_results: Dict[str, Any], 
                               llm_judge_results: Dict[str, Any],
                               output_dir: Path):
    """Create cost-benefit trade-off visualization."""
    metrics = calculate_system_metrics(comparison_results)
    system_averages = llm_judge_results.get('system_averages', {})
    
    if not metrics or not system_averages:
        print("⚠️  Insufficient data for cost-benefit plot")
        return
    
    # Calculate cost per query (rough estimate: $0.00015 per 1K input tokens)
    systems = []
    costs = []
    qualities = []
    times = []
    
    for system_name in metrics.keys():
        if system_name in system_averages:
            systems.append(system_name)
            # Estimate cost: input tokens * price per 1K tokens
            avg_tokens = metrics[system_name]['avg_tokens']
            cost = (avg_tokens / 1000) * 0.00015  # gpt-4o-mini input price
            costs.append(cost * 1000)  # Convert to cost per 1K queries
            qualities.append(system_averages[system_name]['avg_overall_score'])
            times.append(metrics[system_name]['avg_time'])
    
    if not systems:
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scatter = ax.scatter(costs, qualities, s=[t*200 for t in times], 
                        c=times, cmap='viridis', alpha=0.7, 
                        edgecolors='black', linewidths=1.5, zorder=3)
    
    # Add labels
    for i, system in enumerate(systems):
        ax.annotate(system.replace('_', ' ').title(), 
                   (costs[i], qualities[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='black', alpha=0.7))
    
    ax.set_xlabel('Estimated Cost per 1K Queries ($)', fontweight='bold')
    ax.set_ylabel('LLM Judge Quality Score', fontweight='bold')
    ax.set_title('Cost-Benefit Trade-off Analysis', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Average Response Time (s)', fontweight='bold', rotation=270, labelpad=20)
    
    # Add legend for bubble size
    sizes = [min(times), np.median(times), max(times)]
    legend_elements = [plt.scatter([], [], s=s*200, c='gray', alpha=0.7, 
                                   edgecolors='black', linewidths=1.5,
                                   label=f'{s:.2f}s') for s in sizes]
    ax.legend(handles=legend_elements, title='Response Time', 
             loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = output_dir / 'cost_benefit_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_efficiency_metrics(comparison_results: Dict[str, Any],
                           llm_judge_results: Dict[str, Any],
                           output_dir: Path):
    """Create efficiency metrics comparison."""
    metrics = calculate_system_metrics(comparison_results)
    system_averages = llm_judge_results.get('system_averages', {})
    
    if not metrics or not system_averages:
        return
    
    systems = []
    quality_per_time = []
    quality_per_token = []
    quality_per_cost = []
    
    for system_name in metrics.keys():
        if system_name in system_averages:
            systems.append(system_name)
            quality = system_averages[system_name]['avg_overall_score']
            time = metrics[system_name]['avg_time']
            tokens = metrics[system_name]['avg_tokens']
            cost = (tokens / 1000) * 0.00015
            
            quality_per_time.append(quality / time if time > 0 else 0)
            quality_per_token.append(quality / tokens if tokens > 0 else 0)
            quality_per_cost.append(quality / cost if cost > 0 else 0)
    
    if not systems:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(systems))
    width = 0.6
    colors = sns.color_palette("muted", 3)
    
    bars1 = axes[0].bar(x, quality_per_time, width, color=colors[0], 
                       alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Quality / Time', fontweight='bold')
    axes[0].set_title('(a) Quality per Second', fontweight='bold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.replace('_', ' ').title() for s in systems], 
                            rotation=15, ha='right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    bars2 = axes[1].bar(x, quality_per_token, width, color=colors[1],
                       alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Quality / Token (x10^-4)', fontweight='bold')
    axes[1].set_title('(b) Quality per Token', fontweight='bold', pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.replace('_', ' ').title() for s in systems],
                            rotation=15, ha='right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    # Scale for readability
    # Scale y-axis labels for readability
    yticks = axes[1].get_yticks()
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels([f'{y*10000:.2f}' for y in yticks])
    
    bars3 = axes[2].bar(x, quality_per_cost, width, color=colors[2],
                       alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Quality / Cost', fontweight='bold')
    axes[2].set_title('(c) Quality per Dollar', fontweight='bold', pad=10)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([s.replace('_', ' ').title() for s in systems],
                            rotation=15, ha='right')
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'efficiency_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    """Main function to generate all graphs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comparison graphs for paper")
    parser.add_argument("--comparison-file", type=str, default=None,
                       help="Path to specific comparison results file")
    parser.add_argument("--llm-judge-file", type=str, default=None,
                       help="Path to specific LLM judge results file")
    parser.add_argument("--output-dir", type=str, default="results/graphs",
                       help="Output directory for graphs")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Generating Comparison Graphs for Paper")
    print("=" * 60)
    
    # Load data
    if args.comparison_file:
        comparison_results = load_specific_comparison_results(args.comparison_file)
    else:
        comparison_results = load_latest_comparison_results()
    
    if args.llm_judge_file:
        with open(args.llm_judge_file, 'r') as f:
            llm_judge_results = json.load(f)
    else:
        llm_judge_results = load_latest_llm_judge_results()
    
    # Generate graphs
    print("\n1. Generating performance comparison...")
    plot_performance_comparison(comparison_results, output_dir)
    
    print("\n2. Generating quality scores comparison...")
    plot_quality_scores(llm_judge_results, output_dir)
    
    print("\n3. Generating chunk size comparison...")
    try:
        plot_chunk_size_comparison(output_dir)
    except Exception as e:
        print(f"⚠️  Chunk size comparison skipped: {e}")
    
    print("\n4. Generating cost-benefit trade-off...")
    plot_cost_benefit_tradeoff(comparison_results, llm_judge_results, output_dir)
    
    print("\n5. Generating efficiency metrics...")
    plot_efficiency_metrics(comparison_results, llm_judge_results, output_dir)
    
    print(f"\n✅ All graphs saved to: {output_dir}")
    print("\n📋 Generated graphs:")
    print("  - performance_comparison.png")
    print("  - quality_scores.png")
    print("  - chunk_size_comparison.png")
    print("  - cost_benefit_tradeoff.png")
    print("  - efficiency_metrics.png")

if __name__ == "__main__":
    main()

