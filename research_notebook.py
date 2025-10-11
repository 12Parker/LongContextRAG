"""
Research Notebook for Hybrid Attention RAG System

This notebook provides a comprehensive research environment for experimenting with
the hybrid attention RAG methodology, including evaluation, analysis, and comparison tools.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from pathlib import Path
import time
from dataclasses import dataclass, asdict

from hybrid_rag_integration import HybridRAGIntegration, create_hybrid_rag_system
from hybrid_attention_rag import AttentionConfig
from neural_retriever import RetrieverConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for research experiments."""
    # Model configurations
    attention_configs: List[Dict[str, Any]]
    retriever_configs: List[Dict[str, Any]]
    
    # Experiment parameters
    num_trials: int = 3
    test_queries: List[str] = None
    evaluation_metrics: List[str] = None
    
    # Output settings
    save_results: bool = True
    output_dir: str = './research_results'
    create_plots: bool = True

class HybridRAGResearcher:
    """
    Research environment for the hybrid attention RAG system.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        
        # Create output directory
        Path(config.output_dir).mkdir(exist_ok=True)
        
    def run_comprehensive_experiment(self) -> Dict[str, Any]:
        """Run comprehensive experiments across different configurations."""
        logger.info("Starting comprehensive hybrid attention RAG experiments")
        
        all_results = {}
        
        # Test different attention configurations
        for i, attn_config in enumerate(self.config.attention_configs):
            logger.info(f"Testing attention configuration {i+1}/{len(self.config.attention_configs)}")
            
            for j, ret_config in enumerate(self.config.retriever_configs):
                logger.info(f"  Testing retriever configuration {j+1}/{len(self.config.retriever_configs)}")
                
                # Create system with current configuration
                system_config = {
                    'attention': attn_config,
                    'retriever': ret_config,
                    'integration': {
                        'use_hybrid_attention': True,
                        'use_neural_retriever': True,
                        'use_dynamic_queries': True
                    }
                }
                
                # Run experiments
                results = self._run_configuration_experiment(system_config, f"attn_{i}_ret_{j}")
                all_results[f"attn_{i}_ret_{j}"] = results
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        
        # Save results
        if self.config.save_results:
            self._save_results(all_results, analysis)
        
        return {
            'experiment_results': all_results,
            'analysis': analysis
        }
    
    def _run_configuration_experiment(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Run experiments for a specific configuration."""
        results = {
            'config': config,
            'config_name': config_name,
            'performance_metrics': {},
            'attention_analysis': {},
            'retrieval_analysis': {},
            'timing': {}
        }
        
        try:
            # Create system
            start_time = time.time()
            hybrid_rag = create_hybrid_rag_system(config)
            setup_time = time.time() - start_time
            
            # Load documents
            sample_file = "data/sample_documents.txt"
            if Path(sample_file).exists():
                documents = hybrid_rag.load_documents([sample_file])
                hybrid_rag.create_vectorstore(documents)
            else:
                logger.warning("Sample documents not found, using base system")
            
            # Run test queries
            test_queries = self.config.test_queries or [
                "What is machine learning and what are its main types?",
                "How do transformer models work in NLP?",
                "What are the benefits of RAG systems?",
                "What challenges do long context models face?",
                "Compare supervised and unsupervised learning approaches"
            ]
            
            query_results = []
            for query in test_queries:
                query_start = time.time()
                
                # Test different methods
                comparison = hybrid_rag.compare_methods(query, 'qa')
                attention_analysis = hybrid_rag.analyze_attention_patterns(query)
                
                query_time = time.time() - query_start
                
                query_results.append({
                    'query': query,
                    'comparison': comparison,
                    'attention_analysis': attention_analysis,
                    'processing_time': query_time
                })
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(query_results)
            
            # Store results
            results.update({
                'setup_time': setup_time,
                'query_results': query_results,
                'performance_metrics': performance_metrics,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Error in configuration experiment {config_name}: {e}")
            results.update({
                'error': str(e),
                'success': False
            })
        
        return results
    
    def _calculate_performance_metrics(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from query results."""
        metrics = {
            'avg_processing_time': 0,
            'success_rate': 0,
            'avg_context_length': 0,
            'avg_retrieved_docs': 0,
            'attention_quality': 0
        }
        
        if not query_results:
            return metrics
        
        # Calculate timing metrics
        processing_times = [qr['processing_time'] for qr in query_results]
        metrics['avg_processing_time'] = np.mean(processing_times)
        metrics['std_processing_time'] = np.std(processing_times)
        
        # Calculate success rate
        successful_queries = sum(1 for qr in query_results if 'error' not in qr['comparison'].get('hybrid_rag', {}))
        metrics['success_rate'] = successful_queries / len(query_results)
        
        # Calculate context and retrieval metrics
        context_lengths = []
        retrieved_docs_counts = []
        attention_qualities = []
        
        for qr in query_results:
            hybrid_result = qr['comparison'].get('hybrid_rag', {})
            if 'error' not in hybrid_result:
                context_lengths.append(hybrid_result.get('context_length', 0))
                retrieved_docs_counts.append(hybrid_result.get('retrieved_docs', 0))
            
            # Attention quality (simplified metric)
            attn_analysis = qr['attention_analysis']
            if 'error' not in attn_analysis:
                attention_quality = attn_analysis.get('attention_std', 0)  # Higher std = more diverse attention
                attention_qualities.append(attention_quality)
        
        if context_lengths:
            metrics['avg_context_length'] = np.mean(context_lengths)
        if retrieved_docs_counts:
            metrics['avg_retrieved_docs'] = np.mean(retrieved_docs_counts)
        if attention_qualities:
            metrics['attention_quality'] = np.mean(attention_qualities)
        
        return metrics
    
    def _analyze_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results across all configurations."""
        analysis = {
            'best_configurations': {},
            'performance_comparison': {},
            'attention_insights': {},
            'retrieval_insights': {}
        }
        
        # Find best configurations
        config_metrics = {}
        for config_name, results in all_results.items():
            if results.get('success', False):
                config_metrics[config_name] = results['performance_metrics']
        
        if config_metrics:
            # Best by different metrics
            analysis['best_configurations'] = {
                'fastest': min(config_metrics.items(), key=lambda x: x[1].get('avg_processing_time', float('inf'))),
                'highest_success': max(config_metrics.items(), key=lambda x: x[1].get('success_rate', 0)),
                'best_attention': max(config_metrics.items(), key=lambda x: x[1].get('attention_quality', 0))
            }
        
        # Performance comparison
        if config_metrics:
            df = pd.DataFrame(config_metrics).T
            analysis['performance_comparison'] = {
                'summary_stats': df.describe().to_dict(),
                'correlations': df.corr().to_dict()
            }
        
        return analysis
    
    def _save_results(self, all_results: Dict[str, Any], analysis: Dict[str, Any]):
        """Save experiment results."""
        # Save raw results
        results_file = Path(self.config.output_dir) / 'experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'results': all_results,
                'analysis': analysis,
                'config': asdict(self.config)
            }, f, indent=2, default=str)
        
        # Create performance plots
        if self.config.create_plots:
            self._create_performance_plots(all_results, analysis)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def _create_performance_plots(self, all_results: Dict[str, Any], analysis: Dict[str, Any]):
        """Create performance visualization plots."""
        # Extract performance data
        config_names = []
        processing_times = []
        success_rates = []
        attention_qualities = []
        
        for config_name, results in all_results.items():
            if results.get('success', False):
                metrics = results['performance_metrics']
                config_names.append(config_name)
                processing_times.append(metrics.get('avg_processing_time', 0))
                success_rates.append(metrics.get('success_rate', 0))
                attention_qualities.append(metrics.get('attention_quality', 0))
        
        if not config_names:
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hybrid Attention RAG Performance Analysis', fontsize=16)
        
        # Processing time comparison
        axes[0, 0].bar(config_names, processing_times)
        axes[0, 0].set_title('Average Processing Time by Configuration')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        axes[0, 1].bar(config_names, success_rates)
        axes[0, 1].set_title('Success Rate by Configuration')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Attention quality comparison
        axes[1, 0].bar(config_names, attention_qualities)
        axes[1, 0].set_title('Attention Quality by Configuration')
        axes[1, 0].set_ylabel('Attention Quality (std)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Scatter plot: Processing time vs Success rate
        axes[1, 1].scatter(processing_times, success_rates, alpha=0.7)
        axes[1, 1].set_xlabel('Processing Time (seconds)')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Processing Time vs Success Rate')
        
        # Add configuration labels to scatter plot
        for i, config_name in enumerate(config_names):
            axes[1, 1].annotate(config_name, (processing_times[i], success_rates[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(self.config.output_dir) / 'performance_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {plot_file}")

def create_research_config() -> ExperimentConfig:
    """Create a comprehensive research configuration."""
    
    # Different attention configurations to test
    attention_configs = [
        {
            'window_size': 256,
            'num_landmark_tokens': 16,
            'max_retrieved_segments': 4,
            'hidden_size': 512,
            'num_attention_heads': 8
        },
        {
            'window_size': 512,
            'num_landmark_tokens': 32,
            'max_retrieved_segments': 8,
            'hidden_size': 768,
            'num_attention_heads': 12
        },
        {
            'window_size': 1024,
            'num_landmark_tokens': 64,
            'max_retrieved_segments': 16,
            'hidden_size': 1024,
            'num_attention_heads': 16
        }
    ]
    
    # Different retriever configurations to test
    retriever_configs = [
        {
            'query_embed_dim': 512,
            'doc_embed_dim': 512,
            'hidden_dim': 256,
            'num_candidates': 50,
            'top_k': 4
        },
        {
            'query_embed_dim': 768,
            'doc_embed_dim': 768,
            'hidden_dim': 512,
            'num_candidates': 100,
            'top_k': 8
        },
        {
            'query_embed_dim': 1024,
            'doc_embed_dim': 1024,
            'hidden_dim': 768,
            'num_candidates': 200,
            'top_k': 16
        }
    ]
    
    return ExperimentConfig(
        attention_configs=attention_configs,
        retriever_configs=retriever_configs,
        num_trials=3,
        test_queries=[
            "What is machine learning and what are its main types?",
            "How do transformer models work in NLP?",
            "What are the benefits of RAG systems?",
            "What challenges do long context models face?",
            "Compare supervised and unsupervised learning approaches",
            "Explain the attention mechanism in transformers",
            "What is the difference between BERT and GPT models?",
            "How does RAG reduce hallucination in language models?"
        ],
        evaluation_metrics=['processing_time', 'success_rate', 'context_quality', 'attention_quality'],
        save_results=True,
        output_dir='./research_results',
        create_plots=True
    )

def run_research_experiment():
    """Run the comprehensive research experiment."""
    print("🔬 Starting Hybrid Attention RAG Research Experiment")
    print("=" * 60)
    
    # Create research configuration
    config = create_research_config()
    
    # Create researcher
    researcher = HybridRAGResearcher(config)
    
    # Run experiments
    results = researcher.run_comprehensive_experiment()
    
    # Print summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    if 'analysis' in results and 'best_configurations' in results['analysis']:
        best_configs = results['analysis']['best_configurations']
        
        print("🏆 Best Configurations:")
        for metric, (config_name, metrics) in best_configs.items():
            print(f"  {metric}: {config_name}")
            print(f"    Value: {metrics}")
    
    print(f"\n📁 Results saved to: {config.output_dir}")
    print("✅ Research experiment completed!")
    
    return results

if __name__ == "__main__":
    run_research_experiment()
