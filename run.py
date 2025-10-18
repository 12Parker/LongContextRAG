#!/usr/bin/env python3
"""
Main Script Runner for LongContextRAG

This script provides easy access to all organized scripts in the project.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_script(script_path, args=None):
    """Run a script with optional arguments."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running script: {e}")
        return e.returncode
    except FileNotFoundError:
        print(f"❌ Script not found: {script_path}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="LongContextRAG Script Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:

SETUP:
  setup-vectordb     Build BookCorpus vector database
  setup-narrativeqa  Setup NarrativeQA evaluation
  monitor-build      Monitor database build progress

COMPARISON:
  compare-systems    Compare RAG systems on NarrativeQA
  compare-bleu       Compare systems with BLEU metrics
  compare-qa         Compare systems with QA metrics

ANALYSIS:
  analyze-qa         Analyze QA evaluation results
  analyze-bleu       Analyze BLEU evaluation results
  analyze-comparison Analyze system comparison results
  summarize-results  Summarize base LLM results

EXAMPLES:
  python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag --num-questions 5
  python run.py analyze-qa
  python run.py setup-vectordb --build-medium
        """
    )
    
    parser.add_argument('command', help='Command to run')
    parser.add_argument('args', nargs='*', help='Additional arguments for the command')
    
    args = parser.parse_args()
    
    # Define script mappings
    scripts = {
        # Setup scripts
        'setup-vectordb': 'scripts/setup/setup_full_vectordb.py',
        'setup-narrativeqa': 'scripts/setup/setup_narrativeqa.py',
        'monitor-build': 'scripts/setup/monitor_build.py',
        
        # Comparison scripts
        'compare-systems': 'scripts/comparison/compare_systems_narrativeqa.py',
        
        # Analysis scripts
        'analyze-qa': 'scripts/analysis/analyze_qa_metrics.py',
        'analyze-bleu': 'scripts/analysis/analyze_bleu_results.py',
        'analyze-comparison': 'scripts/analysis/analyze_narrativeqa_comparison.py',
        'summarize-results': 'scripts/analysis/summarize_base_llm_results.py',
    }
    
    if args.command not in scripts:
        print(f"❌ Unknown command: {args.command}")
        print("\nAvailable commands:")
        for cmd in scripts.keys():
            print(f"  {cmd}")
        return 1
    
    script_path = project_root / scripts[args.command]
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1
    
    return run_script(script_path, args.args)

if __name__ == "__main__":
    sys.exit(main())
