#!/usr/bin/env python3
"""
Setup NarrativeQA Benchmark Integration

This script helps you set up and test the NarrativeQA benchmark integration
for evaluating your RAG systems on complex, long-form question-answering tasks.

Usage:
    python setup_narrativeqa.py --install-deps    # Install required dependencies
    python setup_narrativeqa.py --test-dataset    # Test dataset loading
    python setup_narrativeqa.py --run-evaluation  # Run full evaluation
    python setup_narrativeqa.py --quick-test      # Quick test with 5 questions
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import subprocess
import json
from datetime import datetime

def install_dependencies():
    """Install required dependencies for NarrativeQA evaluation."""
    print("🔧 Installing NarrativeQA evaluation dependencies...")
    
    try:
        # Install from requirements file
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            "requirements_narrativeqa.txt"
        ], check=True)
        
        # Download NLTK data
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_dataset_loading():
    """Test loading the NarrativeQA dataset."""
    print("🧪 Testing NarrativeQA dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Test loading test subset
        print("Loading test subset...")
        test_dataset = load_dataset("narrativeqa", split="test")
        print(f"✅ Test dataset loaded: {len(test_dataset)} examples")
        
        # Show sample
        sample = test_dataset[0]
        print(f"\n📊 Sample Question:")
        print(f"Question: {sample['question']}")
        print(f"Answers: {sample['answers']}")
        print(f"Story ID: {sample['story_id']}")
        
        # Test loading validation subset
        print("\nLoading validation subset...")
        val_dataset = load_dataset("narrativeqa", split="validation")
        print(f"✅ Validation dataset loaded: {len(val_dataset)} examples")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

def run_quick_test():
    """Run a quick test with 5 questions."""
    print("🚀 Running quick NarrativeQA test...")
    
    try:
        # Run a quick comparison test
        print("Running quick comparison test...")
        import subprocess
        result = subprocess.run([sys.executable, "run.py", "compare-systems", "--systems", "base_llm", "--num-questions", "2"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Quick test passed")
        else:
            print("❌ Quick test failed")
            print(result.stdout)
            print(result.stderr)
            return False
        
        print("\n✅ Quick test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def run_full_evaluation():
    """Run full NarrativeQA evaluation."""
    print("🚀 Running full NarrativeQA evaluation...")
    
    try:
        from evaluation.narrativeqa_evaluator import NarrativeQAEvaluator
        
        # Initialize evaluator
        evaluator = NarrativeQAEvaluator(
            db_path="./full_bookcorpus_db",
            max_questions=50,  # Start with 50 questions
            subset="test"
        )
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        if 'error' in results:
            print(f"❌ Evaluation failed: {results['error']}")
            return False
        
        # Print results
        print("\n📊 EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total questions: {results['total_questions']}")
        print(f"Best system: {results['best_system']}")
        
        for system, metrics in results.items():
            if system not in ['total_questions', 'best_system']:
                print(f"\n{system.upper()}:")
                print(f"  Quality Score: {metrics['avg_quality_score']:.3f}")
                print(f"  BLEU Score: {metrics['avg_bleu']:.3f}")
                print(f"  ROUGE Score: {metrics['avg_rouge']:.3f}")
                print(f"  Exact Match Rate: {metrics['exact_match_rate']:.3f}")
        
        print("\n✅ Full evaluation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Full evaluation failed: {e}")
        return False

def show_usage_examples():
    """Show usage examples for NarrativeQA evaluation."""
    print("📚 NARRATIVEQA USAGE EXAMPLES")
    print("=" * 50)
    
    print("\n1. Basic Evaluation:")
    print("   python evaluation/narrativeqa_evaluator.py")
    
    print("\n2. Custom Parameters:")
    print("   python evaluation/narrativeqa_evaluator.py --subset test --num-questions 100")
    
    print("\n3. Validation Set:")
    print("   python evaluation/narrativeqa_evaluator.py --subset validation --num-questions 50")
    
    print("\n4. Custom Database Path:")
    print("   python evaluation/narrativeqa_evaluator.py --db-path ./my_database")
    
    print("\n5. Quick Test (5 questions):")
    print("   python setup_narrativeqa.py --quick-test")
    
    print("\n6. Full Evaluation (50 questions):")
    print("   python setup_narrativeqa.py --run-evaluation")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup NarrativeQA Benchmark Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install dependencies
  python setup_narrativeqa.py --install-deps
  
  # Test dataset loading
  python setup_narrativeqa.py --test-dataset
  
  # Run quick test
  python setup_narrativeqa.py --quick-test
  
  # Run full evaluation
  python setup_narrativeqa.py --run-evaluation
  
  # Show usage examples
  python setup_narrativeqa.py --examples
        """
    )
    
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install required dependencies")
    parser.add_argument("--test-dataset", action="store_true", 
                       help="Test dataset loading")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick test with 5 questions")
    parser.add_argument("--run-evaluation", action="store_true", 
                       help="Run full evaluation")
    parser.add_argument("--examples", action="store_true", 
                       help="Show usage examples")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
    
    elif args.test_dataset:
        test_dataset_loading()
    
    elif args.quick_test:
        run_quick_test()
    
    elif args.run_evaluation:
        run_full_evaluation()
    
    elif args.examples:
        show_usage_examples()
    
    else:
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python setup_narrativeqa.py --install-deps    # Install dependencies")
        print("  python setup_narrativeqa.py --test-dataset    # Test dataset loading")
        print("  python setup_narrativeqa.py --quick-test      # Quick test")
        print("  python setup_narrativeqa.py --run-evaluation # Full evaluation")

if __name__ == "__main__":
    main()
