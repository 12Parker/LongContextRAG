#!/usr/bin/env python3
"""
Compare Systems Against NarrativeQA

This script compares your base LLM and RAG systems against NarrativeQA questions
to evaluate performance differences on complex, long-form question-answering tasks.

NEW: Now includes Hybrid Retriever (BM25+Dense) system for comparison!

Usage:
    python compare_systems_narrativeqa.py
    python compare_systems_narrativeqa.py --num-questions 10
    python compare_systems_narrativeqa.py --systems base_llm,hybrid_bm25_dense
    python compare_systems_narrativeqa.py --systems hybrid_bm25_dense,hybrid_dense_only,hybrid_bm25_only
"""

import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_system_availability():
    """Test which systems are available."""
    print("🧪 Testing system availability...")
    
    available_systems = {}
    
    # Test Base LLM
    try:
        from examples.narrativeqa_base_llm import NarrativeQABaseLLM
        available_systems['base_llm'] = NarrativeQABaseLLM
        print("  ✅ Base LLM available")
    except ImportError as e:
        print(f"  ❌ Base LLM not available: {e}")
    
    # Test NarrativeQA RAG
    try:
        from examples.narrativeqa_rag_baseline import NarrativeQARAGBaseline
        available_systems['narrativeqa_rag'] = NarrativeQARAGBaseline
        print("  ✅ NarrativeQA RAG available")
    except ImportError as e:
        print(f"  ❌ NarrativeQA RAG not available: {e}")
    
    # Test NarrativeQA Hybrid RAG (with neural retriever support - OLD)
    try:
        from hybrid.narrativeqa_hybrid_rag_neural_retriever import NarrativeQAHybridRAG
        available_systems['narrativeqa_hybrid_rag_neural'] = NarrativeQAHybridRAG
        print("  ✅ NarrativeQA Hybrid RAG (with neural retriever) available")
    except ImportError as e:
        print(f"  ❌ NarrativeQA Hybrid RAG (neural) not available: {e}")
    
    # Test NarrativeQA Hybrid RAG with BM25+Dense (NEW - RECOMMENDED)
    try:
        from hybrid.narrativeqa_hybrid_rag_improved import NarrativeQAHybridRAG as HybridRAGImproved
        available_systems['hybrid_bm25_dense'] = HybridRAGImproved
        print("  ✅ NarrativeQA Hybrid RAG (BM25+Dense) available")
        
        # Also make mode-specific variants available
        available_systems['hybrid_dense_only'] = HybridRAGImproved
        available_systems['hybrid_bm25_only'] = HybridRAGImproved
        print("  ✅ Dense-only and BM25-only modes available")
    except ImportError as e:
        print(f"  ❌ NarrativeQA Hybrid RAG (BM25+Dense) not available: {e}")
        print(f"     Make sure narrativeqa_hybrid_rag_improved.py is in the hybrid/ directory")
    
    return available_systems

def test_single_question_with_system(question_data: Dict[str, Any], 
                                     system_name: str, 
                                     system_class, 
                                     retriever_checkpoint: str = None,
                                     expand_context: bool = False,
                                     context_window: int = 1) -> Dict[str, Any]:
    """Test a single NarrativeQA question with a specific system."""
    question = question_data['question']
    reference_answers = question_data['answers']
    
    print(f"\n🔍 Testing {system_name}: {question[:60]}...")
    
    start_time = time.time()
    
    try:
        # Initialize system based on type
        story_text = question_data.get('story', '')
        
        if system_name == 'base_llm':
            # Use NarrativeQA Base LLM
            system = system_class()
        
        elif system_name == 'narrativeqa_rag':
            # Use the story text from the question data
            system = system_class(db_path="./narrativeqa_vectordb", top_k_results=20, story_text=story_text)
        
        elif system_name == 'narrativeqa_hybrid_rag_neural':
            # Use the story text from the question data (old neural retriever version)
            system = system_class(
                db_path="./narrativeqa_hybrid_vectordb_neural", 
                top_k_results=20, 
                story_text=story_text,
                retriever_checkpoint=retriever_checkpoint
            )
        
        elif system_name == 'hybrid_bm25_dense':
            # NEW: Hybrid retriever with BM25+Dense
            system = system_class(
                chunk_size=1500,
                top_k_results=10,
                db_path="./narrativeqa_hybrid_bm25_dense",
                story_text=story_text,
                retrieval_mode='hybrid',
                hybrid_alpha=0.5  # Equal weight BM25 and Dense
            )
        
        elif system_name == 'hybrid_dense_only':
            # Dense retrieval only
            system = system_class(
                chunk_size=1500,
                top_k_results=10,
                db_path="./narrativeqa_dense_only",
                story_text=story_text,
                retrieval_mode='dense'
            )
        
        elif system_name == 'hybrid_bm25_only':
            # BM25 retrieval only
            system = system_class(
                chunk_size=1500,
                top_k_results=10,
                db_path="./narrativeqa_bm25_only",
                story_text=story_text,
                retrieval_mode='bm25'
            )
        
        elif system_name == 'standard_rag':
            system = system_class(db_path="./full_bookcorpus_db", top_k_results=10)
        
        elif system_name == 'hybrid_rag':
            system = system_class(db_path="./full_bookcorpus_db")
        
        else:
            raise ValueError(f"Unknown system: {system_name}")
        
        # Generate response
        if system_name == 'base_llm':
            # Use NarrativeQA Base LLM
            story = question_data.get('story', '')
            summary = question_data.get('summary', '')
            
            response = system.generate_response(question, story, summary)
            generated_answer = response['response']
        
        elif system_name in ['hybrid_bm25_dense', 'hybrid_dense_only', 'hybrid_bm25_only']:
            # Use new hybrid system with context expansion options
            response = system.generate_response(
                question,
                expand_context=expand_context,
                context_window=context_window
            )
            generated_answer = response['response']
        
        else:
            # Standard RAG systems
            response = system.generate_response(question)
            generated_answer = response['response']
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        answer_length = len(generated_answer)
        
        # Get metrics from response
        context_length = response.get('context_length', 0)
        context_tokens = response.get('context_tokens', 0)
        retrieved_docs = response.get('retrieved_docs', 0)
        retrieval_mode = response.get('retrieval_mode', 'unknown')
        
        # Calculate comprehensive evaluation metrics
        evaluation_metrics = {}
        relevance_score = 0.0
        
        if reference_answers:
            try:
                from evaluation.bleu_evaluator import BLEUEvaluator
                from evaluation.qa_metrics import QAMetrics
                
                # Get BLEU and ROUGE metrics
                bleu_evaluator = BLEUEvaluator()
                bleu_metrics = bleu_evaluator.evaluate_answer(generated_answer, reference_answers)
                
                # Get QA-specific metrics
                qa_evaluator = QAMetrics()
                qa_metrics = qa_evaluator.evaluate_answer(generated_answer, reference_answers)
                
                # Combine all metrics
                evaluation_metrics = {**bleu_metrics, **qa_metrics}
                
                # Use F1 score as the primary relevance metric (more appropriate for QA)
                relevance_score = qa_metrics.get('f1_score', 0.0)
                
            except ImportError:
                print("  ⚠️  Evaluation modules not available, falling back to simple word overlap")
                # Fallback to simple word overlap
                for ref_answer in reference_answers:
                    ref_words = set(ref_answer.lower().split())
                    gen_words = set(generated_answer.lower().split())
                    overlap = len(ref_words.intersection(gen_words))
                    if len(ref_words) > 0:
                        relevance_score = max(relevance_score, overlap / len(ref_words))
            except Exception as e:
                print(f"  ⚠️  Evaluation failed: {e}, using simple word overlap")
                # Fallback to simple word overlap
                for ref_answer in reference_answers:
                    ref_words = set(ref_answer.lower().split())
                    gen_words = set(generated_answer.lower().split())
                    overlap = len(ref_words.intersection(gen_words))
                    if len(ref_words) > 0:
                        relevance_score = max(relevance_score, overlap / len(ref_words))
        
        result = {
            'question_id': question_data.get('question_id', 'unknown'),
            'story_id': question_data.get('story_id', 'unknown'),
            'question': question,
            'reference_answers': reference_answers,
            'generated_answer': generated_answer,
            'response_time': elapsed_time,
            'answer_length': answer_length,
            'context_length': context_length,
            'context_tokens': context_tokens,
            'retrieved_docs': retrieved_docs,
            'relevance_score': relevance_score,
            'evaluation_metrics': evaluation_metrics,
            'method': system_name,
            'retrieval_mode': retrieval_mode
        }
        
        print(f"  ✅ Response generated in {elapsed_time:.2f}s")
        print(f"  📊 Answer length: {answer_length} chars")
        print(f"  📊 Context tokens: {context_tokens}")
        print(f"  📊 Retrieved docs: {retrieved_docs}")
        if retrieval_mode != 'unknown':
            print(f"  📊 Retrieval mode: {retrieval_mode}")
        print(f"  📊 Relevance score: {relevance_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error with {system_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'question_id': question_data.get('question_id', 'unknown'),
            'story_id': question_data.get('story_id', 'unknown'),
            'question': question,
            'reference_answers': reference_answers,
            'generated_answer': '',
            'response_time': 0.0,
            'answer_length': 0,
            'context_length': 0,
            'context_tokens': 0,
            'retrieved_docs': 0,
            'relevance_score': 0.0,
            'method': system_name,
            'error': str(e)
        }

def run_comparison_test(systems_to_test: List[str], 
                       num_questions: int = 5, 
                       subset: str = "test", 
                       retriever_checkpoint: str = None,
                       expand_context: bool = False,
                       context_window: int = 1):
    """Run comparison test between systems."""
    print(f"🚀 Comparing Systems Against NarrativeQA ({subset} subset)")
    print(f"📊 Systems to test: {', '.join(systems_to_test)}")
    print(f"📊 Number of questions: {num_questions}")
    if expand_context:
        print(f"📊 Context expansion: enabled (window={context_window})")
    print("=" * 60)
    
    # Test system availability
    available_systems = test_system_availability()
    
    # Filter to only available systems
    systems_to_test = [s for s in systems_to_test if s in available_systems]
    
    if not systems_to_test:
        print("❌ No systems available for testing")
        return False
    
    print(f"\n✅ Testing systems: {', '.join(systems_to_test)}")
    
    # Load NarrativeQA questions
    print(f"\n📚 Loading NarrativeQA questions...")
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("narrativeqa", split=subset)
        questions = []
        
        for i, example in enumerate(dataset):
            if i >= num_questions:
                break
            
            # Handle different dataset structures
            question_text = example.get('question', {})
            if isinstance(question_text, dict):
                question_text = question_text.get('text', '')
            
            answers = example.get('answers', [])
            if isinstance(answers, list) and len(answers) > 0:
                if isinstance(answers[0], dict):
                    answers = [ans.get('text', '') for ans in answers]
            
            # Handle document field (it's a dict with 'text' key)
            document_data = example.get('document', {})
            if isinstance(document_data, dict):
                story = document_data.get('text', '')
                story_id = document_data.get('id', f"story_{i}")
            else:
                story = str(document_data)
                story_id = f"story_{i}"
            
            questions.append({
                'question_id': example.get('question_id', f"q_{i}"),
                'story_id': story_id,
                'question': question_text,
                'answers': answers,
                'story': story,
                'summary': ''  # NarrativeQA doesn't have summaries
            })
        
        print(f"  ✅ Loaded {len(questions)} questions")
        
    except Exception as e:
        print(f"  ❌ Failed to load questions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test each system
    all_results = {}
    
    for system_name in systems_to_test:
        print(f"\n🧪 Testing {system_name.upper()}...")
        if retriever_checkpoint and 'neural' in system_name:
            print(f"  📦 Using trained checkpoint: {retriever_checkpoint}")
        
        system_class = available_systems[system_name]
        system_results = []
        
        for i, question_data in enumerate(questions):
            print(f"\n📋 Question {i+1}/{len(questions)}")
            result = test_single_question_with_system(
                question_data, 
                system_name, 
                system_class, 
                retriever_checkpoint,
                expand_context=expand_context,
                context_window=context_window
            )
            system_results.append(result)
        
        all_results[system_name] = system_results
    
    # Calculate and display comparison
    print(f"\n📊 SYSTEM COMPARISON RESULTS")
    print("=" * 80)
    
    # Create comparison table
    print(f"\n{'System':<25} {'Success':<10} {'Avg Time':<12} {'Avg Tokens':<12} {'Relevance':<12}")
    print("-" * 80)
    
    for system_name, results in all_results.items():
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            avg_response_time = sum(r['response_time'] for r in successful_results) / len(successful_results)
            avg_answer_length = sum(r['answer_length'] for r in successful_results) / len(successful_results)
            avg_context_tokens = sum(r['context_tokens'] for r in successful_results) / len(successful_results)
            avg_relevance_score = sum(r['relevance_score'] for r in successful_results) / len(successful_results)
            success_rate = len(successful_results) / len(results) * 100
            
            print(f"{system_name:<25} {success_rate:>6.1f}%   {avg_response_time:>8.2f}s   {avg_context_tokens:>9.0f}   {avg_relevance_score:>10.3f}")
        else:
            print(f"{system_name:<25} {'FAILED':<10}")
    
    # Detailed breakdown
    print(f"\n📊 DETAILED BREAKDOWN")
    print("=" * 80)
    
    for system_name, results in all_results.items():
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            avg_response_time = sum(r['response_time'] for r in successful_results) / len(successful_results)
            avg_answer_length = sum(r['answer_length'] for r in successful_results) / len(successful_results)
            avg_context_tokens = sum(r['context_tokens'] for r in successful_results) / len(successful_results)
            avg_relevance_score = sum(r['relevance_score'] for r in successful_results) / len(successful_results)
            success_rate = len(successful_results) / len(results) * 100
            
            print(f"\n{system_name.upper()}:")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Avg response time: {avg_response_time:.2f}s")
            print(f"  Avg answer length: {avg_answer_length:.0f} chars")
            print(f"  Avg context tokens: {avg_context_tokens:.0f}")
            print(f"  Avg relevance score: {avg_relevance_score:.3f}")
            
            # Show retrieval mode if available
            modes = [r.get('retrieval_mode', 'unknown') for r in successful_results]
            if modes and modes[0] != 'unknown':
                print(f"  Retrieval mode: {modes[0]}")
        else:
            print(f"\n{system_name.upper()}: ❌ No successful responses")
    
    # Show sample responses
    print(f"\n📋 SAMPLE RESPONSES")
    print("=" * 80)
    
    for i in range(min(2, len(questions))):  # Show first 2 questions
        question_data = questions[i]
        print(f"\nQuestion {i+1}: {question_data['question'][:80]}...")
        print(f"Reference: {question_data['answers'][0][:100]}...")
        
        for system_name in systems_to_test:
            if system_name in all_results:
                result = all_results[system_name][i]
                if 'error' not in result:
                    print(f"\n{system_name.upper()}:")
                    print(f"  {result['generated_answer'][:150]}...")
                    print(f"  Relevance: {result['relevance_score']:.3f} | Time: {result['response_time']:.2f}s")
                else:
                    print(f"\n{system_name.upper()}: ❌ Error - {result['error'][:50]}...")
    
    # Highlight best system
    print(f"\n🏆 WINNER ANALYSIS")
    print("=" * 80)
    
    best_relevance = 0.0
    best_relevance_system = None
    best_speed = float('inf')
    best_speed_system = None
    best_efficiency = 0.0
    best_efficiency_system = None
    
    for system_name, results in all_results.items():
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            avg_relevance = sum(r['relevance_score'] for r in successful_results) / len(successful_results)
            avg_time = sum(r['response_time'] for r in successful_results) / len(successful_results)
            efficiency = avg_relevance / avg_time if avg_time > 0 else 0
            
            if avg_relevance > best_relevance:
                best_relevance = avg_relevance
                best_relevance_system = system_name
            
            if avg_time < best_speed:
                best_speed = avg_time
                best_speed_system = system_name
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_efficiency_system = system_name
    
    if best_relevance_system:
        print(f"🥇 Best Relevance: {best_relevance_system} ({best_relevance:.3f})")
    if best_speed_system:
        print(f"⚡ Fastest: {best_speed_system} ({best_speed:.2f}s)")
    if best_efficiency_system:
        print(f"💎 Best Efficiency (relevance/time): {best_efficiency_system} ({best_efficiency:.3f})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"system_comparison_narrativeqa_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/system_comparisons")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / filename
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare Systems Against NarrativeQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare hybrid retriever modes (RECOMMENDED)
  python compare_systems_narrativeqa.py --systems hybrid_bm25_dense,hybrid_dense_only,hybrid_bm25_only
  
  # Compare all available systems with 5 questions
  python compare_systems_narrativeqa.py
  
  # Compare specific systems with 10 questions
  python compare_systems_narrativeqa.py --systems base_llm,hybrid_bm25_dense --num-questions 10
  
  # Test with context expansion
  python compare_systems_narrativeqa.py --systems hybrid_bm25_dense --expand-context --context-window 1
  
  # Compare on validation set
  python compare_systems_narrativeqa.py --subset validation --num-questions 15
        """
    )
    
    parser.add_argument("--systems", type=str, default="hybrid_bm25_dense,narrativeqa_rag",
                       help="Comma-separated list of systems to test (default: hybrid_bm25_dense,narrativeqa_rag)")
    parser.add_argument("--num-questions", type=int, default=5,
                       help="Number of questions to test (default: 5)")
    parser.add_argument("--subset", type=str, default="test",
                       choices=["train", "test", "validation"],
                       help="Dataset subset to use (default: test)")
    parser.add_argument("--retriever-checkpoint", type=str, default=None,
                       help="Path to trained neural retriever checkpoint (for neural hybrid RAG)")
    parser.add_argument("--expand-context", action="store_true",
                       help="Enable context expansion (include neighboring chunks)")
    parser.add_argument("--context-window", type=int, default=1,
                       help="Number of neighboring chunks to include (default: 1)")
    
    args = parser.parse_args()
    
    # Parse systems to test
    systems_to_test = [s.strip() for s in args.systems.split(',')]
    
    # Run the comparison
    success = run_comparison_test(
        systems_to_test=systems_to_test,
        num_questions=args.num_questions,
        subset=args.subset,
        retriever_checkpoint=args.retriever_checkpoint,
        expand_context=args.expand_context,
        context_window=args.context_window
    )
    
    if success:
        print("\n🎉 System comparison completed successfully!")
        print("\n💡 RECOMMENDATIONS:")
        print("1. 'hybrid_bm25_dense' - Best balance of accuracy and speed (no training)")
        print("2. 'hybrid_dense_only' - Good for semantic queries")
        print("3. 'hybrid_bm25_only' - Good for keyword queries")
        print("4. Use --expand-context for better handling of fragmented answers")
        print("\nNext steps:")
        print("1. Review the comparison results")
        print("2. Identify which system performs best for your use case")
        print("3. Tune alpha parameter (0.3-0.7) for hybrid systems")
    else:
        print("\n❌ System comparison failed. Please check your setup.")

if __name__ == "__main__":
    main()