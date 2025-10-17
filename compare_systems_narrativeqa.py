#!/usr/bin/env python3
"""
Compare Systems Against NarrativeQA

This script compares your base LLM and RAG systems against NarrativeQA questions
to evaluate performance differences on complex, long-form question-answering tasks.

Usage:
    python compare_systems_narrativeqa.py
    python compare_systems_narrativeqa.py --num-questions 10
    python compare_systems_narrativeqa.py --systems base_llm,standard_rag
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
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_system_availability():
    """Test which systems are available."""
    print("🧪 Testing system availability...")
    
    available_systems = {}
    
    # Test Base LLM
    try:
        from examples.bookcorpus_raw_llm_baseline import BookCorpusRawLLMBaseline
        available_systems['base_llm'] = BookCorpusRawLLMBaseline
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
    
    # Test Standard RAG (BookCorpus-based)
    try:
        from examples.standard_rag_baseline import StandardRAGBaseline
        available_systems['standard_rag'] = StandardRAGBaseline
        print("  ✅ Standard RAG (BookCorpus) available")
    except ImportError as e:
        print(f"  ❌ Standard RAG not available: {e}")
    
    # Test Hybrid RAG
    try:
        from hybrid.working_hybrid_rag import WorkingHybridRAG
        available_systems['hybrid_rag'] = WorkingHybridRAG
        print("  ✅ Hybrid RAG available")
    except ImportError as e:
        print(f"  ❌ Hybrid RAG not available: {e}")
    
    return available_systems

def test_single_question_with_system(question_data: Dict[str, Any], system_name: str, system_class) -> Dict[str, Any]:
    """Test a single NarrativeQA question with a specific system."""
    question = question_data['question']
    reference_answers = question_data['answers']
    
    print(f"\n🔍 Testing {system_name}: {question[:60]}...")
    
    start_time = time.time()
    
    try:
        # Initialize system if needed
        if system_name == 'base_llm':
            # Use simple LLM with NarrativeQA stories
            from core.config import config
            from langchain_openai import ChatOpenAI
            system = ChatOpenAI(
                model=config.OPENAI_MODEL,
                openai_api_key=config.OPENAI_API_KEY,
                temperature=0.1
            )
        elif system_name == 'narrativeqa_rag':
            # Use the story text from the question data
            story_text = question_data.get('story', '')
            system = system_class(db_path="./narrativeqa_vectordb", top_k_results=5, story_text=story_text)
        elif system_name == 'standard_rag':
            system = system_class(db_path="./full_bookcorpus_db", top_k_results=10)
        elif system_name == 'hybrid_rag':
            system = system_class(db_path="./full_bookcorpus_db")
        else:
            raise ValueError(f"Unknown system: {system_name}")
        
        # Generate response
        if system_name == 'base_llm':
            # Use NarrativeQA story and summary for base LLM
            story = question_data.get('story', '')
            summary = question_data.get('summary', '')
            
            # Create context from NarrativeQA story and summary
            context_parts = []
            if summary:
                context_parts.append(f"Summary: {summary}")
            if story:
                # Truncate story to fit in context window
                max_story_length = 8000
                if len(story) > max_story_length:
                    story = story[:max_story_length] + "..."
                context_parts.append(f"Story: {story}")
            
            context = "\n\n".join(context_parts) if context_parts else "No context available."
            
            # Create prompt
            prompt = f"""Based on the following story and summary, answer the question.

{context}

Question: {question}

Please provide a comprehensive answer based on the story content. If the story doesn't contain enough information to answer the question, please say so and provide what information you can from the available context."""

            # Generate response
            response = system.invoke(prompt)
            generated_answer = response.content if hasattr(response, 'content') else str(response)
        elif system_name == 'narrativeqa_rag':
            # Use NarrativeQA RAG system
            response = system.generate_response(question)
            generated_answer = response['response']
        else:
            response = system.generate_response(question)
            generated_answer = response['response']
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        answer_length = len(generated_answer)
        
        # Handle different response formats
        if system_name == 'base_llm':
            # For base LLM, calculate metrics from the story and prompt
            story = question_data.get('story', '')
            context_length = len(story)
            context_tokens = len(prompt.split()) if 'prompt' in locals() else 0
            retrieved_docs = 0
        else:
            context_length = response.get('context_length', 0)
            context_tokens = response.get('context_tokens', 0)
            retrieved_docs = response.get('retrieved_docs', 0)
        
        # Simple relevance check
        relevance_score = 0.0
        if reference_answers:
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
            'method': system_name
        }
        
        print(f"  ✅ Response generated in {elapsed_time:.2f}s")
        print(f"  📊 Answer length: {answer_length} chars")
        print(f"  📊 Context tokens: {context_tokens}")
        print(f"  📊 Retrieved docs: {retrieved_docs}")
        print(f"  📊 Relevance score: {relevance_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error with {system_name}: {e}")
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

def run_comparison_test(systems_to_test: List[str], num_questions: int = 5, subset: str = "test"):
    """Run comparison test between systems."""
    print(f"🚀 Comparing Systems Against NarrativeQA ({subset} subset)")
    print(f"📊 Systems to test: {', '.join(systems_to_test)}")
    print(f"📊 Number of questions: {num_questions}")
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
        return False
    
    # Test each system
    all_results = {}
    
    for system_name in systems_to_test:
        print(f"\n🧪 Testing {system_name.upper()}...")
        system_class = available_systems[system_name]
        system_results = []
        
        for i, question_data in enumerate(questions):
            print(f"\n📋 Question {i+1}/{len(questions)}")
            result = test_single_question_with_system(question_data, system_name, system_class)
            system_results.append(result)
        
        all_results[system_name] = system_results
    
    # Calculate and display comparison
    print(f"\n📊 SYSTEM COMPARISON RESULTS")
    print("=" * 60)
    
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
        else:
            print(f"\n{system_name.upper()}: ❌ No successful responses")
    
    # Show sample responses
    print(f"\n📋 SAMPLE RESPONSES")
    print("=" * 60)
    
    for i in range(min(2, len(questions))):  # Show first 2 questions
        question_data = questions[i]
        print(f"\nQuestion {i+1}: {question_data['question'][:50]}...")
        print(f"Reference: {question_data['answers'][0][:100]}...")
        
        for system_name in systems_to_test:
            if system_name in all_results:
                result = all_results[system_name][i]
                if 'error' not in result:
                    print(f"\n{system_name.upper()}:")
                    print(f"  {result['generated_answer'][:100]}...")
                    print(f"  Relevance: {result['relevance_score']:.3f}")
                else:
                    print(f"\n{system_name.upper()}: ❌ Error - {result['error']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"system_comparison_narrativeqa_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare Systems Against NarrativeQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all available systems with 5 questions
  python compare_systems_narrativeqa.py
  
  # Compare specific systems with 10 questions
  python compare_systems_narrativeqa.py --systems base_llm,standard_rag --num-questions 10
  
  # Compare on validation set
  python compare_systems_narrativeqa.py --subset validation --num-questions 15
        """
    )
    
    parser.add_argument("--systems", type=str, default="base_llm,narrativeqa_rag",
                       help="Comma-separated list of systems to test (default: base_llm,narrativeqa_rag)")
    parser.add_argument("--num-questions", type=int, default=5,
                       help="Number of questions to test (default: 5)")
    parser.add_argument("--subset", type=str, default="test",
                       choices=["train", "test", "validation"],
                       help="Dataset subset to use (default: test)")
    
    args = parser.parse_args()
    
    # Parse systems to test
    systems_to_test = [s.strip() for s in args.systems.split(',')]
    
    # Run the comparison
    success = run_comparison_test(
        systems_to_test=systems_to_test,
        num_questions=args.num_questions,
        subset=args.subset
    )
    
    if success:
        print("\n🎉 System comparison completed successfully!")
        print("\nNext steps:")
        print("1. Review the comparison results")
        print("2. Identify which system performs best")
        print("3. Use results to improve your RAG systems")
    else:
        print("\n❌ System comparison failed. Please check your setup.")

if __name__ == "__main__":
    main()
