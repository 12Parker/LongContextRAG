#!/usr/bin/env python3
"""
Quick RAG Evaluation Script

A simplified script to quickly compare RAG vs Base LLM performance
with clear metrics and visual comparisons.
"""

import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from hybrid.working_hybrid_rag import WorkingHybridRAG
from core.long_context_config import LongContextManager, ContextSize

def quick_evaluation():
    """Run a quick evaluation comparing RAG vs Base LLM."""
    print("🚀 Quick RAG vs Base LLM Evaluation")
    print("=" * 60)
    
    # Initialize systems
    print("🔧 Initializing systems...")
    
    # Create hybrid RAG system
    vectordb_config = {
        'db_path': './vector_store_eval',
        'collection_name': 'evaluation_test',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'batch_size': 50
    }
    
    hybrid_rag = WorkingHybridRAG(
        use_hybrid_attention=True,
        vectordb_config=vectordb_config,
        max_retrieved_docs=10
    )
    
    # Create vector store
    print("📚 Creating vector store...")
    hybrid_rag.create_vectorstore(
        use_vectordb=True,
        num_documents=3000  # Smaller for quick testing
    )
    
    # Test queries
    test_queries = [
        "What are the main themes in the stories?",
        "Analyze the character development patterns",
        "Compare the narrative techniques used",
        "What conflicts and resolutions are present?",
        "How do the authors develop atmosphere?"
    ]
    
    print(f"\n🔍 Testing {len(test_queries)} queries...")
    print("=" * 60)
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 50)
        
        # Test Base LLM
        print("🤖 Testing Base LLM...")
        start_time = time.time()
        try:
            base_response = hybrid_rag.base_rag.generate_response(query, use_rag=False)
            base_time = time.time() - start_time
            base_success = True
        except Exception as e:
            base_response = {'response': f'Error: {e}', 'method': 'base_llm'}
            base_time = time.time() - start_time
            base_success = False
        
        # Test Base RAG
        print("📚 Testing Base RAG...")
        start_time = time.time()
        try:
            rag_response = hybrid_rag.base_rag.generate_response(query, use_rag=True)
            rag_time = time.time() - start_time
            rag_success = True
        except Exception as e:
            rag_response = {'response': f'Error: {e}', 'method': 'base_rag'}
            rag_time = time.time() - start_time
            rag_success = False
        
        # Test Hybrid RAG
        print("🧠 Testing Hybrid RAG...")
        start_time = time.time()
        try:
            hybrid_response = hybrid_rag.generate_response(query, task_type='qa')
            hybrid_time = time.time() - start_time
            hybrid_success = True
        except Exception as e:
            hybrid_response = {'response': f'Error: {e}', 'method': 'hybrid_rag'}
            hybrid_time = time.time() - start_time
            hybrid_success = False
        
        # Collect metrics
        result = {
            'query': query,
            'base_llm': {
                'response_length': len(base_response.get('response', '')),
                'response_time': base_time,
                'context_length': 0,
                'retrieved_docs': 0,
                'success': base_success,
                'method': base_response.get('method', 'base_llm')
            },
            'base_rag': {
                'response_length': len(rag_response.get('response', '')),
                'response_time': rag_time,
                'context_length': rag_response.get('context_length', 0),
                'retrieved_docs': rag_response.get('retrieved_docs', 0),
                'success': rag_success,
                'method': rag_response.get('method', 'base_rag')
            },
            'hybrid_rag': {
                'response_length': len(hybrid_response.get('response', '')),
                'response_time': hybrid_time,
                'context_length': hybrid_response.get('context_length', 0),
                'retrieved_docs': hybrid_response.get('retrieved_docs', 0),
                'success': hybrid_success,
                'method': hybrid_response.get('method', 'hybrid_rag')
            }
        }
        
        results.append(result)
        
        # Print immediate results
        print(f"📊 Results:")
        print(f"   Base LLM:    {result['base_llm']['response_length']:>4} chars, "
              f"{result['base_llm']['response_time']:>5.2f}s, "
              f"{result['base_llm']['context_length']:>4} context")
        print(f"   Base RAG:    {result['base_rag']['response_length']:>4} chars, "
              f"{result['base_rag']['response_time']:>5.2f}s, "
              f"{result['base_rag']['context_length']:>4} context, "
              f"{result['base_rag']['retrieved_docs']} docs")
        print(f"   Hybrid RAG:  {result['hybrid_rag']['response_length']:>4} chars, "
              f"{result['hybrid_rag']['response_time']:>5.2f}s, "
              f"{result['hybrid_rag']['context_length']:>4} context, "
              f"{result['hybrid_rag']['retrieved_docs']} docs")
        
        # Show response previews
        print(f"\n💬 Response Previews:")
        base_preview = base_response.get('response', '')[:150]
        rag_preview = rag_response.get('response', '')[:150]
        hybrid_preview = hybrid_response.get('response', '')[:150]
        
        print(f"   Base LLM:    {base_preview}...")
        print(f"   Base RAG:    {rag_preview}...")
        print(f"   Hybrid RAG:  {hybrid_preview}...")
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print("📈 COMPREHENSIVE SUMMARY")
    print(f"{'='*60}")
    
    # Calculate averages
    base_avg_length = sum(r['base_llm']['response_length'] for r in results) / len(results)
    rag_avg_length = sum(r['base_rag']['response_length'] for r in results) / len(results)
    hybrid_avg_length = sum(r['hybrid_rag']['response_length'] for r in results) / len(results)
    
    base_avg_time = sum(r['base_llm']['response_time'] for r in results) / len(results)
    rag_avg_time = sum(r['base_rag']['response_time'] for r in results) / len(results)
    hybrid_avg_time = sum(r['hybrid_rag']['response_time'] for r in results) / len(results)
    
    base_avg_context = sum(r['base_llm']['context_length'] for r in results) / len(results)
    rag_avg_context = sum(r['base_rag']['context_length'] for r in results) / len(results)
    hybrid_avg_context = sum(r['hybrid_rag']['context_length'] for r in results) / len(results)
    
    base_success_rate = sum(1 for r in results if r['base_llm']['success']) / len(results) * 100
    rag_success_rate = sum(1 for r in results if r['base_rag']['success']) / len(results) * 100
    hybrid_success_rate = sum(1 for r in results if r['hybrid_rag']['success']) / len(results) * 100
    
    print(f"📊 Average Response Length:")
    print(f"   Base LLM:    {base_avg_length:>6.0f} characters")
    print(f"   Base RAG:    {rag_avg_length:>6.0f} characters ({rag_avg_length/base_avg_length:>4.2f}x)")
    print(f"   Hybrid RAG:  {hybrid_avg_length:>6.0f} characters ({hybrid_avg_length/base_avg_length:>4.2f}x)")
    
    print(f"\n⏱️  Average Response Time:")
    print(f"   Base LLM:    {base_avg_time:>6.2f} seconds")
    print(f"   Base RAG:    {rag_avg_time:>6.2f} seconds ({rag_avg_time/base_avg_time:>4.2f}x)")
    print(f"   Hybrid RAG:  {hybrid_avg_time:>6.2f} seconds ({hybrid_avg_time/base_avg_time:>4.2f}x)")
    
    print(f"\n📄 Average Context Usage:")
    print(f"   Base LLM:    {base_avg_context:>6.0f} characters")
    print(f"   Base RAG:    {rag_avg_context:>6.0f} characters")
    print(f"   Hybrid RAG:  {hybrid_avg_context:>6.0f} characters")
    
    print(f"\n✅ Success Rates:")
    print(f"   Base LLM:    {base_success_rate:>5.1f}%")
    print(f"   Base RAG:    {rag_success_rate:>5.1f}%")
    print(f"   Hybrid RAG:  {hybrid_success_rate:>5.1f}%")
    
    # Key insights
    print(f"\n💡 Key Insights:")
    
    if hybrid_avg_length > base_avg_length * 1.3:
        print(f"   ✅ Hybrid RAG provides {hybrid_avg_length/base_avg_length:.1f}x more detailed responses")
    
    if hybrid_avg_context > 0:
        print(f"   ✅ Hybrid RAG uses {hybrid_avg_context:.0f} characters of retrieved context")
    
    if hybrid_avg_time < base_avg_time * 3:
        print(f"   ✅ Performance overhead is reasonable ({hybrid_avg_time/base_avg_time:.1f}x)")
    
    if hybrid_success_rate >= base_success_rate:
        print(f"   ✅ Hybrid RAG maintains reliability ({hybrid_success_rate:.1f}% success)")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/quick_evaluation_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'summary': {
                'base_llm_avg_length': base_avg_length,
                'rag_avg_length': rag_avg_length,
                'hybrid_rag_avg_length': hybrid_avg_length,
                'base_llm_avg_time': base_avg_time,
                'rag_avg_time': rag_avg_time,
                'hybrid_rag_avg_time': hybrid_avg_time,
                'base_llm_avg_context': base_avg_context,
                'rag_avg_context': rag_avg_context,
                'hybrid_rag_avg_context': hybrid_avg_context,
                'base_llm_success_rate': base_success_rate,
                'rag_success_rate': rag_success_rate,
                'hybrid_rag_success_rate': hybrid_success_rate
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Final recommendation
    print(f"\n🎯 Recommendation:")
    if hybrid_avg_length > base_avg_length * 1.2 and hybrid_avg_context > 0:
        print(f"   ✅ Hybrid RAG is working better than Base LLM!")
        print(f"   📈 Provides {hybrid_avg_length/base_avg_length:.1f}x more detailed responses")
        print(f"   📚 Uses {hybrid_avg_context:.0f} characters of relevant context")
        print(f"   ⚡ Performance overhead: {hybrid_avg_time/base_avg_time:.1f}x")
    else:
        print(f"   ⚠️  Results are mixed - may need more tuning or better queries")

if __name__ == "__main__":
    quick_evaluation()
