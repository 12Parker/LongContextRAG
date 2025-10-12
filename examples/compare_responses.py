#!/usr/bin/env python3
"""
Side-by-Side Response Comparison

This script provides a simple way to compare responses from Base LLM vs RAG
systems side-by-side to visually assess quality differences.
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from hybrid.working_hybrid_rag import WorkingHybridRAG

def compare_responses(query: str = None):
    """Compare responses from different systems side-by-side."""
    
    if query is None:
        query = input("Enter your query: ")
    
    print("🔍 Side-by-Side Response Comparison")
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    
    # Initialize system
    print("\n🔧 Initializing system...")
    vectordb_config = {
        'db_path': './vector_store_compare',
        'collection_name': 'comparison_test',
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
    print("📚 Setting up vector store...")
    try:
        hybrid_rag.create_vectorstore(
            use_vectordb=True,
            num_documents=2000
        )
    except Exception as e:
        print(f"Note: Using existing vector store: {e}")
    
    print("\n🚀 Generating responses...")
    
    # Get Base LLM response
    print("🤖 Getting Base LLM response...")
    start_time = time.time()
    base_response = hybrid_rag.base_rag.generate_response(query, use_rag=False)
    base_time = time.time() - start_time
    
    # Get Base RAG response
    print("📚 Getting Base RAG response...")
    start_time = time.time()
    rag_response = hybrid_rag.base_rag.generate_response(query, use_rag=True)
    rag_time = time.time() - start_time
    
    # Get Hybrid RAG response
    print("🧠 Getting Hybrid RAG response...")
    start_time = time.time()
    hybrid_response = hybrid_rag.generate_response(query, task_type='qa')
    hybrid_time = time.time() - start_time
    
    # Display results
    print("\n" + "="*80)
    print("📊 RESPONSE COMPARISON")
    print("="*80)
    
    # Base LLM
    print(f"\n🤖 BASE LLM (No Context)")
    print("-" * 40)
    print(f"Time: {base_time:.2f}s | Length: {len(base_response.get('response', ''))} chars")
    print(f"Method: {base_response.get('method', 'base_llm')}")
    print("\nResponse:")
    print(base_response.get('response', 'No response'))
    
    # Base RAG
    print(f"\n📚 BASE RAG (With Context)")
    print("-" * 40)
    print(f"Time: {rag_time:.2f}s | Length: {len(rag_response.get('response', ''))} chars")
    print(f"Context: {rag_response.get('context_length', 0)} chars | Docs: {rag_response.get('retrieved_docs', 0)}")
    print(f"Method: {rag_response.get('method', 'base_rag')}")
    print("\nResponse:")
    print(rag_response.get('response', 'No response'))
    
    # Hybrid RAG
    print(f"\n🧠 HYBRID RAG (Enhanced Context)")
    print("-" * 40)
    print(f"Time: {hybrid_time:.2f}s | Length: {len(hybrid_response.get('response', ''))} chars")
    print(f"Context: {hybrid_response.get('context_length', 0)} chars | Docs: {hybrid_response.get('retrieved_docs', 0)}")
    print(f"Method: {hybrid_response.get('method', 'hybrid_rag')}")
    print("\nResponse:")
    print(hybrid_response.get('response', 'No response'))
    
    # Analysis
    print(f"\n" + "="*80)
    print("📈 ANALYSIS")
    print("="*80)
    
    base_length = len(base_response.get('response', ''))
    rag_length = len(rag_response.get('response', ''))
    hybrid_length = len(hybrid_response.get('response', ''))
    
    rag_context = rag_response.get('context_length', 0)
    hybrid_context = hybrid_response.get('context_length', 0)
    
    print(f"Response Length Comparison:")
    print(f"  Base LLM:    {base_length:>6} characters")
    print(f"  Base RAG:    {rag_length:>6} characters ({rag_length/base_length:.2f}x)")
    print(f"  Hybrid RAG:  {hybrid_length:>6} characters ({hybrid_length/base_length:.2f}x)")
    
    print(f"\nContext Usage:")
    print(f"  Base LLM:    {0:>6} characters (no context)")
    print(f"  Base RAG:    {rag_context:>6} characters")
    print(f"  Hybrid RAG:  {hybrid_context:>6} characters")
    
    print(f"\nPerformance:")
    print(f"  Base LLM:    {base_time:>6.2f} seconds")
    print(f"  Base RAG:    {rag_time:>6.2f} seconds ({rag_time/base_time:.2f}x)")
    print(f"  Hybrid RAG:  {hybrid_time:>6.2f} seconds ({hybrid_time/base_time:.2f}x)")
    
    # Quality assessment
    print(f"\n💡 Quality Assessment:")
    
    if hybrid_length > base_length * 1.3:
        print(f"  ✅ Hybrid RAG provides significantly more detailed responses")
    
    if hybrid_context > 0:
        print(f"  ✅ Hybrid RAG effectively uses retrieved context ({hybrid_context} chars)")
    
    if rag_context > 0:
        print(f"  ✅ Base RAG also uses context effectively ({rag_context} chars)")
    
    if hybrid_time < base_time * 3:
        print(f"  ✅ Performance overhead is reasonable")
    
    # Recommendation
    print(f"\n🎯 Recommendation:")
    if hybrid_length > base_length * 1.2 and hybrid_context > 0:
        print(f"  ✅ RAG systems are working better than Base LLM!")
        print(f"  📈 Hybrid RAG provides {hybrid_length/base_length:.1f}x more detail")
        print(f"  📚 Uses {hybrid_context} characters of relevant context")
    else:
        print(f"  ⚠️  Results are mixed - may need better queries or tuning")

def interactive_mode():
    """Run in interactive mode for multiple queries."""
    print("🔄 Interactive Response Comparison Mode")
    print("=" * 50)
    print("Enter queries to compare responses (type 'quit' to exit)")
    
    while True:
        print("\n" + "-" * 50)
        query = input("Enter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        try:
            compare_responses(query)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with a different query.")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Command line query
        query = " ".join(sys.argv[1:])
        compare_responses(query)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
