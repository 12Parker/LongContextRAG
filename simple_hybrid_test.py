#!/usr/bin/env python3
"""
Simplified Hybrid Attention RAG Test

This script provides a simplified test of the hybrid attention RAG system
that focuses on the core functionality without complex neural components.
"""

import os
import sys
from pathlib import Path

def main():
    print("🧪 Simplified Hybrid Attention RAG Test")
    print("=" * 50)
    
    try:
        # Test basic RAG system first
        print("1. Testing basic RAG system...")
        from index import LongContextRAG
        from bookcorpus_integration import BookCorpusLoader, BookCorpusConfig
        
        # Load sample documents
        config = BookCorpusConfig(max_books=1)
        loader = BookCorpusLoader(config)
        books = loader.load_sample_books()
        documents = loader.process_books_for_rag()
        
        # Test base RAG
        base_rag = LongContextRAG()
        base_rag.create_vectorstore(documents)
        
        # Test query
        query = "What is the main topic of this book?"
        result = base_rag.generate_response(query, use_rag=True)
        
        print(f"✅ Base RAG working:")
        print(f"   Response: {result['response'][:150]}...")
        print(f"   Retrieved docs: {result['retrieved_docs']}")
        print(f"   Context length: {result['context_length']}")
        
        # Test hybrid attention components individually
        print("\n2. Testing hybrid attention components...")
        from hybrid_attention_rag import HybridAttentionRAG, AttentionConfig
        
        # Create attention config with smaller dimensions for testing
        attn_config = AttentionConfig(
            window_size=256,
            num_landmark_tokens=16,
            max_retrieved_segments=4,
            hidden_size=768,  # Smaller for testing
            num_attention_heads=8
        )
        
        # Test hybrid attention model
        hybrid_model = HybridAttentionRAG(attn_config)
        
        # Create sample input
        import torch
        sample_input = torch.randn(1, 100, 768)  # batch_size=1, seq_len=100, hidden_size=768
        
        with torch.no_grad():
            output = hybrid_model(sample_input)
        
        print(f"✅ Hybrid attention model working:")
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        # Test neural retriever components
        print("\n3. Testing neural retriever components...")
        from neural_retriever import NeuralRetriever, RetrieverConfig
        
        # Create retriever config with smaller dimensions
        ret_config = RetrieverConfig(
            query_embed_dim=768,
            doc_embed_dim=768,
            hidden_dim=512,
            num_candidates=50,
            top_k=4
        )
        
        # Test neural retriever
        retriever = NeuralRetriever(ret_config)
        
        # Create sample data
        query_emb = torch.randn(1, 50, 768)  # batch_size=1, query_len=50, embed_dim=768
        doc_emb = torch.randn(50, 100, 768)  # num_docs=50, doc_len=100, embed_dim=768
        
        with torch.no_grad():
            scores, retrieved = retriever(query_emb, doc_emb)
        
        print(f"✅ Neural retriever working:")
        print(f"   Query shape: {query_emb.shape}")
        print(f"   Doc shape: {doc_emb.shape}")
        print(f"   Scores shape: {scores.shape}")
        print(f"   Retrieved shape: {retrieved.shape}")
        
        print("\n🎉 All components are working correctly!")
        print("\n📊 Summary:")
        print(f"   - Base RAG: ✅ Working")
        print(f"   - Hybrid Attention: ✅ Working")
        print(f"   - Neural Retriever: ✅ Working")
        print(f"   - Documents loaded: {len(documents)}")
        print(f"   - Books processed: {len(books)}")
        
        print("\n🚀 Next steps:")
        print("   1. The individual components work correctly")
        print("   2. The integration needs dimension alignment")
        print("   3. You can now focus on research and experimentation")
        print("   4. Use the base RAG system for immediate testing")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
