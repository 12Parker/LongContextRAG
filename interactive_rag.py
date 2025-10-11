#!/usr/bin/env python3
"""
Interactive RAG system for testing queries and exploring the Long Context RAG capabilities.
"""

import os
import sys
from index import LongContextRAG
from langchain.schema import Document

def load_sample_data():
    """Load sample data for the RAG system."""
    rag = LongContextRAG()
    
    # Try to load from sample file
    sample_file = "data/sample_documents.txt"
    if os.path.exists(sample_file):
        print(f"📚 Loading sample documents from {sample_file}")
        documents = rag.load_documents([sample_file])
    else:
        print("📚 Using built-in sample documents")
        documents = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
                metadata={"source": "sample1.txt", "topic": "machine_learning"}
            ),
            Document(
                page_content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language. Transformer models, introduced in 2017, revolutionized NLP with self-attention mechanisms and have led to models like GPT, BERT, and T5.",
                metadata={"source": "sample2.txt", "topic": "nlp"}
            ),
            Document(
                page_content="Retrieval Augmented Generation (RAG) combines large language models with external knowledge retrieval. It involves document processing, vector storage, retrieval, and generation to provide accurate, up-to-date information while reducing hallucination.",
                metadata={"source": "sample3.txt", "topic": "rag"}
            )
        ]
    
    return rag, documents

def print_help():
    """Print help information."""
    print("\n" + "="*60)
    print("🤖 Long Context RAG Interactive System")
    print("="*60)
    print("Commands:")
    print("  /help     - Show this help message")
    print("  /examples - Show example queries")
    print("  /stats    - Show system statistics")
    print("  /quit     - Exit the system")
    print("  /clear    - Clear the screen")
    print("\nJust type your question to get started!")
    print("="*60)

def print_examples():
    """Print example queries."""
    examples = [
        "What is machine learning and what are its main types?",
        "How do transformer models work in NLP?",
        "What are the benefits of RAG systems?",
        "What challenges do long context models face?",
        "Compare supervised and unsupervised learning",
        "Explain the attention mechanism in transformers",
        "What is the difference between BERT and GPT models?",
        "How does RAG reduce hallucination in language models?"
    ]
    
    print("\n📝 Example Queries:")
    print("-" * 40)
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    print("-" * 40)

def print_stats(rag):
    """Print system statistics."""
    if rag.vectorstore:
        try:
            # Get collection info
            collection = rag.vectorstore._collection
            count = collection.count()
            print(f"\n📊 System Statistics:")
            print(f"   - Documents in vector store: {count}")
            print(f"   - Embedding model: {rag.embeddings.model}")
            print(f"   - LLM model: {rag.llm.model_name}")
            print(f"   - Chunk size: {rag.text_splitter._chunk_size}")
            print(f"   - Top K results: {rag.retriever.search_kwargs.get('k', 'N/A')}")
        except Exception as e:
            print(f"📊 System Statistics: Available (details unavailable: {e})")
    else:
        print("📊 System Statistics: Vector store not initialized")

def main():
    """Main interactive loop."""
    print("🚀 Initializing Long Context RAG System...")
    
    try:
        rag, documents = load_sample_data()
        rag.create_vectorstore(documents)
        print("✅ System ready!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")
        return
    
    print_help()
    
    while True:
        try:
            # Get user input
            query = input("\n💬 Your question: ").strip()
            
            # Handle commands
            if query.lower() in ['/quit', '/exit', '/q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == '/help':
                print_help()
                continue
            elif query.lower() == '/examples':
                print_examples()
                continue
            elif query.lower() == '/stats':
                print_stats(rag)
                continue
            elif query.lower() == '/clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif not query:
                continue
            
            # Process query
            print(f"\n🔍 Processing: {query}")
            print("-" * 60)
            
            result = rag.generate_response(query, use_rag=True)
            
            print(f"🤖 Response:\n{result['response']}")
            print(f"\n📈 Stats: Method={result['method']}, Retrieved docs={result['retrieved_docs']}, Context length={result['context_length']} chars")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try rephrasing your question or type /help for assistance")

if __name__ == "__main__":
    main()
