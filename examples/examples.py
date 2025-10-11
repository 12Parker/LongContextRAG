"""
Example usage and test cases for the Long Context RAG system.
"""

import os
from typing import List
from langchain.schema import Document

from core.index import LongContextRAG
from core.prompts import RAGPrompts, PromptFactory

def create_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    sample_docs = [
        Document(
            page_content="""
            Machine Learning Fundamentals
            
            Machine learning is a subset of artificial intelligence (AI) that focuses on the development 
            of algorithms and statistical models that enable computer systems to improve their performance 
            on a specific task through experience, without being explicitly programmed.
            
            Key Types of Machine Learning:
            1. Supervised Learning: Learning with labeled training data
            2. Unsupervised Learning: Finding patterns in data without labels
            3. Reinforcement Learning: Learning through interaction with an environment
            
            Popular algorithms include linear regression, decision trees, neural networks, and support 
            vector machines. The field has seen tremendous growth with the advent of deep learning, 
            which uses multi-layered neural networks to solve complex problems.
            """,
            metadata={"source": "ml_fundamentals.txt", "topic": "machine_learning"}
        ),
        
        Document(
            page_content="""
            Natural Language Processing and Transformers
            
            Natural Language Processing (NLP) is a field of AI that focuses on the interaction between 
            computers and humans through natural language. The ultimate objective of NLP is to read, 
            decipher, understand, and make sense of human language in a valuable way.
            
            Transformer Architecture:
            The transformer architecture, introduced in "Attention Is All You Need" (2017), revolutionized 
            NLP. Key components include:
            - Self-attention mechanisms
            - Multi-head attention
            - Positional encoding
            - Feed-forward networks
            
            Modern applications include language models like GPT, BERT, and T5, which have achieved 
            state-of-the-art performance on various NLP tasks including text generation, translation, 
            and question answering.
            """,
            metadata={"source": "nlp_transformers.txt", "topic": "nlp"}
        ),
        
        Document(
            page_content="""
            Retrieval Augmented Generation (RAG)
            
            RAG is a technique that combines the power of large language models with external knowledge 
            retrieval. It addresses the limitations of LLMs by allowing them to access up-to-date 
            information and domain-specific knowledge.
            
            RAG Process:
            1. Document Processing: Split documents into chunks and create embeddings
            2. Vector Storage: Store embeddings in a vector database
            3. Retrieval: Find relevant documents for a given query
            4. Generation: Use retrieved context to generate responses
            
            Benefits of RAG:
            - Access to current information
            - Reduced hallucination
            - Domain-specific knowledge integration
            - Improved accuracy and relevance
            
            Challenges include maintaining vector store consistency, handling long contexts, and 
            optimizing retrieval quality.
            """,
            metadata={"source": "rag_overview.txt", "topic": "rag"}
        ),
        
        Document(
            page_content="""
            Long Context Language Models
            
            Long context language models are designed to handle and process very long sequences of text, 
            often exceeding 100,000 tokens. This capability is crucial for applications requiring 
            analysis of large documents, codebases, or conversations.
            
            Technical Challenges:
            - Computational complexity grows quadratically with sequence length
            - Memory requirements increase significantly
            - Attention mechanisms need optimization
            - Quality degradation at very long sequences
            
            Solutions and Approaches:
            - Sparse attention patterns
            - Hierarchical processing
            - Sliding window techniques
            - Memory-efficient attention mechanisms
            
            Applications include document analysis, code understanding, long-form content generation, 
            and research assistance where comprehensive context is essential.
            """,
            metadata={"source": "long_context.txt", "topic": "long_context"}
        )
    ]
    
    return sample_docs

def example_basic_rag():
    """Example of basic RAG usage."""
    print("=== Basic RAG Example ===")
    
    # Initialize RAG system
    rag = LongContextRAG()
    
    # Load sample documents
    documents = create_sample_documents()
    
    # Create vector store
    rag.create_vectorstore(documents)
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How do transformers work in NLP?",
        "What are the benefits of RAG?",
        "What challenges do long context models face?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = rag.generate_response(query, use_rag=True)
        print(f"Response: {result['response'][:200]}...")
        print(f"Method: {result['method']}, Retrieved docs: {result['retrieved_docs']}")

def example_research_rag():
    """Example of research-focused RAG usage."""
    print("\n=== Research RAG Example ===")
    
    rag = LongContextRAG()
    documents = create_sample_documents()
    rag.create_vectorstore(documents)
    
    # Research question
    research_question = "Compare the approaches used in traditional machine learning versus modern transformer-based NLP models"
    
    # Retrieve relevant documents
    relevant_docs = rag.retrieve_relevant_docs(research_question)
    
    # Create context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Use research prompt
    from langchain_openai import ChatOpenAI
    from core.config import config
    
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.1
    )
    
    research_prompt = RAGPrompts.RESEARCH_RAG
    chain = research_prompt | llm
    
    response = chain.invoke({
        "context": context,
        "question": research_question
    })
    
    print(f"Research Question: {research_question}")
    print(f"Response: {response.content}")

def example_long_context_analysis():
    """Example of long context analysis."""
    print("\n=== Long Context Analysis Example ===")
    
    rag = LongContextRAG()
    documents = create_sample_documents()
    rag.create_vectorstore(documents)
    
    # Long context question
    question = "Analyze the evolution of AI techniques from traditional machine learning to modern transformer models, focusing on their capabilities and limitations"
    
    # Get all documents for comprehensive analysis
    all_docs = rag.vectorstore.similarity_search(question, k=10)
    context = "\n\n".join([doc.page_content for doc in all_docs])
    
    # Use long context analysis prompt
    from langchain_openai import ChatOpenAI
    from core.config import config
    
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.1
    )
    
    analysis_prompt = RAGPrompts.LONG_CONTEXT_ANALYSIS
    chain = analysis_prompt | llm
    
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    print(f"Analysis Question: {question}")
    print(f"Context Length: {len(context)} characters")
    print(f"Response: {response.content}")

def example_custom_prompt():
    """Example of using custom prompts."""
    print("\n=== Custom Prompt Example ===")
    
    rag = LongContextRAG()
    documents = create_sample_documents()
    rag.create_vectorstore(documents)
    
    # Create custom research prompt
    custom_prompt = PromptFactory.create_research_prompt("artificial intelligence")
    
    # Test query
    query = "What are the key technical innovations that enabled the development of modern AI systems?"
    
    # Retrieve and generate
    relevant_docs = rag.retrieve_relevant_docs(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    from langchain_openai import ChatOpenAI
    from core.config import config
    
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.1
    )
    
    chain = custom_prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })
    
    print(f"Custom Prompt Query: {query}")
    print(f"Response: {response.content}")

def example_evaluation():
    """Example of evaluating RAG responses."""
    print("\n=== RAG Evaluation Example ===")
    
    rag = LongContextRAG()
    documents = create_sample_documents()
    rag.create_vectorstore(documents)
    
    # Generate a response
    query = "What is the difference between supervised and unsupervised learning?"
    result = rag.generate_response(query, use_rag=True)
    
    # Get the context used
    relevant_docs = rag.retrieve_relevant_docs(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create evaluation prompt
    eval_prompt = PromptFactory.create_evaluation_prompt()
    
    from langchain_openai import ChatOpenAI
    from core.config import config
    
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.1
    )
    
    chain = eval_prompt | llm
    evaluation = chain.invoke({
        "question": query,
        "context": context,
        "response": result['response']
    })
    
    print(f"Original Query: {query}")
    print(f"Generated Response: {result['response']}")
    print(f"Evaluation: {evaluation.content}")

def run_all_examples():
    """Run all example functions."""
    try:
        example_basic_rag()
        example_research_rag()
        example_long_context_analysis()
        example_custom_prompt()
        example_evaluation()
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")

if __name__ == "__main__":
    run_all_examples()
