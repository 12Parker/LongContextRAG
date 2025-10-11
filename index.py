"""
Long Context RAG Research Project
Main implementation for Retrieval Augmented Generation with long context support.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LongContextRAG:
    """
    Long Context Retrieval Augmented Generation system.
    
    This class implements a RAG system optimized for handling long context documents
    with efficient retrieval and generation capabilities.
    """
    
    def __init__(self):
        """Initialize the RAG system with configuration."""
        config.validate_config()
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for the project."""
        Path(config.DATA_DIR).mkdir(exist_ok=True)
        Path(config.VECTOR_STORE_DIR).mkdir(exist_ok=True)
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "filename": os.path.basename(file_path)
                    }
                )
                documents.append(doc)
                logger.info(f"Loaded document: {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def create_vectorstore(self, documents: List[Document], persist: bool = True):
        """
        Create and populate vector store with document embeddings.
        
        Args:
            documents: List of documents to embed
            persist: Whether to persist the vector store to disk
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create vector store
        persist_directory = config.VECTOR_STORE_DIR if persist else None
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K_RESULTS}
        )
        
        logger.info("Vector store created successfully")
    
    def load_existing_vectorstore(self):
        """Load existing vector store from disk."""
        if not os.path.exists(config.VECTOR_STORE_DIR):
            raise FileNotFoundError("No existing vector store found")
        
        self.vectorstore = Chroma(
            persist_directory=config.VECTOR_STORE_DIR,
            embedding_function=self.embeddings
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K_RESULTS}
        )
        
        logger.info("Loaded existing vector store")
    
    def retrieve_relevant_docs(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The search query
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} relevant documents")
        return docs
    
    def generate_response(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Generate a response using RAG or direct LLM.
        
        Args:
            query: The user query
            use_rag: Whether to use RAG or direct LLM response
            
        Returns:
            Dictionary containing response and metadata
        """
        if use_rag and not self.retriever:
            logger.warning("RAG requested but no retriever available. Using direct LLM.")
            use_rag = False
        
        if use_rag:
            # Use RAG with retrieved context
            retrieved_docs = self.retrieve_relevant_docs(query)
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create prompt with context
            prompt = ChatPromptTemplate.from_template("""
            Context:
            {context}
            
            Question: {question}
            
            Please provide a comprehensive answer based on the provided context. 
            If the context doesn't contain enough information to answer the question, 
            please say so and provide what information you can.
            """)
            
            # Create chain
            chain = prompt | self.llm
            
            # Generate response
            response = chain.invoke({
                "context": context,
                "question": query
            })
            
            return {
                "response": response.content,
                "method": "rag",
                "retrieved_docs": len(retrieved_docs),
                "context_length": len(context)
            }
        
        else:
            # Direct LLM response
            response = self.llm.invoke(query)
            
            return {
                "response": response.content,
                "method": "direct",
                "retrieved_docs": 0,
                "context_length": 0
            }

def main():
    """Example usage of the LongContextRAG system."""
    # Initialize RAG system
    rag = LongContextRAG()
    
    # Try to load sample documents from file
    sample_file = "data/sample_documents.txt"
    if os.path.exists(sample_file):
        print(f"Loading sample documents from {sample_file}")
        documents = rag.load_documents([sample_file])
    else:
        print("Sample documents file not found. Using basic examples.")
        # Fallback to basic examples
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
    
    # Create vector store
    rag.create_vectorstore(documents)
    
    # Example queries
    queries = [
        "What is machine learning and what are its main types?",
        "How do transformer models work in NLP?",
        "What are the benefits of RAG systems?",
        "What challenges do long context models face?"
    ]
    
    # Test RAG responses
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = rag.generate_response(query, use_rag=True)
        print(f"Response: {result['response']}")
        print(f"\nMethod: {result['method']}, Retrieved docs: {result['retrieved_docs']}, Context length: {result['context_length']} chars")

if __name__ == "__main__":
    main()
