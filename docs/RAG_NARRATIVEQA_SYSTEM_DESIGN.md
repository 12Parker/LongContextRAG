# RAG with NarrativeQA: System Design

## Overview

The RAG (Retrieval-Augmented Generation) system for NarrativeQA implements a **retrieval-based approach** that chunks stories into segments, creates vector embeddings, and retrieves relevant chunks for question answering. This approach balances efficiency with context quality.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NARRATIVEQA RAG SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │ NarrativeQA │    │   Text       │    │   Vector        │    │
│  │   Story     │───▶│   Chunker    │───▶│   Embeddings    │    │
│  │             │    │              │    │                 │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Question  │    │   Similarity │    │   Retrieved     │    │
│  │   Embedding │───▶│   Search     │───▶│   Chunks        │    │
│  │             │    │              │    │                 │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Context   │    │   LLM        │    │   Generated     │    │
│  │   Builder   │───▶│   Engine     │───▶│   Answer        │    │
│  │             │    │              │    │                 │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Story Processing Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                    STORY PROCESSING LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │ NarrativeQA │    │   Text      │    │   Chunked       │    │
│  │   Story     │───▶│   Splitter  │───▶│   Segments      │    │
│  │   (54K)     │    │             │    │   (72 chunks)   │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Text Chunking Process                      │    │
│  │                                                         │    │
│  │  • Chunk Size: 1,000 characters                        │    │
│  │  • Chunk Overlap: 100 characters                       │    │
│  │  • Separators: ["\n\n", "\n", " ", ""]                │    │
│  │  • Result: 72 chunks from 54K story                  │    │
│  │                                                         │    │
│  │  Example Chunk:                                        │    │
│  │  "Mark Hunter is a high school student in Phoenix.     │    │
│  │   He runs a pirate radio station from his parents'     │    │
│  │   basement. The station broadcasts music and..."      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Vector Embedding Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR EMBEDDING LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   Text      │    │   Hugging    │    │   Vector        │    │
│  │   Chunks    │───▶│   Face       │───▶│   Embeddings    │    │
│  │             │    │   Model      │    │   (384-dim)     │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Embedding Configuration                    │    │
│  │                                                         │    │
│  │  • Model: all-MiniLM-L6-v2                             │    │
│  │  • Dimensions: 384                                     │    │
│  │  • Device: CPU                                         │    │
│  │  │                                                      │    │
│  │  • Process:                                            │    │
│  │    1. Convert text chunks to embeddings               │    │
│  │    2. Store in ChromaDB vector database                │    │
│  │    3. Create metadata for each chunk                   │    │
│  │    4. Enable similarity search                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Vector Database Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   Vector    │    │   ChromaDB   │    │   Persistent    │    │
│  │   Store     │───▶│   Database   │───▶│   Storage       │    │
│  │             │    │             │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Database Structure                          │    │
│  │                                                         │    │
│  │  Collection: narrativeqa_vectordb                       │    │
│  │  Documents: 72 chunks                                  │    │
│  │  Embeddings: 384-dimensional vectors                   │    │
│  │  Metadata: {                                           │    │
│  │    "story_id": "current_story",                        │    │
│  │    "chunk_id": 0,                                      │    │
│  │    "total_chunks": 72,                                │    │
│  │    "story_index": 0                                   │    │
│  │  }                                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Retrieval Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   Question  │    │   Similarity│    │   Top-K        │    │
│  │   Embedding │───▶│   Search    │───▶│   Results      │    │
│  │             │    │             │    │   (5 chunks)   │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Retrieval Process                          │    │
│  │                                                         │    │
│  │  1. Convert question to embedding                      │    │
│  │  2. Compute cosine similarity with all chunks          │    │
│  │  3. Rank chunks by similarity score                   │    │
│  │  4. Return top-K most relevant chunks                 │    │
│  │                                                         │    │
│  │  Example Results:                                      │    │
│  │  • Chunk 1: "Mark Hunter is a high school student..." │    │
│  │  • Chunk 2: "The radio station operates from..."      │    │
│  │  • Chunk 3: "Students tune in to hear music..."       │    │
│  │  • Chunk 4: "The basement location provides..."       │    │
│  │  • Chunk 5: "Mark's parents are unaware of..."        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Context Assembly Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT ASSEMBLY LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   Retrieved │    │   Context   │    │   Formatted     │    │
│  │   Chunks    │───▶│   Builder   │───▶│   Context       │    │
│  │             │    │             │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Context Assembly Process                   │    │
│  │                                                         │    │
│  │  1. Combine retrieved chunks with headers              │    │
│  │  2. Add chunk metadata (if available)                 │    │
│  │  3. Truncate if context exceeds token limit           │    │
│  │  4. Format for LLM consumption                       │    │
│  │                                                         │    │
│  │  Example Context:                                      │    │
│  │  "Chunk 1: Mark Hunter is a high school student...    │    │
│  │   Chunk 2: The radio station operates from...         │    │
│  │   Chunk 3: Students tune in to hear music...           │    │
│  │   Chunk 4: The basement location provides...           │    │
│  │   Chunk 5: Mark's parents are unaware of..."          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6. LLM Generation Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM GENERATION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   Context   │    │   Prompt    │    │   Generated     │    │
│  │   +         │───▶│   Template  │───▶│   Answer        │    │
│  │   Question  │    │             │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LLM Processing                             │    │
│  │                                                         │    │
│  │  Prompt Template:                                      │    │
│  │  "Based on the following story excerpts, answer the    │    │
│  │   question.                                            │    │
│  │                                                         │    │
│  │  Context: {context}                                   │    │
│  │                                                         │    │
│  │  Question: {question}                                  │    │
│  │                                                         │    │
│  │  Please provide a comprehensive answer based on the     │    │
│  │  story content..."                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG DATA FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NarrativeQA Story (54K chars)                                 │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Text        │                                               │
│  │ Chunking    │ → 72 chunks (1K chars each)                   │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Vector      │                                               │
│  │ Embeddings  │ → 384-dim vectors                             │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ ChromaDB    │                                               │
│  │ Storage     │ → Persistent vector database                 │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Question    │                                               │
│  │ Embedding   │ → Query vector                                │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Similarity  │                                               │
│  │ Search      │ → Top-5 relevant chunks                      │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Context     │                                               │
│  │ Assembly    │ → Formatted context                           │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ LLM         │                                               │
│  │ Generation  │ → Final answer                               │
│  └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. **Efficient Retrieval**
- Chunk-based processing for scalability
- Vector similarity search for relevance
- Top-K retrieval for focused context

### 2. **Context Optimization**
- Only relevant chunks used for generation
- Reduced token usage compared to full story
- Maintained narrative coherence through overlap

### 3. **Persistent Storage**
- ChromaDB for vector storage
- Fresh database creation for each story
- Metadata tracking for chunks

### 4. **Modular Architecture**
- Separate components for each processing step
- Easy to modify individual components
- Clear interfaces between layers

## Performance Characteristics

### **Strengths**
- ✅ **Efficient Processing**: Only relevant chunks processed
- ✅ **Scalable**: Can handle large documents
- ✅ **Focused Context**: Retrieval finds relevant information
- ✅ **Lower Token Usage**: ~610 tokens vs 1,524 for base LLM

### **Limitations**
- ❌ **Information Loss**: Chunking may break narrative flow
- ❌ **Retrieval Quality**: Depends on embedding similarity
- ❌ **Context Fragmentation**: May miss cross-chunk connections
- ❌ **Setup Overhead**: Requires vector database creation

## Comparison with Base LLM

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM COMPARISON                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base LLM Approach:                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │ Question +  │    │   Direct    │    │   Answer        │    │
│  │ Full Story  │───▶│   LLM      │───▶│   (Complete)    │    │
│  │ (1,524 tok) │    │   Process  │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│                                                                 │
│  RAG Approach:                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │ Question    │    │  Retrieval  │    │   Answer        │    │
│  │             │───▶│  + LLM      │───▶│   (Focused)     │    │
│  │             │    │  Process    │    │                 │    │
│  │ (610 tok)   │    │             │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### **Core Class Structure**
```python
class NarrativeQARAGBaseline:
    def __init__(self, chunk_size=1000, top_k_results=5):
        self.chunk_size = chunk_size
        self.top_k_results = top_k_results
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(...)
        self.vectorstore = None
    
    def generate_response(self, question):
        # Retrieve relevant chunks
        docs = self.vectorstore.similarity_search(question, k=self.top_k_results)
        
        # Build context
        context = self._build_context(docs)
        
        # Generate response
        response = self.llm.invoke(self._create_prompt(question, context))
        return self._format_response(response, docs)
```

### **Text Chunking Algorithm**
```python
def _create_vectordb(self, story_text):
    # Split story into chunks
    chunks = self.text_splitter.split_text(story_text)
    
    # Create documents and metadata
    documents = []
    metadatas = []
    
    for j, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            'story_id': 'current_story',
            'chunk_id': j,
            'total_chunks': len(chunks),
            'story_index': 0
        })
    
    # Create vector store
    self.vectorstore = Chroma.from_texts(
        texts=documents,
        metadatas=metadatas,
        embedding=self.embeddings,
        persist_directory=self.db_path
    )
```

### **Retrieval Process**
```python
def _retrieve_chunks(self, question):
    # Get relevant chunks using similarity search
    docs = self.vectorstore.similarity_search(
        question, 
        k=self.top_k_results
    )
    
    # Build context from retrieved chunks
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"Chunk {i+1}: {doc.page_content}")
    
    return "\n\n".join(context_parts)
```

## Performance Metrics

Based on NarrativeQA testing results:

| Metric | Value |
|--------|-------|
| **Success Rate** | 100% |
| **Average Response Time** | 7.13s |
| **Average Answer Length** | 941 characters |
| **Average Context Tokens** | 790 |
| **Average Relevance Score** | 0.395 |
| **Retrieved Documents** | 5.0 |

## Use Cases

### ✅ **Best For**
1. **Large Documents**: When stories exceed context windows
2. **Focused Questions**: Specific information retrieval
3. **Scalable Systems**: Multiple documents or stories
4. **Efficient Processing**: When token usage is a concern

### ❌ **Not Ideal For**
1. **Narrative Coherence**: Questions requiring full story context
2. **Cross-Reference Questions**: Information spanning multiple chunks
3. **Simple Stories**: When full context fits in window
4. **Complex Relationships**: Questions about character development

## Conclusion

The RAG system for NarrativeQA represents a **sophisticated retrieval-based approach** that balances efficiency with context quality. While it may not always match the performance of direct LLM approaches for single-story scenarios, it provides a scalable foundation for handling larger documents and more complex retrieval scenarios.

The system's strength lies in its **modular architecture** and **efficient processing**, making it particularly suitable for scenarios where document size or processing efficiency are primary concerns.
