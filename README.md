# Long Context RAG Research Project

A comprehensive research framework for exploring Retrieval Augmented Generation (RAG) with long context language models. This project provides tools and examples for studying how to effectively combine large language models with external knowledge retrieval for research applications.

## Features

- **OpenAI API Integration**: Full support for GPT-4 and other OpenAI models
- **Vector Store Management**: ChromaDB integration for efficient document storage and retrieval
- **Long Context Support**: Optimized for handling large documents and extended contexts
- **Hybrid Attention RAG**: Advanced RAG system with neural retrieval and attention mechanisms
- **Multiple Prompt Templates**: Pre-built prompts for various research scenarios
- **Flexible Configuration**: Easy-to-customize settings for different use cases
- **Comprehensive Examples**: Ready-to-run examples demonstrating different RAG techniques
- **Evaluation Framework**: Tools to compare RAG vs Base LLM performance
- **Long Context Testing**: Support for context lengths up to 128k+ tokens

## Quick Start

1. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your environment**:
   ```bash
   python setup_env.py
   ```
   Then edit the `.env` file and add your API keys:
   - `OPENAI_API_KEY=your_actual_api_key_here`
   - Add any other API keys you need

4. **Test the setup**:
   ```bash
   python test_setup.py
   ```

5. **Run basic example**:
   ```bash
   python index.py
   ```

6. **Explore advanced examples**:
   ```bash
   python examples.py
   ```

7. **Try the interactive system**:
   ```bash
   python examples/interactive_rag.py
   ```

## Project Structure

```
LongContextRAG/
├── core/                    # Core RAG implementation
│   ├── config.py           # Configuration management
│   ├── index.py            # Main RAG system
│   ├── long_context_config.py  # Long context configuration
│   └── prompts.py          # Prompt templates
├── hybrid/                 # Hybrid attention RAG
│   ├── working_hybrid_rag.py    # Main hybrid RAG system
│   ├── hybrid_attention_rag.py  # Attention mechanisms
│   └── hybrid_rag_integration.py # Integration components
├── evaluation/             # Evaluation framework
│   ├── rag_evaluation.py   # Comprehensive evaluation
│   └── quick_rag_evaluation.py  # Quick evaluation script
├── examples/               # Example scripts and demos
│   ├── compare_responses.py     # Side-by-side comparison
│   ├── long_context_testing.py  # Long context testing
│   ├── run_long_context_test.py # Quick long context tests
│   └── interactive_rag.py       # Interactive RAG demo
├── docs/                   # Documentation
│   ├── EVALUATION_GUIDE.md      # Evaluation guide
│   └── LONG_CONTEXT_TESTING_README.md  # Long context guide
├── VectorDB/               # Vector database builder
│   └── build_db.py         # ChromaDB builder for BookCorpus
├── testing/                # Testing utilities
├── training/               # Training components
└── utils/                  # Utility functions
```

### Alternative: Use the activation script
```bash
chmod +x activate.sh
./activate.sh
```

## Evaluation and Testing

### Quick RAG vs Base LLM Evaluation
```bash
# Quick evaluation with 5 test queries
python evaluation/quick_rag_evaluation.py

# Side-by-side response comparison
python examples/compare_responses.py "Your query here"
```

### Long Context Testing
```bash
# Test different context sizes
python examples/run_long_context_test.py large
python examples/run_long_context_test.py ultra

# Comprehensive long context testing
python examples/long_context_testing.py
```

### Comprehensive Evaluation
```bash
# Full evaluation framework
python evaluation/rag_evaluation.py
```

## Documentation

- **[Evaluation Guide](docs/EVALUATION_GUIDE.md)**: Complete guide to evaluating RAG performance
- **[Long Context Testing Guide](docs/LONG_CONTEXT_TESTING_README.md)**: Guide for testing with 32k+ token contexts
- **[Hybrid Attention README](HYBRID_ATTENTION_README.md)**: Technical details about hybrid attention mechanisms

## Project Structure

```
LongContextRAG/
├── core/                    # Core RAG implementation
│   ├── config.py           # Configuration management
│   ├── index.py            # Main RAG system
│   ├── long_context_config.py  # Long context configuration
│   └── prompts.py          # Prompt templates
├── hybrid/                 # Hybrid attention RAG
│   ├── working_hybrid_rag.py    # Main hybrid RAG system
│   ├── hybrid_attention_rag.py  # Attention mechanisms
│   └── hybrid_rag_integration.py # Integration components
├── evaluation/             # Evaluation framework
│   ├── rag_evaluation.py   # Comprehensive evaluation
│   └── quick_rag_evaluation.py  # Quick evaluation script
├── examples/               # Example scripts and demos
│   ├── compare_responses.py     # Side-by-side comparison
│   ├── long_context_testing.py  # Long context testing
│   ├── run_long_context_test.py # Quick long context tests
│   └── interactive_rag.py       # Interactive RAG demo
├── docs/                   # Documentation
│   ├── EVALUATION_GUIDE.md      # Evaluation guide
│   └── LONG_CONTEXT_TESTING_README.md  # Long context guide
├── VectorDB/               # Vector database builder
│   └── build_db.py         # ChromaDB builder for BookCorpus
├── testing/                # Testing utilities
├── training/               # Training components
├── utils/                  # Utility functions
├── data/                   # Directory for documents
├── vector_store/           # Persistent vector stores
├── results/                # Research results and outputs
├── requirements.txt        # Python dependencies
├── env.template           # Environment variables template
└── README.md              # This file
```

## Usage Examples

### Basic RAG Usage

```python
from index import LongContextRAG
from langchain.schema import Document

# Initialize RAG system
rag = LongContextRAG()

# Load your documents
documents = [
    Document(page_content="Your document content here", metadata={"source": "doc1.txt"})
]

# Create vector store
rag.create_vectorstore(documents)

# Query the system
result = rag.generate_response("Your question here", use_rag=True)
print(result['response'])
```

### Research-Focused Queries

```python
from prompts import RAGPrompts

# Use research-specific prompt
research_prompt = RAGPrompts.RESEARCH_RAG
# ... (see examples.py for full implementation)
```

## Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4-turbo-preview)
- `CHUNK_SIZE`: Document chunk size for processing (default: 1000)
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 5)
- `CONTEXT_WINDOW_SIZE`: Maximum context length (default: 32000)

## Research Applications

This framework is designed for research in:

- **Long Context Processing**: How to effectively handle and process very long documents
- **Retrieval Quality**: Optimizing document retrieval for better RAG performance
- **Prompt Engineering**: Testing different prompt strategies for research tasks
- **Context Management**: Balancing context length with response quality
- **Domain Adaptation**: Adapting RAG systems for specific research domains

## 🔬 Advanced: Hybrid Attention RAG

The project includes an advanced implementation of a **novel hybrid attention mechanism** combining:

- **Sliding Window Attention** for local relationships
- **Sparse Global Attention** with strategic landmark tokens
- **Retrieval-Augmented Segments** with dynamic passage integration
- **Neural Retrievers** trained end-to-end with the language model
- **Dynamic Query Generation** for task-specific adaptation

### Quick Start with Hybrid Attention

```bash
# Test the hybrid attention system
python -c "from hybrid_attention_rag import test_hybrid_attention_rag; test_hybrid_attention_rag()"

# Test neural retriever components
python -c "from neural_retriever import test_neural_retriever; test_neural_retriever()"

# Run comprehensive research experiments
python research_notebook.py

# Test with BookCorpus-like data for long context research
python start_bookcorpus_testing.py
```

### Key Files for Hybrid Attention

- `hybrid_attention_rag.py` - Core hybrid attention implementation
- `neural_retriever.py` - Neural retriever components
- `hybrid_rag_integration.py` - Integration with existing RAG system
- `train_hybrid_rag.py` - End-to-end training framework
- `research_notebook.py` - Research experimentation tools
- `bookcorpus_integration.py` - BookCorpus testing for long context research
- `start_bookcorpus_testing.py` - Quick start script for BookCorpus testing
- `HYBRID_ATTENTION_README.md` - Detailed documentation
- `BOOKCORPUS_TESTING_GUIDE.md` - BookCorpus testing guide

See `HYBRID_ATTENTION_README.md` for comprehensive documentation of the hybrid attention methodology and implementation.

## Security

🔒 **API Key Protection**: Your API keys are kept secure through:

- `.env` file is in `.gitignore` (never committed to git)
- Only `env.template` is tracked (contains no real keys)
- Use `python setup_env.py` to create your local `.env` file
- Your actual API keys stay local to your machine

## Contributing

This is a research project. Feel free to:

- Add new prompt templates in `prompts.py`
- Implement new retrieval strategies
- Add evaluation metrics
- Contribute research findings and optimizations

## License

See LICENSE file for details.
