# Long Context RAG Research Project

A comprehensive research framework for exploring Retrieval Augmented Generation (RAG) with long context language models. This project provides tools and examples for studying how to effectively combine large language models with external knowledge retrieval for research applications.

## Features

- **OpenAI API Integration**: Full support for GPT-4 and other OpenAI models
- **Vector Store Management**: ChromaDB integration for efficient document storage and retrieval
- **Long Context Support**: Optimized for handling large documents and extended contexts
- **Multiple Prompt Templates**: Pre-built prompts for various research scenarios
- **Flexible Configuration**: Easy-to-customize settings for different use cases
- **Comprehensive Examples**: Ready-to-run examples demonstrating different RAG techniques

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
   python interactive_rag.py
   ```

### Alternative: Use the activation script
```bash
chmod +x activate.sh
./activate.sh
```

## Project Structure

```
LongContextRAG/
├── index.py              # Main RAG implementation
├── config.py             # Configuration management
├── prompts.py            # Prompt templates for different scenarios
├── examples.py           # Usage examples and test cases
├── interactive_rag.py    # Interactive RAG system for testing
├── test_setup.py         # Setup verification script
├── setup.py              # Project setup script
├── setup_env.py          # Environment configuration script
├── activate.sh           # Virtual environment activation script
├── requirements.txt      # Python dependencies
├── env.template          # Environment variables template
├── .env                  # Your environment variables (not tracked by git)
├── .gitignore           # Git ignore file (protects your API keys)
├── data/                 # Directory for your documents
├── vector_store/         # Persistent vector store
├── logs/                 # Log files
├── results/              # Research results and outputs
└── README.md            # This file
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
