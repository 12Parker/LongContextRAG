# Long Context RAG Project Organization Guide

This document explains the new organized structure of the Long Context RAG project.

## 📁 Folder Structure

```
LongContextRAG/
├── core/                    # Core RAG system components
│   ├── __init__.py
│   ├── index.py            # Main LongContextRAG class
│   ├── config.py           # Configuration management
│   └── prompts.py          # RAG prompts and templates
│
├── hybrid/                 # Hybrid attention RAG implementations
│   ├── __init__.py
│   ├── hybrid_attention_rag.py      # Core hybrid attention mechanism
│   ├── hybrid_rag_integration.py    # Integration with base RAG
│   ├── fixed_hybrid_rag.py          # Fixed dimension version
│   └── working_hybrid_rag.py        # Working implementation
│
├── training/               # Training and neural retriever components
│   ├── __init__.py
│   ├── neural_retriever.py # Neural retriever implementations
│   └── train_hybrid_rag.py # Training pipeline
│
├── testing/                # Testing and evaluation scripts
│   ├── __init__.py
│   ├── bookcorpus_integration.py    # BookCorpus dataset integration
│   ├── start_bookcorpus_testing.py  # BookCorpus test runner
│   └── simple_hybrid_test.py        # Simple component tests
│
├── examples/               # Example usage and demonstrations
│   ├── __init__.py
│   ├── examples.py         # Comprehensive examples
│   └── interactive_rag.py  # Interactive RAG demo
│
├── utils/                  # Utility scripts and setup
│   ├── __init__.py
│   ├── setup_env.py        # Environment setup
│   ├── test_setup.py       # Setup verification
│   ├── token_usage_analysis.py # Token usage analysis
│   └── setup.py            # Package setup
│
├── research/               # Research and experimentation tools
│   ├── __init__.py
│   └── research_notebook.py # Research experimentation framework
│
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── README.md              # Project documentation
└── ORGANIZATION_GUIDE.md  # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
python utils/setup_env.py
python utils/test_setup.py
```

### 2. Run Main System
```bash
python main.py
```

### 3. Individual Components

#### Core RAG System
```bash
python examples/examples.py
python examples/interactive_rag.py
```

#### Hybrid Attention RAG
```bash
python testing/simple_hybrid_test.py
python testing/start_bookcorpus_testing.py
```

#### Research and Training
```bash
python research/research_notebook.py
python training/train_hybrid_rag.py
```

## 📋 Component Descriptions

### Core (`core/`)
- **`index.py`**: Main LongContextRAG class with basic RAG functionality
- **`config.py`**: Configuration management and environment variables
- **`prompts.py`**: RAG prompts, templates, and prompt factories

### Hybrid (`hybrid/`)
- **`hybrid_attention_rag.py`**: Novel hybrid attention mechanism implementation
- **`hybrid_rag_integration.py`**: Integration layer between base RAG and hybrid attention
- **`fixed_hybrid_rag.py`**: Simplified, working version with fixed dimensions
- **`working_hybrid_rag.py`**: Alternative working implementation

### Training (`training/`)
- **`neural_retriever.py`**: Neural retriever components and end-to-end training
- **`train_hybrid_rag.py`**: Complete training pipeline for hybrid RAG system

### Testing (`testing/`)
- **`bookcorpus_integration.py`**: BookCorpus dataset integration for long context testing
- **`start_bookcorpus_testing.py`**: Quick start script for BookCorpus testing
- **`simple_hybrid_test.py`**: Simple tests for individual components

### Examples (`examples/`)
- **`examples.py`**: Comprehensive examples of RAG usage
- **`interactive_rag.py`**: Interactive command-line RAG demo

### Utils (`utils/`)
- **`setup_env.py`**: Environment setup and .env file creation
- **`test_setup.py`**: Setup verification and testing
- **`token_usage_analysis.py`**: Token usage analysis tools
- **`setup.py`**: Package setup configuration

### Research (`research/`)
- **`research_notebook.py`**: Research experimentation framework with comprehensive testing

## 🔧 Import Structure

The project now uses proper Python package structure with relative imports:

```python
# Core components
from core.index import LongContextRAG
from core.config import config
from core.prompts import RAGPrompts

# Hybrid attention
from hybrid.hybrid_attention_rag import HybridAttentionRAG
from hybrid.hybrid_rag_integration import HybridRAGIntegration

# Training components
from training.neural_retriever import NeuralRetriever
from training.train_hybrid_rag import HybridRAGTrainer

# Testing utilities
from testing.bookcorpus_integration import BookCorpusLoader
```

## 🎯 Usage Patterns

### For Basic RAG Usage
```python
from core.index import LongContextRAG

rag = LongContextRAG()
documents = rag.load_documents(["path/to/docs"])
rag.create_vectorstore(documents)
result = rag.generate_response("Your question here")
```

### For Hybrid Attention RAG
```python
from hybrid.hybrid_rag_integration import create_hybrid_rag_system

hybrid_rag = create_hybrid_rag_system()
# ... same usage as basic RAG
```

### For Research and Experimentation
```python
from research.research_notebook import run_research_experiment

results = run_research_experiment()
```

## 📊 Benefits of This Organization

1. **Clear Separation of Concerns**: Each folder has a specific purpose
2. **Easy Navigation**: Related files are grouped together
3. **Modular Design**: Components can be imported and used independently
4. **Scalable Structure**: Easy to add new components in appropriate folders
5. **Better Testing**: Testing components are clearly separated
6. **Research Focus**: Dedicated research tools and experimentation framework

## 🔄 Migration Notes

- All import statements have been updated to use the new package structure
- The main functionality remains the same
- Entry points have been preserved in `main.py`
- All existing scripts work with the new structure

## 🚀 Next Steps

1. **Development**: Add new features to appropriate folders
2. **Testing**: Use the testing framework for validation
3. **Research**: Leverage the research notebook for experiments
4. **Documentation**: Update documentation as you add features

This organization makes the project more maintainable, scalable, and easier to understand for both development and research purposes.
