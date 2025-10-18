# Project Organization Guide

## 📁 **Directory Structure**

```
LongContextRAG/
├── 📁 core/                    # Core configuration and utilities
│   ├── config.py              # Main configuration
│   ├── prompts.py             # Prompt templates
│   └── long_context_config.py # Long context settings
│
├── 📁 data/                    # Data storage
│   ├── bookcorpus/            # BookCorpus dataset
│   ├── bookcorpus_processed/  # Processed BookCorpus
│   └── pg19_raw/              # PG19 dataset
│
├── 📁 docs/                    # Documentation
│   ├── SETUP_GUIDE.md         # Setup instructions
│   ├── NARRATIVEQA_EVALUATION_GUIDE.md # Evaluation guide
│   ├── BASE_LLM_NARRATIVEQA_SYSTEM_DESIGN.md # Base LLM design
│   └── RAG_NARRATIVEQA_SYSTEM_DESIGN.md # RAG system design
│
├── 📁 evaluation/              # Evaluation frameworks
│   ├── narrativeqa_evaluator.py # Main NarrativeQA evaluator
│   ├── qa_metrics.py          # QA-specific metrics
│   ├── bleu_evaluator.py      # BLEU evaluation
│   └── rag_evaluation.py      # RAG evaluation
│
├── 📁 examples/                # Example implementations
│   ├── narrativeqa_base_llm.py # Base LLM for NarrativeQA
│   ├── narrativeqa_rag_baseline.py # RAG baseline
│   ├── standard_rag_baseline.py # Standard RAG
│   └── compare_responses.py   # Response comparison
│
├── 📁 hybrid/                   # Hybrid RAG implementations
│   ├── narrativeqa_hybrid_rag.py # NarrativeQA Hybrid RAG
│   ├── working_hybrid_rag.py   # Working Hybrid RAG
│   ├── hybrid_attention_rag.py # Hybrid attention RAG
│   └── hybrid_rag_integration.py # Integration utilities
│
├── 📁 scripts/                 # Organized scripts
│   ├── 📁 analysis/            # Analysis scripts
│   │   ├── analyze_qa_metrics.py # QA metrics analysis
│   │   ├── analyze_bleu_results.py # BLEU analysis
│   │   ├── analyze_narrativeqa_comparison.py # Comparison analysis
│   │   └── summarize_base_llm_results.py # Results summarization
│   │
│   ├── 📁 comparison/          # Comparison scripts
│   │   └── compare_systems_narrativeqa.py # System comparison
│   │
│   └── 📁 setup/               # Setup scripts
│       ├── setup_full_vectordb.py # Vector DB setup
│       ├── setup_narrativeqa.py # NarrativeQA setup
│       └── monitor_build.py    # Build monitoring
│
├── 📁 testing/                 # Testing utilities
│   ├── bookcorpus_integration.py # BookCorpus testing
│   └── start_bookcorpus_testing.py # Testing launcher
│
├── 📁 training/                # Training scripts
│   ├── neural_retriever.py    # Neural retriever training
│   └── train_hybrid_rag.py    # Hybrid RAG training
│
├── 📁 utils/                   # Utility functions
│   ├── setup_env.py           # Environment setup
│   ├── test_setup.py          # Setup testing
│   └── token_usage_analysis.py # Token usage analysis
│
├── 📁 VectorDB/                # Vector database utilities
│   ├── build_db.py            # Database builder
│   ├── build_full_bookcorpus_db.py # Full BookCorpus builder
│   └── vectordb_manager.py    # Database manager
│
├── 📁 results/                 # Evaluation results
│   └── bookcorpus/            # BookCorpus results
│
├── run.py                     # Main script runner
├── main.py                    # Main entry point
├── requirements.txt           # Core dependencies
├── requirements_narrativeqa.txt # NarrativeQA dependencies
└── README.md                  # Project overview
```

## 🚀 **Script Organization**

### **Main Script Runner**
The `run.py` script provides easy access to all organized scripts:

```bash
# Setup scripts
python run.py setup-vectordb --build-medium
python run.py setup-narrativeqa
python run.py monitor-build

# Comparison scripts
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag --num-questions 5

# Analysis scripts
python run.py analyze-qa
python run.py analyze-bleu
python run.py analyze-comparison
python run.py summarize-results
```

### **Direct Script Access**
You can also run scripts directly:

```bash
# Analysis scripts
python scripts/analysis/analyze_qa_metrics.py
python scripts/analysis/analyze_bleu_results.py

# Comparison scripts
python scripts/comparison/compare_systems_narrativeqa.py --systems base_llm --num-questions 5

# Setup scripts
python scripts/setup/setup_full_vectordb.py --build-medium
python scripts/setup/monitor_build.py
```

## 📋 **Script Categories**

### **🔧 Setup Scripts (`scripts/setup/`)**
- **`setup_full_vectordb.py`**: Build BookCorpus vector database
- **`setup_narrativeqa.py`**: Setup NarrativeQA evaluation
- **`monitor_build.py`**: Monitor database build progress

### **⚖️ Comparison Scripts (`scripts/comparison/`)**
- **`compare_systems_narrativeqa.py`**: Compare RAG systems on NarrativeQA

### **📊 Analysis Scripts (`scripts/analysis/`)**
- **`analyze_qa_metrics.py`**: Analyze QA evaluation results
- **`analyze_bleu_results.py`**: Analyze BLEU evaluation results
- **`analyze_narrativeqa_comparison.py`**: Analyze system comparison results
- **`summarize_base_llm_results.py`**: Summarize base LLM results

## 🎯 **Usage Examples**

### **Quick Evaluation**
```bash
# Run quick comparison
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag --num-questions 2

# Analyze results
python run.py analyze-qa
```

### **Comprehensive Evaluation**
```bash
# Run full comparison
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag,narrativeqa_rag --num-questions 10

# Analyze all metrics
python run.py analyze-qa
python run.py analyze-bleu
```

### **Setup and Monitoring**
```bash
# Setup vector database
python run.py setup-vectordb --build-medium

# Monitor build progress
python run.py monitor-build

# Setup NarrativeQA
python run.py setup-narrativeqa
```

## 📚 **Documentation Organization**

### **Setup and Configuration**
- **`SETUP_GUIDE.md`**: Initial project setup
- **`NARRATIVEQA_EVALUATION_GUIDE.md`**: Evaluation instructions

### **System Design**
- **`BASE_LLM_NARRATIVEQA_SYSTEM_DESIGN.md`**: Base LLM architecture
- **`RAG_NARRATIVEQA_SYSTEM_DESIGN.md`**: RAG system design

### **Project Organization**
- **`PROJECT_ORGANIZATION.md`**: This file - project structure guide

## 🔄 **Migration from Old Structure**

### **Before (Scattered Scripts)**
```bash
# Scripts were in main directory
python compare_systems_narrativeqa.py --systems base_llm --num-questions 5
python analyze_qa_metrics.py
python setup_full_vectordb.py --build-medium
```

### **After (Organized Structure)**
```bash
# Using main runner (recommended)
python run.py compare-systems --systems base_llm --num-questions 5
python run.py analyze-qa
python run.py setup-vectordb --build-medium

# Or direct access
python scripts/comparison/compare_systems_narrativeqa.py --systems base_llm --num-questions 5
python scripts/analysis/analyze_qa_metrics.py
python scripts/setup/setup_full_vectordb.py --build-medium
```

## ✅ **Benefits of Organization**

### **1. Clear Separation of Concerns**
- **Setup scripts**: Database and environment setup
- **Comparison scripts**: System evaluation
- **Analysis scripts**: Results analysis

### **2. Easy Discovery**
- **`run.py`**: Single entry point for all scripts
- **Organized directories**: Clear script categories
- **Documentation**: Comprehensive usage guides

### **3. Maintainability**
- **Modular structure**: Easy to add new scripts
- **Clear dependencies**: Scripts in appropriate folders
- **Consistent patterns**: Similar scripts grouped together

### **4. User Experience**
- **Simple commands**: `python run.py <command>`
- **Help system**: Built-in help and examples
- **Flexible access**: Direct script access when needed

This organization makes the project much more maintainable and user-friendly! 🎯
