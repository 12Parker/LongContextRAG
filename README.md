# NarrativeQA RAG Evaluation Framework

A focused research framework for evaluating Retrieval Augmented Generation (RAG) systems against the NarrativeQA benchmark. This project provides comprehensive tools for comparing Base LLM, Standard RAG, and Hybrid RAG approaches on long-form question-answering tasks.

## 🎯 **Project Focus: NarrativeQA Evaluation**

This repository is specifically designed for evaluating RAG systems against the NarrativeQA benchmark, which tests deep reading comprehension over long narratives. The framework includes three main evaluation approaches:

- **Base LLM**: Direct processing of full story context
- **Standard RAG**: Traditional retrieval-augmented generation
- **Hybrid RAG**: Advanced RAG with neural retrieval and attention mechanisms

## ✨ **Key Features**

- **NarrativeQA Integration**: Full support for the NarrativeQA benchmark dataset
- **Multiple RAG Systems**: Base LLM, Standard RAG, and Hybrid RAG implementations
- **Comprehensive Evaluation**: BLEU, ROUGE, F1, METEOR, BERTScore, and Exact Match metrics
- **Organized Scripts**: Clean command-line interface with `run.py`
- **Results Management**: Automated results storage and analysis
- **Documentation**: Complete guides for setup, evaluation, and analysis

## 🚀 **Quick Start**

### 1. **Setup Environment**
```bash
# Clone and navigate to repository
cd LongContextRAG

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_narrativeqa.txt
```

### 2. **Configure API Keys**
```bash
# Copy environment template
cp env.template .env

# Edit .env file and add your API keys:
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. **Setup NarrativeQA**
```bash
# Setup NarrativeQA evaluation
python run.py setup-narrativeqa --install-deps
python run.py setup-narrativeqa --test-dataset
```

### 4. **Run Evaluation**
```bash
# Quick comparison (2 questions)
python run.py compare-systems --systems base_llm --num-questions 2

# Full comparison (10 questions)
python run.py compare-systems --systems base_llm,narrativeqa_rag,narrativeqa_hybrid_rag --num-questions 10

# Analyze results
python run.py analyze-qa
python run.py analyze-bleu
```

## 📁 **Repository Structure**

```
LongContextRAG/
├── 📁 core/                     # Core configuration
│   ├── config.py               # Main configuration
│   ├── prompts.py              # Prompt templates
│   └── long_context_config.py  # Long context settings
│
├── 📁 evaluation/              # Evaluation frameworks
│   ├── narrativeqa_evaluator.py # Main NarrativeQA evaluator
│   ├── qa_metrics.py           # QA-specific metrics (F1, METEOR, etc.)
│   ├── bleu_evaluator.py       # BLEU and ROUGE evaluation
│   └── rag_evaluation.py       # RAG evaluation utilities
│
├── 📁 examples/                # NarrativeQA examples
│   ├── narrativeqa_base_llm.py # Base LLM implementation
│   └── narrativeqa_rag_baseline.py # Standard RAG implementation
│
├── 📁 hybrid/                  # Hybrid RAG system
│   └── narrativeqa_hybrid_rag.py # Advanced RAG with neural retrieval
│
├── 📁 scripts/                 # Organized scripts
│   ├── 📁 setup/               # Setup scripts
│   │   └── setup_narrativeqa.py # NarrativeQA setup
│   ├── 📁 comparison/          # Comparison scripts
│   │   └── compare_systems_narrativeqa.py # System comparison
│   └── 📁 analysis/            # Analysis scripts
│       ├── analyze_qa_metrics.py # QA metrics analysis
│       └── analyze_bleu_results.py # BLEU analysis
│
├── 📁 docs/                    # Documentation
│   ├── NARRATIVEQA_EVALUATION_GUIDE.md # Complete evaluation guide
│   ├── BASE_LLM_NARRATIVEQA_SYSTEM_DESIGN.md # Base LLM architecture
│   ├── RAG_NARRATIVEQA_SYSTEM_DESIGN.md # RAG architecture
│   └── PROJECT_ORGANIZATION.md # Project organization guide
│
├── 📁 results/                 # Evaluation results
│   └── system_comparisons/     # System comparison results
│
├── run.py                      # Main script runner
├── main.py                     # Original entry point
├── requirements.txt            # Core dependencies
├── requirements_narrativeqa.txt # NarrativeQA dependencies
└── env.template                # Environment template
```

## 🛠️ **Available Commands**

### **Setup Commands**
```bash
python run.py setup-narrativeqa --install-deps    # Install dependencies
python run.py setup-narrativeqa --test-dataset    # Test dataset loading
python run.py setup-narrativeqa --quick-test      # Quick functionality test
```

### **Comparison Commands**
```bash
# Compare specific systems
python run.py compare-systems --systems base_llm --num-questions 5
python run.py compare-systems --systems base_llm,narrativeqa_rag --num-questions 10
python run.py compare-systems --systems base_llm,narrativeqa_rag,narrativeqa_hybrid_rag --num-questions 15

# Use different dataset subsets
python run.py compare-systems --systems base_llm --subset validation --num-questions 5
```

### **Analysis Commands**
```bash
python run.py analyze-qa        # Analyze QA metrics (F1, METEOR, BERTScore, etc.)
python run.py analyze-bleu      # Analyze BLEU and ROUGE metrics
```

## 📊 **Evaluation Metrics**

The framework provides comprehensive evaluation metrics:

### **QA-Specific Metrics**
- **Exact Match (EM)**: Binary match with reference answers
- **F1 Score**: Token-level harmonic mean of precision and recall
- **BERTScore**: Semantic similarity using BERT embeddings
- **METEOR**: Alignment considering synonyms and paraphrases

### **Traditional Metrics**
- **BLEU**: N-gram overlap with reference answers
- **ROUGE-1/2/L**: Recall-oriented understudy for gisting evaluation
- **Word Overlap**: Simple word-level overlap

### **Performance Metrics**
- **Response Time**: Generation latency
- **Context Length**: Input context size
- **Answer Length**: Generated response length
- **Retrieved Docs**: Number of retrieved documents

## 🔬 **System Architectures**

### **Base LLM**
- Direct processing of full story context
- No retrieval mechanism
- Serves as baseline for comparison

### **Standard RAG**
- Traditional similarity-based retrieval
- ChromaDB vector database
- HuggingFace embeddings (all-MiniLM-L6-v2)

### **Hybrid RAG**
- Advanced RAG with neural retrieval
- Multi-head attention mechanisms
- Neural re-ranking of retrieved documents
- Dynamic context integration

## 📈 **Example Results**

```
📊 QA SYSTEM COMPARISON
============================================================
System               Questions  EM     F1     BERT   METEOR   BLEU   ROUGE-1  Time    
------------------------------------------------------------------------------------------
base_llm             10         0.000  0.087  0.000  0.110    0.004  0.060    3.18    
narrativeqa_rag      10         0.100  0.234  0.156  0.189    0.023  0.145    4.52    
narrativeqa_hybrid_rag 10       0.200  0.312  0.201  0.245    0.045  0.198    5.23    
```

## 📚 **Documentation**

- **[NarrativeQA Evaluation Guide](docs/NARRATIVEQA_EVALUATION_GUIDE.md)**: Complete guide to running evaluations
- **[Base LLM System Design](docs/BASE_LLM_NARRATIVEQA_SYSTEM_DESIGN.md)**: Architecture and implementation
- **[RAG System Design](docs/RAG_NARRATIVEQA_SYSTEM_DESIGN.md)**: RAG architecture and implementation
- **[Project Organization](docs/PROJECT_ORGANIZATION.md)**: Repository structure and organization

## 🔧 **Configuration**

Key configuration options in `.env`:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults provided)
OPENAI_MODEL=gpt-4o-mini
CHUNK_SIZE=1000
TOP_K_RESULTS=5
MAX_STORY_LENGTH=8000
```

## 🎯 **Research Applications**

This framework is designed for research in:

- **Long Context Processing**: How to effectively handle long narratives
- **Retrieval Quality**: Optimizing document retrieval for QA tasks
- **Neural Retrieval**: Advanced retrieval mechanisms
- **Attention Mechanisms**: Multi-head attention in RAG systems
- **Evaluation Metrics**: Comprehensive QA evaluation methodologies

## 🔒 **Security**

- API keys are stored in `.env` (not tracked in git)
- Only `env.template` is tracked (contains no real keys)
- Results are automatically saved to `results/` (ignored by git)

## 📝 **Usage Examples**

### **Basic Evaluation**
```bash
# Quick test with 2 questions
python run.py compare-systems --systems base_llm --num-questions 2

# Full evaluation with all systems
python run.py compare-systems --systems base_llm,narrativeqa_rag,narrativeqa_hybrid_rag --num-questions 10
```

### **Analysis**
```bash
# Analyze QA metrics
python run.py analyze-qa

# Analyze BLEU/ROUGE metrics
python run.py analyze-bleu
```

### **Direct Script Usage**
```bash
# Run comparison directly
python scripts/comparison/compare_systems_narrativeqa.py --systems base_llm --num-questions 5

# Run analysis directly
python scripts/analysis/analyze_qa_metrics.py
```

## 🤝 **Contributing**

This is a research framework. Contributions welcome:

- New evaluation metrics
- Additional RAG architectures
- Performance optimizations
- Documentation improvements

## 📄 **License**

See LICENSE file for details.

---

**🎯 Focus**: This repository is specifically designed for NarrativeQA evaluation and RAG research. All components are optimized for long-form question-answering tasks and comprehensive system comparison.