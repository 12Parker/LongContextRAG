# NarrativeQA Evaluation Guide

## Overview

This guide provides step-by-step instructions for running the NarrativeQA dataset evaluation and analyzing the results using comprehensive QA-specific metrics.

## 🚀 **Quick Start**

### **1. Basic Evaluation (2 questions)**
```bash
# Activate virtual environment
source venv/bin/activate

# Using organized scripts
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag --num-questions 2

# Or directly
python scripts/comparison/compare_systems_narrativeqa.py --systems base_llm,narrativeqa_hybrid_rag --num-questions 2
```

### **2. Comprehensive Evaluation (10 questions)**
```bash
# Run larger comparison with all systems
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag,narrativeqa_rag --num-questions 10

# Or directly
python scripts/comparison/compare_systems_narrativeqa.py --systems base_llm,narrativeqa_hybrid_rag,narrativeqa_rag --num-questions 10
```

### **3. Analyze Results**
```bash
# Analyze QA metrics
python run.py analyze-qa

# Analyze BLEU metrics (alternative)
python run.py analyze-bleu

# Or directly
python scripts/analysis/analyze_qa_metrics.py
python scripts/analysis/analyze_bleu_results.py
```

## 📋 **Prerequisites**

### **Required Dependencies**
```bash
# Install core dependencies
pip install -r requirements.txt

# Install NarrativeQA-specific dependencies
pip install -r requirements_narrativeqa.txt

# Download NLTK data
python -m nltk.downloader punkt_tab wordnet omw-1.4
```

### **Environment Setup**
```bash
# Copy environment template
cp env.template .env

# Edit .env file with your OpenAI API key
# OPENAI_API_KEY=your_api_key_here
```

## 🎯 **Available Systems**

### **1. Base LLM (`base_llm`)**
- **Description**: Direct processing of full story context
- **Strengths**: Reliable, comprehensive answers, no retrieval errors
- **Use case**: Baseline comparison, reliability testing

### **2. NarrativeQA Hybrid RAG (`narrativeqa_hybrid_rag`)**
- **Description**: Advanced RAG with neural retrieval and attention mechanisms
- **Strengths**: Best accuracy, efficient context usage, neural components
- **Use case**: Best performance, production deployment

### **3. NarrativeQA RAG (`narrativeqa_rag`)**
- **Description**: Standard RAG with similarity search
- **Strengths**: Fast responses, concise answers
- **Use case**: Speed testing, standard RAG comparison

## 🔧 **Running Evaluations**

### **Basic Evaluation Commands**

#### **Test Single System**
```bash
# Test Base LLM only
python run.py compare-systems --systems base_llm --num-questions 5

# Test Hybrid RAG only
python run.py compare-systems --systems narrativeqa_hybrid_rag --num-questions 5
```

#### **Compare Multiple Systems**
```bash
# Compare Base LLM vs Hybrid RAG
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag --num-questions 10

# Compare all three systems
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag,narrativeqa_rag --num-questions 10
```

#### **Custom Question Count**
```bash
# Quick test (2 questions)
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag --num-questions 2

# Medium test (5 questions)
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag --num-questions 5

# Comprehensive test (10+ questions)
python run.py compare-systems --systems base_llm,narrativeqa_hybrid_rag,narrativeqa_rag --num-questions 10
```

### **Advanced Evaluation Options**

#### **Interactive Mode**
```bash
# Test Base LLM interactively
python baselines/narrativeqa_base_llm.py --interactive

# Test with custom story
python baselines/narrativeqa_base_llm.py --story "Your custom story here" --question "Your question here"
```

#### **Individual System Testing**
```bash
# Test Base LLM standalone
python baselines/narrativeqa_base_llm.py "Who is Mark Hunter?" --story "Your story text"

# Test RAG baseline
python baselines/narrativeqa_rag_baseline.py --question "Your question" --story-text "Your story text"

# Test Hybrid RAG
python hybrid/narrativeqa_hybrid_rag.py --question "Your question" --story-text "Your story text"
```

## 📊 **Understanding the Results**

### **Output Files**
- **Results**: `system_comparison_narrativeqa_YYYYMMDD_HHMMSS.json`
- **Logs**: Console output with real-time progress
- **Vector DBs**: `./narrativeqa_vectordb/`, `./narrativeqa_hybrid_vectordb/`

### **Key Metrics Explained**

#### **Primary QA Metrics**
- **F1 Score**: Token-level precision and recall (0.0-1.0, higher is better)
- **METEOR**: Semantic similarity considering synonyms (0.0-1.0, higher is better)
- **Exact Match**: Binary exact string match (0.0 or 1.0)
- **Token Precision**: How many generated tokens appear in reference
- **Token Recall**: How many reference tokens appear in generated answer

#### **Traditional Metrics**
- **BLEU Score**: N-gram precision (typically low for QA tasks)
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

#### **Performance Metrics**
- **Response Time**: Average time per question
- **Answer Length**: Average characters in generated answers
- **Context Tokens**: Number of tokens used for context
- **Retrieved Docs**: Number of documents retrieved (RAG systems)

## 🔍 **Analyzing Results**

### **1. Basic Analysis**
```bash
# Analyze QA metrics (recommended)
python run.py analyze-qa

# Analyze BLEU metrics (alternative)
python run.py analyze-bleu

# Or directly
python scripts/analysis/analyze_qa_metrics.py
python scripts/analysis/analyze_bleu_results.py
```

### **2. Understanding the Output**

#### **QA Metrics Analysis**
```
📊 QA METRICS ANALYSIS
============================================================

🔍 BASE_LLM:
----------------------------------------
  exact_match    : 0.0000 (max: 0.0000, min: 0.0000)
  f1_score       : 0.0642 (max: 0.1164, min: 0.0000)
  meteor_score   : 0.0949 (max: 0.1518, min: 0.0000)
  token_precision: 0.0348 (max: 0.0640, min: 0.0000)
  token_recall   : 0.5089 (max: 0.7500, min: 0.0000)
```

#### **System Comparison**
```
📊 QA SYSTEM COMPARISON
============================================================
System               Questions  EM     F1     BERT   METEOR   BLEU   ROUGE-1  Time    
------------------------------------------------------------------------------------------
base_llm             10         0.000  0.064  0.000  0.095    0.002  0.041    3.04    
narrativeqa_hybrid_rag 10         0.000  0.075  0.000  0.107    0.002  0.049    4.08    
narrativeqa_rag      10         0.000  0.052  0.000  0.075    0.002  0.036    2.84    
```

### **3. Interpreting Results**

#### **F1 Score Interpretation**
- **0.0-0.2**: Low overlap (check answer relevance)
- **0.2-0.5**: Moderate overlap (good token-level similarity)
- **0.5+**: High overlap (excellent token-level match)

#### **METEOR Score Interpretation**
- **0.0-0.2**: Low semantic similarity
- **0.2-0.5**: Moderate semantic similarity
- **0.5+**: High semantic similarity

#### **Performance Insights**
- **Exact Match (0.0)**: Normal for paraphrased answers
- **Token Recall (50%+)**: Good content coverage
- **Response Time**: Consider speed vs accuracy trade-offs

## 🎯 **Best Practices**

### **1. Evaluation Strategy**
```bash
# Start with quick test
python compare_systems_narrativeqa.py --systems base_llm,narrativeqa_hybrid_rag --num-questions 2

# Run comprehensive test
python compare_systems_narrativeqa.py --systems base_llm,narrativeqa_hybrid_rag,narrativeqa_rag --num-questions 10

# Analyze results
python analyze_qa_metrics.py
```

### **2. System Selection**
- **For Accuracy**: Use `narrativeqa_hybrid_rag` (best F1 and METEOR scores)
- **For Reliability**: Use `base_llm` (no retrieval errors, comprehensive answers)
- **For Speed**: Use `narrativeqa_rag` (fastest responses)

### **3. Question Count Guidelines**
- **2-3 questions**: Quick testing, development
- **5-10 questions**: Standard evaluation
- **10+ questions**: Comprehensive analysis

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Install missing dependencies
pip install nltk rouge-score

# Download NLTK data
python -m nltk.downloader punkt_tab wordnet omw-1.4
```

#### **API Key Issues**
```bash
# Check environment file
cat .env

# Ensure OpenAI API key is set
export OPENAI_API_KEY=your_api_key_here
```

#### **Retrieval Errors**
- **Standard RAG**: Some questions may fail with "Error finding id"
- **Solution**: Use Hybrid RAG or Base LLM for reliability

#### **Memory Issues**
- **Large datasets**: Reduce question count for testing
- **Vector DBs**: Clear old databases if needed

### **Debug Mode**
```bash
# Run with verbose output
python compare_systems_narrativeqa.py --systems base_llm --num-questions 1 --verbose

# Test individual components
python baselines/narrativeqa_base_llm.py --interactive
```

## 📈 **Expected Results**

### **Typical Performance Ranges**

#### **F1 Scores**
- **Base LLM**: 0.05-0.08
- **Hybrid RAG**: 0.06-0.09 (best)
- **Standard RAG**: 0.04-0.06

#### **METEOR Scores**
- **Base LLM**: 0.08-0.12
- **Hybrid RAG**: 0.09-0.13 (best)
- **Standard RAG**: 0.06-0.10

#### **Response Times**
- **Base LLM**: 2-4 seconds
- **Hybrid RAG**: 3-5 seconds
- **Standard RAG**: 2-3 seconds (fastest)

### **Performance Insights**
- **Hybrid RAG typically wins** on accuracy metrics
- **Base LLM provides reliable baseline** with comprehensive answers
- **Standard RAG is fastest** but may have retrieval issues
- **All systems show low exact match** (normal for paraphrased answers)

## 🎉 **Next Steps**

### **1. Further Analysis**
- Run larger question sets (20+ questions)
- Compare with other benchmarks
- Analyze specific question types

### **2. System Improvement**
- Fix Standard RAG retrieval errors
- Optimize Hybrid RAG response time
- Implement additional metrics (BERTScore)

### **3. Production Deployment**
- Use Hybrid RAG for best accuracy
- Use Base LLM for reliability
- Monitor performance metrics

## 📚 **Additional Resources**

### **Documentation**
- `docs/BASE_LLM_NARRATIVEQA_SYSTEM_DESIGN.md`: Base LLM architecture
- `docs/RAG_NARRATIVEQA_SYSTEM_DESIGN.md`: RAG system design
- `docs/QA_EVALUATION_METRICS_GUIDE.md`: Detailed metrics explanation

### **Code Examples**
- `baselines/narrativeqa_base_llm.py`: Base LLM implementation
- `baselines/narrativeqa_rag_baseline.py`: Standard RAG implementation
- `hybrid/narrativeqa_hybrid_rag.py`: Hybrid RAG implementation

### **Analysis Tools**
- `analyze_qa_metrics.py`: QA-specific metrics analysis
- `analyze_bleu_results.py`: BLEU metrics analysis
- `compare_systems_narrativeqa.py`: Main evaluation script

This guide provides everything needed to run comprehensive NarrativeQA evaluations and analyze the results effectively! 🎯
