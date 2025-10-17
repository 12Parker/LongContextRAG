# NarrativeQA Benchmark Integration Guide

This guide explains how to integrate the NarrativeQA benchmark into your RAG evaluation framework for comprehensive testing on complex, long-form question-answering tasks.

## 🎯 **What is NarrativeQA?**

NarrativeQA is a benchmark dataset designed to evaluate deep reading comprehension on long narratives (books and movie scripts). It's perfect for testing RAG systems because it requires:

- **Integrative reasoning** over long texts
- **Complex question answering** beyond simple fact retrieval
- **Context understanding** across entire narratives
- **Multi-hop reasoning** to connect information

### **Dataset Statistics:**
- **1,567 stories** (books and movie scripts)
- **46,765 question-answer pairs**
- **16 genres** of narrative content
- **Complex questions** requiring deep understanding

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
# Install required packages
python setup_narrativeqa.py --install-deps

# Test dataset loading
python setup_narrativeqa.py --test-dataset
```

### **2. Run Quick Test**
```bash
# Test with 5 questions
python setup_narrativeqa.py --quick-test

# Run full evaluation
python setup_narrativeqa.py --run-evaluation
```

### **3. Custom Evaluation**
```bash
# Evaluate on test set
python evaluation/narrativeqa_evaluator.py --subset test --num-questions 100

# Evaluate on validation set
python evaluation/narrativeqa_evaluator.py --subset validation --num-questions 50
```

## 📊 **Evaluation Metrics**

NarrativeQA evaluation uses multiple metrics to assess performance:

### **1. BLEU Score**
- Measures n-gram overlap between generated and reference answers
- Range: 0.0 to 1.0 (higher is better)
- Good for measuring answer quality

### **2. ROUGE-L Score**
- Measures longest common subsequence
- Range: 0.0 to 1.0 (higher is better)
- Good for measuring answer structure

### **3. METEOR Score**
- Combines precision, recall, and alignment
- Range: 0.0 to 1.0 (higher is better)
- Good for measuring semantic similarity

### **4. Exact Match Rate**
- Percentage of answers that exactly match reference
- Range: 0.0 to 1.0 (higher is better)
- Good for measuring precision

### **5. Quality Score**
- Combined metric: (BLEU + ROUGE + METEOR) / 3
- Range: 0.0 to 1.0 (higher is better)
- Overall performance indicator

## 🔍 **System Comparison**

NarrativeQA evaluates three systems:

### **1. Raw LLM Baseline**
- Uses full context without retrieval
- Baseline for comparison
- Shows performance without RAG

### **2. Standard RAG Baseline**
- Classic retrieve → augment → generate pipeline
- Uses vector database for retrieval
- Shows RAG improvement over raw LLM

### **3. Hybrid RAG System**
- Advanced RAG with hybrid attention
- Combines multiple retrieval strategies
- Shows advanced RAG capabilities

## 📈 **Expected Results**

### **Typical Performance Ranges:**

| System | BLEU | ROUGE | METEOR | Exact Match | Quality Score |
|--------|------|-------|--------|-------------|---------------|
| **Raw LLM** | 0.15-0.25 | 0.20-0.30 | 0.20-0.30 | 0.05-0.15 | 0.20-0.28 |
| **Standard RAG** | 0.20-0.30 | 0.25-0.35 | 0.25-0.35 | 0.10-0.20 | 0.23-0.33 |
| **Hybrid RAG** | 0.25-0.35 | 0.30-0.40 | 0.30-0.40 | 0.15-0.25 | 0.28-0.38 |

### **Good Performance Indicators:**
- **Quality Score > 0.25**: Decent performance
- **Quality Score > 0.30**: Good performance
- **Quality Score > 0.35**: Excellent performance
- **Exact Match > 0.15**: High precision
- **BLEU > 0.25**: Good answer quality

## 🛠️ **Configuration Options**

### **Dataset Subsets:**
- **`train`**: Training set (largest, for development)
- **`validation`**: Validation set (for tuning)
- **`test`**: Test set (for final evaluation)

### **Evaluation Parameters:**
```python
# Custom evaluation
evaluator = NarrativeQAEvaluator(
    db_path="./full_bookcorpus_db",    # Your vector database
    max_questions=100,                  # Number of questions to evaluate
    subset="test"                      # Dataset subset
)
```

### **System Configuration:**
```python
# Raw LLM configuration
raw_llm = BookCorpusRawLLMBaseline(
    max_context_tokens=8000,           # Context limit
    num_books=10                       # Number of books to use
)

# Standard RAG configuration
standard_rag = StandardRAGBaseline(
    db_path="./full_bookcorpus_db",    # Vector database path
    top_k_results=10                   # Number of chunks to retrieve
)

# Hybrid RAG configuration
hybrid_rag = WorkingHybridRAG(
    db_path="./full_bookcorpus_db",   # Vector database path
    context_size=ContextSize.LARGE     # Context size setting
)
```

## 📋 **Sample Questions**

NarrativeQA questions are designed to test deep understanding:

### **Character Analysis:**
- "Who is the main character and what motivates them?"
- "How does the protagonist change throughout the story?"

### **Plot Understanding:**
- "What is the central conflict and how is it resolved?"
- "What are the key events that drive the plot forward?"

### **Theme Analysis:**
- "What are the main themes explored in this story?"
- "How does the author convey the theme of redemption?"

### **Complex Reasoning:**
- "Why did the character make that decision given their background?"
- "How do the events in chapter 3 relate to the ending?"

## 🎯 **Best Practices**

### **1. Start Small**
- Begin with 5-10 questions for testing
- Gradually increase to 50-100 questions
- Use validation set for tuning

### **2. Monitor Performance**
- Track BLEU, ROUGE, and METEOR scores
- Compare systems systematically
- Identify improvement opportunities

### **3. Analyze Results**
- Look at individual question performance
- Identify patterns in failures
- Use results to improve your RAG system

### **4. Iterate and Improve**
- Use validation set for hyperparameter tuning
- Test different retrieval strategies
- Optimize context management

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **"Dataset loading failed"**
   - Check internet connection
   - Verify HuggingFace datasets installation
   - Try different subset (train/validation/test)

2. **"Evaluation failed"**
   - Check vector database exists
   - Verify RAG systems are initialized
   - Check API keys and configuration

3. **"Low scores"**
   - This is normal for NarrativeQA (it's challenging)
   - Focus on relative improvement between systems
   - Consider question difficulty

4. **"Memory issues"**
   - Reduce max_questions
   - Use smaller context sizes
   - Process in batches

## 📊 **Interpreting Results**

### **System Comparison:**
```python
# Example results
results = {
    'total_questions': 100,
    'best_system': 'hybrid_rag',
    'raw_llm': {
        'avg_quality_score': 0.23,
        'avg_bleu': 0.18,
        'avg_rouge': 0.25,
        'exact_match_rate': 0.08
    },
    'standard_rag': {
        'avg_quality_score': 0.28,
        'avg_bleu': 0.22,
        'avg_rouge': 0.30,
        'exact_match_rate': 0.12
    },
    'hybrid_rag': {
        'avg_quality_score': 0.32,
        'avg_bleu': 0.26,
        'avg_rouge': 0.35,
        'exact_match_rate': 0.18
    }
}
```

### **Key Insights:**
- **Hybrid RAG performs best** (0.32 vs 0.23 baseline)
- **RAG systems improve over raw LLM** (0.28-0.32 vs 0.23)
- **Exact match rates are low** (normal for complex questions)
- **Quality scores show clear improvement** (0.23 → 0.32)

## 🚀 **Advanced Usage**

### **Custom Evaluation:**
```python
from evaluation.narrativeqa_evaluator import NarrativeQAEvaluator

# Custom evaluation
evaluator = NarrativeQAEvaluator(
    db_path="./my_database",
    max_questions=200,
    subset="test"
)

# Run evaluation
results = evaluator.run_evaluation()

# Analyze results
print(f"Best system: {results['best_system']}")
print(f"Total questions: {results['total_questions']}")
```

### **Batch Processing:**
```python
# Process in batches
for batch in range(0, 1000, 100):
    evaluator = NarrativeQAEvaluator(
        max_questions=100,
        subset="test"
    )
    results = evaluator.run_evaluation()
    # Save batch results
```

## 📚 **Additional Resources**

### **NarrativeQA Paper:**
- [NarrativeQA: Reading Comprehension on Long Stories](https://arxiv.org/abs/1712.07040)

### **HuggingFace Dataset:**
- [narrativeqa Dataset](https://huggingface.co/datasets/narrativeqa)

### **Evaluation Metrics:**
- [BLEU Score](https://en.wikipedia.org/wiki/BLEU)
- [ROUGE Score](https://en.wikipedia.org/wiki/ROUGE_(metric))
- [METEOR Score](https://en.wikipedia.org/wiki/METEOR_(metric))

## 🎯 **Next Steps**

1. **Run initial evaluation** with 10-20 questions
2. **Analyze results** and identify improvement areas
3. **Tune your RAG system** based on findings
4. **Run full evaluation** with 100+ questions
5. **Compare with other benchmarks** (BEIR, MS MARCO)
6. **Publish results** or use for research

NarrativeQA provides a comprehensive benchmark for evaluating RAG systems on complex, long-form question-answering tasks! 🎉
