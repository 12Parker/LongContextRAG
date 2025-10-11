# 📚 BookCorpus Testing Guide for Hybrid Attention RAG

This guide explains how to test your hybrid attention RAG system with BookCorpus-like data for long context research.

## 🎯 Why BookCorpus for RAG Testing?

BookCorpus is ideal for testing long context RAG systems because:

- **Long-form content**: Books provide extended narratives and coherent long contexts
- **Diverse genres**: Different types of content test various attention patterns
- **Complex relationships**: Long narratives require understanding of distant relationships
- **Real-world scenarios**: Books represent realistic long context use cases

## 🚀 Quick Start

### 1. Run the Basic Test
```bash
# Activate your virtual environment
source venv/bin/activate

# Run the BookCorpus integration test
python start_bookcorpus_testing.py
```

### 2. What the Test Does
The test will:
- Create sample book-like texts (8 different genres)
- Process them into document chunks
- Generate test queries for each genre
- Compare base RAG vs hybrid attention RAG
- Save results for analysis

## 📊 Understanding the Results

### Sample Output
```
📚 Testing BookCorpus Integration with Hybrid Attention RAG
============================================================

📊 TEST SUMMARY
Books loaded: 8
Documents processed: 160
Test queries: 24

🔍 SYSTEM COMPARISON:
  base_rag:
    Avg context length: 3245
    Avg retrieved docs: 5.0
  hybrid_rag:
    Avg context length: 4567
    Avg retrieved docs: 8.0

📁 Results saved to: results/bookcorpus
✅ BookCorpus integration test completed!
```

### Key Metrics
- **Context Length**: How much context each system uses
- **Retrieved Docs**: Number of document chunks retrieved
- **Success Rate**: Percentage of successful responses
- **Processing Time**: Time taken per query

## 🔧 Customizing the Test

### 1. Modify Configuration
Edit the configuration in `bookcorpus_integration.py`:

```python
config = BookCorpusConfig(
    max_books=8,              # Number of books to test
    min_book_length=10000,    # Minimum book length
    max_book_length=80000,    # Maximum book length
    chunk_size=2000,          # Size of document chunks
    chunk_overlap=200,        # Overlap between chunks
    test_queries_per_book=3,  # Queries per book
)
```

### 2. Add Your Own Books
To test with your own long documents:

```python
# Create your own book data
my_book = {
    'title': 'My Research Paper',
    'genre': 'scientific',
    'content': 'Your long document content here...',
    'metadata': {
        'author': 'Your Name',
        'year': 2024
    }
}

# Add to the loader
loader = BookCorpusLoader(config)
loader.books.append(my_book)
```

### 3. Custom Test Queries
Create genre-specific test queries:

```python
custom_queries = [
    "What are the main arguments presented?",
    "How does the author support their claims?",
    "What evidence is provided for the conclusions?",
    "What are the limitations of the study?",
    "What future research is suggested?"
]
```

## 🧪 Advanced Testing

### 1. Test Different Context Lengths
```python
# Test with different context window sizes
context_lengths = [1000, 2000, 4000, 8000, 16000]

for length in context_lengths:
    config.chunk_size = length
    results = test_with_config(config)
    analyze_context_performance(results, length)
```

### 2. Compare Attention Patterns
```python
# Analyze attention patterns across different genres
for genre in ['technical', 'narrative', 'scientific']:
    genre_queries = get_queries_for_genre(genre)
    attention_analysis = analyze_attention_patterns(genre_queries)
    plot_attention_heatmap(attention_analysis, genre)
```

### 3. Test Long Context Scenarios
```python
# Test with very long contexts
long_context_config = BookCorpusConfig(
    max_book_length=200000,  # Very long books
    chunk_size=4000,         # Larger chunks
    chunk_overlap=400,       # More overlap
)

# Test cross-chapter relationships
cross_chapter_queries = [
    "How does the conclusion relate to the introduction?",
    "What themes connect the different chapters?",
    "How do the arguments develop throughout the book?"
]
```

## 📈 Analyzing Results

### 1. Performance Comparison
```python
# Compare systems across different metrics
def analyze_performance(results):
    base_rag = results['system_comparisons']['base_rag']
    hybrid_rag = results['system_comparisons']['hybrid_rag']
    
    print(f"Context Length Improvement: {hybrid_rag['avg_context_length'] / base_rag['avg_context_length']:.2f}x")
    print(f"Retrieval Improvement: {hybrid_rag['avg_retrieved_docs'] / base_rag['avg_retrieved_docs']:.2f}x")
```

### 2. Genre-Specific Analysis
```python
# Analyze performance by genre
def analyze_by_genre(results):
    for genre in ['technical', 'narrative', 'scientific']:
        genre_results = filter_by_genre(results, genre)
        print(f"{genre.capitalize()} Genre:")
        print(f"  Avg context length: {genre_results['avg_context_length']}")
        print(f"  Success rate: {genre_results['success_rate']}")
```

### 3. Attention Pattern Analysis
```python
# Analyze attention patterns
def analyze_attention_patterns(results):
    for query_result in results['query_results']:
        attention_shape = query_result.get('attention_shape')
        if attention_shape:
            print(f"Attention shape: {attention_shape}")
            print(f"Attention diversity: {calculate_attention_diversity(attention_shape)}")
```

## 🔬 Research Applications

### 1. Long Context Understanding
Test how well the system maintains context across long sequences:
- Cross-reference information from different parts of the book
- Track character or concept development over time
- Understand narrative structure and themes

### 2. Attention Mechanism Analysis
Study how attention patterns differ:
- Between genres (technical vs narrative)
- Across different context lengths
- For different types of queries

### 3. Retrieval Quality Assessment
Evaluate retrieval effectiveness:
- Relevance of retrieved chunks
- Coverage of important information
- Handling of long-range dependencies

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Issues with Large Books**
   ```python
   # Reduce book size or chunk size
   config.max_book_length = 50000
   config.chunk_size = 1000
   ```

2. **Slow Processing**
   ```python
   # Reduce number of test queries
   config.test_queries_per_book = 2
   config.max_books = 4
   ```

3. **API Rate Limits**
   ```python
   # Add delays between API calls
   import time
   time.sleep(1)  # 1 second delay
   ```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Next Steps

### 1. Scale Up Testing
- Test with real BookCorpus data (if available)
- Increase the number of books and queries
- Test with different languages

### 2. Advanced Analysis
- Implement automatic evaluation metrics
- Create visualization tools for attention patterns
- Develop genre-specific evaluation criteria

### 3. Integration with Research
- Use results for academic papers
- Compare with other long context methods
- Develop new evaluation benchmarks

## 📖 Example Research Questions

Your BookCorpus testing can help answer:

1. **How does hybrid attention perform on different text genres?**
2. **What is the optimal context length for different types of queries?**
3. **How do attention patterns differ between local and global relationships?**
4. **Can the system maintain coherence across very long contexts?**
5. **How does retrieval quality change with document length?**

## 🎯 Success Metrics

Track these metrics for your research:

- **Context Utilization**: How much of the available context is used effectively
- **Coherence Score**: How well the system maintains narrative coherence
- **Retrieval Precision**: Relevance of retrieved document chunks
- **Attention Diversity**: How attention is distributed across the context
- **Processing Efficiency**: Time and computational cost per query

This BookCorpus testing framework provides a solid foundation for evaluating your hybrid attention RAG system on long context tasks. Use it as a starting point for your research and adapt it to your specific needs!
