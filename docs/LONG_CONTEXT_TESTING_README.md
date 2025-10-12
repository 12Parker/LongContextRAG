# Long Context Testing Guide

This guide explains how to test the WorkingHybridRAG system with context lengths above 32k tokens.

## Overview

The system now supports testing with various context lengths:
- **Standard**: 8k tokens
- **Medium**: 16k tokens  
- **Large**: 32k tokens (default)
- **XLarge**: 64k tokens
- **Ultra**: 128k+ tokens

## Quick Start

### 1. Basic Long Context Test

```bash
# Test with large context (32k tokens)
python run_long_context_test.py large

# Test with ultra context (128k+ tokens)
python run_long_context_test.py ultra

# Run comprehensive test across all sizes
python run_long_context_test.py comprehensive
```

### 2. Advanced Testing

```bash
# Run the full long context testing suite
python examples/long_context_testing.py
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Long Context Testing Settings
LONG_CONTEXT_SIZE=large  # Options: standard, medium, large, xlarge, ultra
LONG_CONTEXT_MAX_TOKENS=32000
LONG_CONTEXT_NUM_DOCS=10000
LONG_CONTEXT_CHUNK_SIZE=2000
```

### Context Size Configurations

| Size | Max Tokens | Documents | Chunk Size | Top-K Results |
|------|------------|-----------|------------|---------------|
| Standard | 8,000 | 5,000 | 1,000 | 5 |
| Medium | 16,000 | 7,500 | 1,500 | 7 |
| Large | 32,000 | 10,000 | 2,000 | 10 |
| XLarge | 64,000 | 15,000 | 3,000 | 15 |
| Ultra | 128,000+ | 20,000 | 4,000 | 20 |

## Usage Examples

### Python API

```python
from core.long_context_config import LongContextManager, ContextSize
from hybrid.working_hybrid_rag import WorkingHybridRAG

# Create manager for ultra-long context
manager = LongContextManager()
result = manager.test_context_length(ContextSize.ULTRA)

# Or create hybrid RAG directly with long context config
from core.long_context_config import LongContextConfig
config = LongContextConfig.for_context_size(ContextSize.XLARGE)

hybrid_rag = WorkingHybridRAG(
    use_hybrid_attention=True,
    vectordb_config=config.to_vectordb_config()
)

# Create vector store with long context settings
hybrid_rag.create_vectorstore(
    use_vectordb=True,
    num_documents=config.num_documents
)

# Test with long context queries
response = hybrid_rag.generate_response(
    "Analyze the narrative structure and character development patterns across multiple chapters",
    task_type='qa'
)
```

### Custom Configuration

```python
from core.long_context_config import LongContextConfig

# Create custom configuration
config = LongContextConfig(
    max_context_tokens=100000,
    num_documents=25000,
    chunk_size=5000,
    top_k_results=25,
    max_retrieved_chunks=50
)

# Use with hybrid RAG
hybrid_rag = WorkingHybridRAG(
    use_hybrid_attention=True,
    vectordb_config=config.to_vectordb_config()
)
```

## Performance Considerations

### Token Usage

- **Standard**: ~7k tokens average
- **Medium**: ~14k tokens average
- **Large**: ~28k tokens average
- **XLarge**: ~56k tokens average
- **Ultra**: ~112k tokens average

### Processing Time

- **Standard**: ~30-60 seconds
- **Medium**: ~60-120 seconds
- **Large**: ~120-300 seconds
- **XLarge**: ~300-600 seconds
- **Ultra**: ~600-1200 seconds

### Memory Requirements

- **Standard**: ~2-4 GB RAM
- **Medium**: ~4-8 GB RAM
- **Large**: ~8-16 GB RAM
- **XLarge**: ~16-32 GB RAM
- **Ultra**: ~32+ GB RAM

## Best Practices

### 1. Start Small
Begin with standard or medium context sizes to verify your setup works correctly.

### 2. Monitor Resources
- Watch memory usage during processing
- Monitor API token usage and costs
- Check disk space for vector stores

### 3. Optimize Configuration
- Adjust chunk sizes based on your content
- Increase document count for richer context
- Use appropriate embedding models

### 4. Query Design
For longer contexts, use more comprehensive queries:
```python
# Good for long context
"Analyze the narrative structure, character development, thematic elements, and literary devices across the entire collection"

# Less effective for long context
"What is the main theme?"
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `num_documents` or `chunk_size`
   - Use smaller context sizes first

2. **API Rate Limits**
   - Add delays between requests
   - Use smaller batch sizes

3. **Poor Results**
   - Increase document count
   - Adjust chunk size and overlap
   - Use more specific queries

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Results Analysis

### Output Files

- `results/long_context_test_results.json` - Detailed test results
- `vector_store_long_*/` - Vector database files
- Log files with processing details

### Key Metrics

- **Context Utilization**: How much of the available context is used
- **Retrieval Quality**: Relevance of retrieved documents
- **Response Quality**: Coherence and accuracy of responses
- **Processing Time**: Time to process and respond

## Advanced Features

### Token Counting

The system automatically counts tokens using tiktoken:
```python
from examples.long_context_testing import LongContextTester
tester = LongContextTester()
token_count = tester.count_tokens(text)
```

### Custom Queries

Create domain-specific queries for your use case:
```python
custom_queries = [
    "Analyze the technical implementation details and architectural patterns",
    "Examine the data flow and processing pipelines",
    "Evaluate the performance characteristics and optimization opportunities"
]
```

### Batch Testing

Test multiple configurations in parallel:
```python
context_sizes = [ContextSize.LARGE, ContextSize.XLARGE, ContextSize.ULTRA]
for size in context_sizes:
    result = manager.test_context_length(size)
    # Process results...
```

## Integration with Existing Workflows

The long context testing integrates seamlessly with existing RAG workflows:

1. **VectorDBBuilder**: Uses the same vector database creation process
2. **Hybrid Attention**: Maintains all hybrid attention capabilities
3. **Base RAG**: Falls back to base RAG when needed
4. **Configuration**: Extends existing configuration system

## Next Steps

1. **Experiment**: Try different context sizes with your specific use case
2. **Optimize**: Adjust parameters based on your performance requirements
3. **Scale**: Gradually increase context size as you become comfortable
4. **Monitor**: Track performance and costs over time
5. **Iterate**: Refine queries and configurations based on results
