# Full BookCorpus Vector Database Setup Guide

This guide explains how to set up and use the full BookCorpus dataset with the standard RAG baseline system.

## Overview

The system now supports persistent vector databases that can be reused across multiple runs, eliminating the need to rebuild the database each time. This is especially important for the full BookCorpus dataset, which contains millions of documents.

## Quick Start

### 1. Build a Small Database (Recommended for Testing)

```bash
# Build with 10k documents (takes ~10-15 minutes)
python setup_full_vectordb.py --build-small
```

### 2. Test the Database

```bash
# Test the database with sample queries
python setup_full_vectordb.py --test
```

### 3. Run Standard RAG Baseline

```bash
# Run with persistent database (default)
python examples/standard_rag_baseline.py "What are the main themes in these stories?"

# Run with specific database path
python examples/standard_rag_baseline.py --db-path ./full_bookcorpus_db "Your query here"
```

## Database Build Options

### Small Database (10k documents)
- **Time**: ~10-15 minutes
- **Disk Space**: ~500MB
- **Use Case**: Testing and development

```bash
python setup_full_vectordb.py --build-small
```

### Medium Database (50k documents)
- **Time**: ~1-2 hours
- **Disk Space**: ~2-3GB
- **Use Case**: More comprehensive testing

```bash
python setup_full_vectordb.py --build-medium
```

### Full Database (Entire BookCorpus)
- **Time**: ~4-8 hours
- **Disk Space**: ~10-20GB
- **Use Case**: Production and research

```bash
python setup_full_vectordb.py --build-full
```

## Using the Standard RAG Baseline

### Basic Usage

```bash
# Single query
python examples/standard_rag_baseline.py "Your query here"

# Interactive mode
python examples/standard_rag_baseline.py --interactive

# Run full baseline tests
python examples/standard_rag_baseline.py
```

### Advanced Options

```bash
# Custom database path
python examples/standard_rag_baseline.py --db-path ./my_custom_db "Your query"

# Disable persistent database (use original method)
python examples/standard_rag_baseline.py --no-persistent-db "Your query"

# Custom parameters
python examples/standard_rag_baseline.py --chunk-size 1000 --top-k 15 "Your query"
```

## Database Management

### Check Database Status

```bash
# Show database statistics
python setup_full_vectordb.py --stats

# Test database functionality
python setup_full_vectordb.py --test
```

### Database Configuration

The database is configured with these default settings:
- **Chunk Size**: 500 words
- **Chunk Overlap**: 50 words
- **Embedding Model**: all-MiniLM-L6-v2
- **Batch Size**: 100 chunks

## File Structure

```
LongContextRAG/
├── full_bookcorpus_db/          # Persistent database directory
│   ├── chroma.sqlite3           # ChromaDB database file
│   └── ...                      # Other ChromaDB files
├── VectorDB/
│   ├── build_full_bookcorpus_db.py  # Full database builder
│   ├── vectordb_manager.py         # Database manager
│   └── build_db.py                 # Original builder
├── examples/
│   └── standard_rag_baseline.py    # Updated baseline
└── setup_full_vectordb.py          # Setup script
```

## Troubleshooting

### Database Not Found

If you get "Database not ready" errors:

1. Check if the database exists:
   ```bash
   python setup_full_vectordb.py --stats
   ```

2. If not, build it:
   ```bash
   python setup_full_vectordb.py --build-small
   ```

### Out of Memory Errors

If you encounter memory issues:

1. Reduce batch size in the configuration
2. Use a smaller database first:
   ```bash
   python setup_full_vectordb.py --build-small
   ```

### Disk Space Issues

Monitor disk usage:
```bash
du -sh ./full_bookcorpus_db/
```

## Performance Tips

### For Development
- Use `--build-small` for quick testing
- Use `--build-medium` for more comprehensive testing

### For Production
- Use `--build-full` for complete dataset coverage
- Consider using SSD storage for better performance
- Monitor disk space during full build

## Configuration Options

### Database Path
```bash
# Custom database location
python setup_full_vectordb.py --build-small --db-path ./my_database
python examples/standard_rag_baseline.py --db-path ./my_database "Your query"
```

### Chunk Parameters
The database is built with these parameters:
- **Chunk Size**: 500 words (configurable)
- **Chunk Overlap**: 50 words (configurable)
- **Minimum Text Length**: 100 characters

## Monitoring and Maintenance

### Check Database Health
```bash
# Test database functionality
python setup_full_vectordb.py --test

# Show detailed statistics
python setup_full_vectordb.py --stats
```

### Database Statistics
The system provides detailed statistics:
- Total chunks in database
- Chunk size and overlap settings
- Embedding model used
- Database path and collection name

## Integration with Other Systems

### Using with Hybrid RAG
The persistent database can be used with other RAG systems:

```python
from VectorDB.vectordb_manager import VectorDBManager

# Initialize manager
manager = VectorDBManager(db_path="./full_bookcorpus_db")

# Check if ready
if manager.is_database_ready():
    results = manager.query("your query", n_results=5)
```

### Custom Queries
You can query the database directly:

```python
from VectorDB.vectordb_manager import VectorDBManager

manager = VectorDBManager()
results = manager.query("science fiction adventure", n_results=10)

for i, (doc, distance, metadata) in enumerate(zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
)):
    print(f"Result {i+1}: {doc[:200]}...")
```

## Next Steps

1. **Start Small**: Build a small database for testing
2. **Test Functionality**: Verify everything works correctly
3. **Scale Up**: Build larger databases as needed
4. **Monitor Performance**: Track query times and accuracy
5. **Optimize**: Adjust parameters based on your needs

## Support

If you encounter issues:

1. Check the logs in `full_bookcorpus_build.log`
2. Verify disk space and memory
3. Test with smaller databases first
4. Check the database statistics

For more information, see the main project documentation.
