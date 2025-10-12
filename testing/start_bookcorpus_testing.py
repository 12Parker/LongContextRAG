#!/usr/bin/env python3
"""
Quick Start Script for BookCorpus Testing with Hybrid Attention RAG

This script provides a simple way to start testing your hybrid attention RAG system
with BookCorpus-like data for long context research.
"""

import os
import sys
from pathlib import Path

def main():
    print("📚 BookCorpus Testing for Hybrid Attention RAG")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("hybrid/hybrid_attention_rag.py").exists():
        print("❌ Error: Please run this script from the LongContextRAG directory")
        sys.exit(1)
    
    print("🚀 Starting BookCorpus integration test...")
    print()
    
    try:
        # Import and run the test
        from testing.bookcorpus_integration import test_bookcorpus_integration
        test_bookcorpus_integration()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Check the logs for more details")
        sys.exit(1)

if __name__ == "__main__":
    main()
