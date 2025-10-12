#!/usr/bin/env python3
"""
Quick script to run long context tests with WorkingHybridRAG

Usage:
    python run_long_context_test.py [context_size]
    
Context sizes: standard, medium, large, xlarge, ultra
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.long_context_config import LongContextManager, ContextSize
from hybrid.working_hybrid_rag import WorkingHybridRAG
import time

def main():
    """Main function to run long context tests."""
    
    # Parse command line arguments
    context_size_str = sys.argv[1] if len(sys.argv) > 1 else "large"
    
    try:
        context_size = ContextSize(context_size_str.lower())
    except ValueError:
        print(f"❌ Invalid context size: {context_size_str}")
        print("Valid options: standard, medium, large, xlarge, ultra")
        return
    
    print(f"🚀 Running Long Context Test: {context_size.value.upper()}")
    print("=" * 60)
    
    # Create long context manager
    manager = LongContextManager()
    
    # Run test for specific context size
    start_time = time.time()
    result = manager.test_context_length(context_size)
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Total test time: {elapsed_time:.2f} seconds")
    print(f"✅ Test completed successfully!")
    
    # Print detailed results
    print(f"\n📊 Detailed Results:")
    print(f"   Context size: {result['context_size']}")
    print(f"   Max tokens: {result['max_context_tokens']:,}")
    print(f"   Documents processed: {result['num_documents']:,}")
    print(f"   Total chunks: {result['total_chunks']:,}")
    
    print(f"\n🔍 Query Results:")
    for i, qr in enumerate(result['query_results'], 1):
        print(f"   Query {i}: {qr['retrieved_docs']} docs, {qr['context_length']} chars")
        print(f"            Method: {qr['method']}")
        print(f"            Response: {qr['response_length']} chars")

def run_comprehensive_test():
    """Run comprehensive test across all context sizes."""
    print("🚀 Running Comprehensive Long Context Test")
    print("=" * 60)
    
    manager = LongContextManager()
    manager.run_comprehensive_test()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
        run_comprehensive_test()
    else:
        main()
