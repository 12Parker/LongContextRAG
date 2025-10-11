#!/usr/bin/env python3
"""
Simple test script to verify the Long Context RAG setup is working correctly.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.index import LongContextRAG
        print("✅ LongContextRAG import successful")
    except ImportError as e:
        print(f"❌ LongContextRAG import failed: {e}")
        return False
    
    try:
        from core.config import config
        print("✅ Config import successful")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from core.prompts import RAGPrompts
        print("✅ Prompts import successful")
    except ImportError as e:
        print(f"❌ Prompts import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from core.config import config
        print(f"✅ Configuration loaded")
        print(f"   - OpenAI Model: {config.OPENAI_MODEL}")
        print(f"   - Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"   - Chunk Size: {config.CHUNK_SIZE}")
        print(f"   - Top K Results: {config.TOP_K_RESULTS}")
        
        # Check if API key is set (don't print it)
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
            print("✅ OpenAI API key is configured")
        else:
            print("⚠️  OpenAI API key needs to be set in .env file")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directories():
    """Test that required directories exist."""
    print("\nTesting directories...")
    
    required_dirs = ["data", "vector_store"]
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            return False
    
    return True

def test_basic_rag_initialization():
    """Test basic RAG system initialization (without API calls)."""
    print("\nTesting RAG system initialization...")
    
    try:
        from core.index import LongContextRAG
        
        # This will fail if API key is not set, but we can catch that
        try:
            rag = LongContextRAG()
            print("✅ RAG system initialized successfully")
            return True
        except ValueError as e:
            if "OPENAI_API_KEY" in str(e):
                print("⚠️  RAG system initialization requires OpenAI API key")
                print("   This is expected if you haven't set your API key yet")
                return True
            else:
                print(f"❌ RAG system initialization failed: {e}")
                return False
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Long Context RAG Setup")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_directories,
        test_basic_rag_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Add your OpenAI API key to the .env file")
        print("2. Run: python index.py")
        print("3. Run: python examples.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
