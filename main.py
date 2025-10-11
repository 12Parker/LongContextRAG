#!/usr/bin/env python3
"""
Main entry point for the Long Context RAG system.

This script provides easy access to the main functionality of the system.
"""

import sys
import os
from pathlib import Path

def main():
    """Main entry point with menu options."""
    print("🚀 Long Context RAG System")
    print("=" * 40)
    print("Choose an option:")
    print("1. Interactive RAG Demo")
    print("2. Run Examples")
    print("3. Test Setup")
    print("4. BookCorpus Testing")
    print("5. Research Notebook")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\n🔍 Starting Interactive RAG Demo...")
                from examples.interactive_rag import main as interactive_main
                interactive_main()
                break
                
            elif choice == "2":
                print("\n📚 Running Examples...")
                from examples.examples import run_all_examples
                run_all_examples()
                break
                
            elif choice == "3":
                print("\n🧪 Testing Setup...")
                from utils.test_setup import main as test_main
                test_main()
                break
                
            elif choice == "4":
                print("\n📖 Starting BookCorpus Testing...")
                from testing.start_bookcorpus_testing import main as bookcorpus_main
                bookcorpus_main()
                break
                
            elif choice == "5":
                print("\n🔬 Starting Research Notebook...")
                from research.research_notebook import run_research_experiment
                run_research_experiment()
                break
                
            elif choice == "6":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
