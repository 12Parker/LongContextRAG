#!/usr/bin/env python3
"""
NarrativeQA Base LLM

This implements a direct LLM approach for NarrativeQA questions.
It provides the complete story context to the LLM without any retrieval mechanisms.
"""

import sys
import os
from pathlib import Path
import time
from typing import Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import config
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class NarrativeQABaseLLM:
    """
    Base LLM system for NarrativeQA questions.
    
    This provides a direct LLM approach that uses the complete story context
    without any retrieval or chunking mechanisms.
    """
    
    def __init__(self, max_story_length: int = 8000):
        """
        Initialize the NarrativeQA Base LLM system.
        
        Args:
            max_story_length: Maximum length of story to include in context
        """
        self.max_story_length = max_story_length
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
    
    def generate_response(self, question: str, story: str = "", summary: str = "") -> Dict[str, Any]:
        """
        Generate a response using the base LLM approach.
        
        Args:
            question: The question to answer
            story: The complete story text
            summary: Optional summary of the story
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Build context from story and summary
            context = self._build_context(story, summary)
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Generate response
            response = self.llm.invoke(prompt)
            generated_answer = response.content if hasattr(response, 'content') else str(response)
            
            elapsed_time = time.time() - start_time
            
            return {
                'response': generated_answer,
                'context': context,
                'retrieved_docs': 0,  # Base LLM doesn't retrieve documents
                'context_length': len(context),
                'context_tokens': len(context.split()),  # Approximate token count
                'response_time': elapsed_time,
                'method': 'narrativeqa_base_llm',
                'story_length': len(story),
                'summary_length': len(summary)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"Error generating response: {str(e)}",
                'context': '',
                'retrieved_docs': 0,
                'context_length': 0,
                'context_tokens': 0,
                'response_time': time.time() - start_time,
                'method': 'narrativeqa_base_llm',
                'error': str(e)
            }
    
    def _build_context(self, story: str, summary: str = "") -> str:
        """
        Build context from story and summary.
        
        Args:
            story: The complete story text
            summary: Optional summary of the story
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        if summary:
            context_parts.append(f"Summary: {summary}")
        
        if story:
            # Truncate story if too long
            if len(story) > self.max_story_length:
                story = story[:self.max_story_length] + "..."
            context_parts.append(f"Story: {story}")
        
        return "\n\n".join(context_parts) if context_parts else "No context available."
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create the prompt for the LLM.
        
        Args:
            question: The question to answer
            context: The story context
            
        Returns:
            Formatted prompt string
        """
        return f"""Based on the following story and summary, answer the question.

{context}

Question: {question}

Please provide a comprehensive answer based on the story content. If the story doesn't contain enough information to answer the question, please say so and provide what information you can from the available context."""

def main():
    """Main function for testing the NarrativeQA Base LLM."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NarrativeQA Base LLM")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--story", type=str, default="", help="Story text")
    parser.add_argument("--summary", type=str, default="", help="Story summary")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max-story-length", type=int, default=8000, help="Maximum story length")
    
    args = parser.parse_args()
    
    # Initialize Base LLM system
    print("🚀 Initializing NarrativeQA Base LLM...")
    base_llm = NarrativeQABaseLLM(max_story_length=args.max_story_length)
    
    if args.interactive:
        print("\n💬 Interactive mode - Enter questions (type 'quit' to exit)")
        while True:
            question = input("\n🔍 Enter your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            story = input("📚 Enter story text (or press Enter for default): ").strip()
            if not story:
                story = "This is a sample story about Mark Hunter, a high school student in Phoenix who runs a pirate radio station from his parents' basement."
            
            print(f"\n🤖 Processing: {question}")
            result = base_llm.generate_response(question, story)
            
            print(f"\n📝 Response:")
            print(f"{result['response']}")
            print(f"\n📊 Metadata:")
            print(f"  Context length: {result['context_length']}")
            print(f"  Context tokens: {result['context_tokens']}")
            print(f"  Response time: {result['response_time']:.2f}s")
            print(f"  Story length: {result['story_length']}")
    
    elif args.question:
        print(f"\n🤖 Processing: {args.question}")
        result = base_llm.generate_response(args.question, args.story, args.summary)
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Context length: {result['context_length']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")
        print(f"  Story length: {result['story_length']}")
    
    else:
        # Default test
        test_question = "Who is the main character in the story?"
        test_story = "This is a story about Mark Hunter, a high school student in Phoenix who runs a pirate radio station from his parents' basement."
        
        print(f"\n🤖 Testing with: {test_question}")
        result = base_llm.generate_response(test_question, test_story)
        
        print(f"\n📝 Response:")
        print(f"{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  Context length: {result['context_length']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")
        print(f"  Story length: {result['story_length']}")

if __name__ == "__main__":
    main()
