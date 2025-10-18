# Base LLM with NarrativeQA: System Design

## Overview

The Base LLM system for NarrativeQA is a **direct language model approach** that processes complete story contexts without any retrieval or chunking mechanisms. It serves as a baseline for comparing against more sophisticated RAG systems.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NARRATIVEQA BASE LLM SYSTEM                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │ NarrativeQA │    │   Context    │    │   LLM Engine    │    │
│  │   Dataset   │───▶│  Builder     │───▶│   (GPT-4)       │    │
│  │             │    │              │    │                 │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Question  │    │   Story +    │    │   Generated     │    │
│  │   Text      │    │   Summary    │    │   Answer        │    │
│  │             │    │              │    │                 │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Data Input Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INPUT LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │ NarrativeQA │    │   Question  │    │   Story Text    │    │
│  │   Dataset   │    │   Text      │    │   (54K chars)   │    │
│  │             │    │             │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Question Data Structure                     │    │
│  │  {                                                      │    │
│  │    "question": "Who is Mark Hunter?",                  │    │
│  │    "story": "This is a story about Mark Hunter...",   │    │
│  │    "summary": "A story about a radio station...",      │    │
│  │    "answers": ["He is a high school student..."]       │    │
│  │  }                                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Context Processing Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT PROCESSING LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   Story     │    │   Summary   │    │  Context        │    │
│  │   Input     │    │   Input     │    │  Builder        │    │
│  │             │    │             │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Context Building Process                   │    │
│  │                                                         │    │
│  │  1. Check if summary exists                            │    │
│  │     → Add "Summary: {summary}"                         │    │
│  │                                                         │    │
│  │  2. Check if story exists                              │    │
│  │     → Truncate if > 8000 chars                        │    │
│  │     → Add "Story: {story}"                             │    │
│  │                                                         │    │
│  │  3. Combine with newlines                              │    │
│  │     → "Summary: ...\n\nStory: ..."                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Prompt Engineering Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT ENGINEERING LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   Question  │    │   Context   │    │   Final        │    │
│  │   Text      │    │   String    │    │   Prompt       │    │
│  │             │    │             │    │                │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Prompt Template                            │    │
│  │                                                         │    │
│  │  "Based on the following story and summary,             │    │
│  │   answer the question.                                  │    │
│  │                                                         │    │
│  │  {context}                                             │    │
│  │                                                         │    │
│  │  Question: {question}                                   │    │
│  │                                                         │    │
│  │  Please provide a comprehensive answer based on the     │    │
│  │  story content. If the story doesn't contain enough     │    │
│  │  information to answer the question, please say so     │    │
│  │  and provide what information you can from the         │    │
│  │  available context."                                   │    │
│  └─────────────────────────────────────────────────────────┘
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. LLM Processing Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM PROCESSING LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   Prompt    │    │   OpenAI    │    │   Generated     │    │
│  │   Input     │───▶│   GPT-4     │───▶│   Response     │    │
│  │             │    │             │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LLM Configuration                         │    │
│  │                                                         │    │
│  │  • Model: GPT-4                                        │    │
│  │  • Temperature: 0.1 (low for consistency)              │    │
│  │  • Max Tokens: Auto                                    │    │
│  │  • Context Window: ~8,000 characters                   │    │
│  │                                                         │    │
│  │  Processing Flow:                                      │    │
│  │  1. Send prompt to OpenAI API                          │    │
│  │  2. Wait for response                                  │    │
│  │  3. Extract content from response                      │    │
│  │  4. Return generated answer                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Response Processing Layer
```
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE PROCESSING LAYER                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │   LLM       │    │   Response  │    │   Final        │    │
│  │   Response  │───▶│   Parser    │───▶│   Answer       │    │
│  │             │    │             │    │                │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Response Structure                         │    │
│  │                                                         │    │
│  │  {                                                      │    │
│  │    "response": "Mark Hunter is a high school...",     │    │
│  │    "context": "Summary: ...\n\nStory: ...",          │    │
│  │    "retrieved_docs": 0,                               │    │
│  │    "context_length": 1524,                            │    │
│  │    "context_tokens": 1524,                            │    │
│  │    "response_time": 2.70,                             │    │
│  │    "method": "narrativeqa_base_llm",                  │    │
│  │    "story_length": 127,                               │    │
│  │    "summary_length": 0                                │    │
│  │  }                                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NarrativeQA Dataset                                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Question +  │                                               │
│  │ Story Data  │                                               │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Context     │                                               │
│  │ Builder     │                                               │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Prompt      │                                               │
│  │ Template    │                                               │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ OpenAI      │                                               │
│  │ GPT-4       │                                               │
│  └─────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │ Generated   │                                               │
│  │ Answer      │                                               │
│  └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. **Simplicity First**
- No complex retrieval mechanisms
- Direct question-to-answer mapping
- Minimal preprocessing

### 2. **Complete Context Access**
- Full story text provided to LLM
- No information loss from chunking
- Maintains narrative coherence

### 3. **Context Window Management**
- Automatic truncation at 8,000 characters
- Preserves most important story content
- Handles long narratives gracefully

### 4. **Consistent Output Format**
- Standardized response structure
- Comprehensive metadata
- Easy integration with comparison systems

## Performance Characteristics

### **Strengths**
- ✅ **High Context Utilization**: ~1,524 tokens average
- ✅ **Complete Information Access**: No retrieval limitations
- ✅ **Narrative Coherence**: Maintains story flow
- ✅ **Simple Architecture**: Easy to understand and debug

### **Limitations**
- ❌ **Context Window Constraints**: Limited to ~8,000 characters
- ❌ **No Retrieval Optimization**: Processes entire story
- ❌ **Computational Cost**: Higher token usage
- ❌ **Scalability Issues**: Cannot handle multiple documents

## Comparison with RAG Systems

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM COMPARISON                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base LLM Approach:                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │ Question +  │    │   Direct    │    │   Answer        │    │
│  │ Full Story  │───▶│   LLM      │───▶│   (Complete)    │    │
│  │             │    │   Process  │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│                                                                 │
│  RAG Approach:                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │ Question    │    │  Retrieval  │    │   Answer        │    │
│  │             │───▶│  + LLM      │───▶│   (Focused)     │    │
│  │             │    │  Process    │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### **Core Class Structure**
```python
class NarrativeQABaseLLM:
    def __init__(self, max_story_length=8000):
        self.max_story_length = max_story_length
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    def generate_response(self, question, story, summary=""):
        context = self._build_context(story, summary)
        prompt = self._create_prompt(question, context)
        response = self.llm.invoke(prompt)
        return self._format_response(response, context)
```

### **Context Building Algorithm**
```python
def _build_context(self, story, summary):
    context_parts = []
    
    if summary:
        context_parts.append(f"Summary: {summary}")
    
    if story:
        if len(story) > self.max_story_length:
            story = story[:self.max_story_length] + "..."
        context_parts.append(f"Story: {story}")
    
    return "\n\n".join(context_parts)
```

## Conclusion

The Base LLM system for NarrativeQA represents a **straightforward, effective approach** to story-based question answering. While it may not be the most sophisticated method, it demonstrates the power of **complete context access** and serves as an important baseline for evaluating more complex RAG systems.

The system's strength lies in its simplicity and ability to maintain narrative coherence, making it particularly effective for single-story scenarios where complete context is available and beneficial.
