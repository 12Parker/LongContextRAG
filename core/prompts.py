"""
Prompt templates for different RAG scenarios and research use cases.
"""

from langchain.prompts import ChatPromptTemplate, PromptTemplate

class RAGPrompts:
    """Collection of prompt templates for various RAG scenarios."""
    
    # Basic RAG prompt
    BASIC_RAG = ChatPromptTemplate.from_template("""
    Context:
    {context}
    
    Question: {question}
    
    Please provide a comprehensive answer based on the provided context. 
    If the context doesn't contain enough information to answer the question, 
    please say so and provide what information you can.
    """)
    
    # Research-focused prompt
    RESEARCH_RAG = ChatPromptTemplate.from_template("""
    You are a research assistant analyzing the following context to answer a research question.
    
    Context:
    {context}
    
    Research Question: {question}
    
    Please provide a detailed, analytical response that:
    1. Directly addresses the research question
    2. Cites specific information from the context
    3. Identifies any gaps in the available information
    4. Suggests areas for further investigation if applicable
    
    Format your response with clear sections and bullet points where appropriate.
    """)
    
    # Long context analysis prompt
    LONG_CONTEXT_ANALYSIS = ChatPromptTemplate.from_template("""
    You are analyzing a long document for research purposes. Given the following context:
    
    Context:
    {context}
    
    Question: {question}
    
    Please provide a thorough analysis that:
    1. Synthesizes information from across the entire context
    2. Identifies key themes and patterns
    3. Draws connections between different parts of the document
    4. Provides specific examples and evidence from the text
    5. Notes any contradictions or inconsistencies
    
    Focus on providing a comprehensive understanding rather than just answering the specific question.
    """)
    
    # Comparative analysis prompt
    COMPARATIVE_ANALYSIS = ChatPromptTemplate.from_template("""
    You are conducting a comparative analysis based on the following context:
    
    Context:
    {context}
    
    Question: {question}
    
    Please provide a comparative analysis that:
    1. Identifies different perspectives or approaches mentioned
    2. Compares and contrasts key concepts or methods
    3. Evaluates strengths and weaknesses of different approaches
    4. Draws conclusions based on the comparison
    5. Suggests which approach might be most suitable for different scenarios
    
    Structure your response with clear comparisons and evidence from the context.
    """)
    
    # Technical documentation prompt
    TECHNICAL_DOCUMENTATION = ChatPromptTemplate.from_template("""
    You are a technical expert analyzing documentation. Based on the following context:
    
    Context:
    {context}
    
    Technical Question: {question}
    
    Please provide a technical response that:
    1. Explains the technical concepts clearly
    2. Provides step-by-step procedures where applicable
    3. Includes relevant code examples or technical specifications
    4. Identifies potential issues or considerations
    5. Suggests best practices or recommendations
    
    Use technical terminology appropriately and provide detailed explanations.
    """)
    
    # Summarization prompt for long documents
    SUMMARIZATION = ChatPromptTemplate.from_template("""
    Please provide a comprehensive summary of the following content:
    
    Content:
    {context}
    
    Focus on:
    1. Key points and main arguments
    2. Important findings or conclusions
    3. Methodology or approach used (if applicable)
    4. Implications or significance
    5. Any limitations or areas for future work
    
    Structure the summary with clear headings and bullet points.
    """)
    
    # Question generation prompt
    QUESTION_GENERATION = ChatPromptTemplate.from_template("""
    Based on the following content, generate thoughtful research questions that could be explored:
    
    Content:
    {context}
    
    Please generate 5-7 research questions that:
    1. Build upon the information presented
    2. Explore different aspects of the topic
    3. Could lead to meaningful research directions
    4. Range from specific to more general inquiries
    5. Consider both theoretical and practical implications
    
    Format each question clearly and explain why it would be valuable to investigate.
    """)

class PromptFactory:
    """Factory class for creating and customizing prompts."""
    
    @staticmethod
    def create_custom_prompt(system_message: str, user_template: str) -> ChatPromptTemplate:
        """
        Create a custom prompt template.
        
        Args:
            system_message: The system message to set the context
            user_template: The user message template with placeholders
            
        Returns:
            ChatPromptTemplate instance
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_template)
        ])
    
    @staticmethod
    def create_research_prompt(research_focus: str) -> ChatPromptTemplate:
        """
        Create a research-focused prompt for a specific domain.
        
        Args:
            research_focus: The specific research domain (e.g., "machine learning", "NLP")
            
        Returns:
            ChatPromptTemplate instance
        """
        system_message = f"""You are a {research_focus} research expert. Your task is to provide 
        detailed, accurate, and insightful analysis based on the provided context. Focus on 
        research-quality responses that could contribute to academic or professional understanding 
        of {research_focus} topics."""
        
        user_template = """
        Context:
        {context}
        
        Research Question: {question}
        
        Please provide a comprehensive research-quality response that includes:
        1. Detailed analysis of the question
        2. Evidence-based conclusions
        3. Critical evaluation of the information
        4. Suggestions for further research
        5. Practical implications where applicable
        """
        
        return PromptFactory.create_custom_prompt(system_message, user_template)
    
    @staticmethod
    def create_evaluation_prompt() -> ChatPromptTemplate:
        """
        Create a prompt for evaluating RAG responses.
        
        Returns:
            ChatPromptTemplate instance
        """
        system_message = """You are an expert evaluator of AI-generated responses. Your task is to 
        assess the quality, accuracy, and usefulness of responses generated by a RAG system."""
        
        user_template = """
        Original Question: {question}
        
        Retrieved Context:
        {context}
        
        Generated Response:
        {response}
        
        Please evaluate this response on the following criteria:
        1. Accuracy: Is the information correct and well-supported by the context?
        2. Completeness: Does it fully address the question?
        3. Clarity: Is the response clear and well-structured?
        4. Relevance: Is the information relevant to the question?
        5. Usefulness: Would this response be helpful to a researcher?
        
        Provide a score from 1-5 for each criterion and an overall assessment.
        """
        
        return PromptFactory.create_custom_prompt(system_message, user_template)

# Example usage and testing
def test_prompts():
    """Test function to demonstrate prompt usage."""
    # Example context and question
    context = "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."
    question = "What is machine learning?"
    
    # Test different prompts
    prompts_to_test = [
        ("Basic RAG", RAGPrompts.BASIC_RAG),
        ("Research RAG", RAGPrompts.RESEARCH_RAG),
        ("Long Context Analysis", RAGPrompts.LONG_CONTEXT_ANALYSIS),
        ("Technical Documentation", RAGPrompts.TECHNICAL_DOCUMENTATION)
    ]
    
    for name, prompt in prompts_to_test:
        print(f"\n=== {name} ===")
        formatted_prompt = prompt.format(context=context, question=question)
        print(formatted_prompt)

if __name__ == "__main__":
    test_prompts()
