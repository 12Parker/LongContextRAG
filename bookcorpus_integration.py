"""
BookCorpus Integration for Hybrid Attention RAG Testing

This module provides tools for loading, processing, and testing the hybrid attention RAG system
with the BookCorpus dataset, which is ideal for long context research.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import requests
import zipfile
from tqdm import tqdm
import random

from hybrid_rag_integration import HybridRAGIntegration, create_hybrid_rag_system
from index import LongContextRAG
from langchain.schema import Document

logger = logging.getLogger(__name__)

@dataclass
class BookCorpusConfig:
    """Configuration for BookCorpus integration."""
    # Dataset parameters
    max_books: int = 100  # Number of books to load (for testing)
    min_book_length: int = 1000  # Minimum characters per book
    max_book_length: int = 100000  # Maximum characters per book
    
    # Processing parameters
    chunk_size: int = 2000  # Size of document chunks
    chunk_overlap: int = 200  # Overlap between chunks
    max_chunks_per_book: int = 50  # Maximum chunks per book
    
    # Testing parameters
    test_queries_per_book: int = 5  # Number of test queries per book
    context_lengths: List[int] = None  # Different context lengths to test
    
    # Paths
    data_dir: str = "data/bookcorpus"
    processed_dir: str = "data/bookcorpus_processed"
    results_dir: str = "results/bookcorpus"

class BookCorpusLoader:
    """
    Loader for BookCorpus dataset with preprocessing for RAG testing.
    """
    
    def __init__(self, config: BookCorpusConfig):
        self.config = config
        self.books = []
        self.processed_books = []
        
        # Create directories
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(config.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    def load_sample_books(self) -> List[Dict[str, Any]]:
        """
        Load sample books for testing. Since BookCorpus is large, we'll create
        sample long-form texts that simulate book-like content.
        """
        logger.info("Creating sample book-like texts for testing...")
        
        # Create sample books with different characteristics
        sample_books = [
            self._create_sample_book("The Art of Machine Learning", "technical", 50000),
            self._create_sample_book("A Journey Through Time", "narrative", 75000),
            self._create_sample_book("Scientific Discoveries", "scientific", 60000),
            self._create_sample_book("Philosophy and Ethics", "philosophical", 40000),
            self._create_sample_book("Historical Events", "historical", 80000),
            self._create_sample_book("Future Technologies", "futuristic", 55000),
            self._create_sample_book("Nature and Environment", "descriptive", 45000),
            self._create_sample_book("Human Psychology", "analytical", 70000),
        ]
        
        # Filter by length
        filtered_books = []
        for book in sample_books:
            if self.config.min_book_length <= len(book['content']) <= self.config.max_book_length:
                filtered_books.append(book)
        
        self.books = filtered_books[:self.config.max_books]
        logger.info(f"Loaded {len(self.books)} books for testing")
        
        return self.books
    
    def _create_sample_book(self, title: str, genre: str, target_length: int) -> Dict[str, Any]:
        """Create a sample book with realistic content structure."""
        
        # Base content templates by genre
        templates = {
            "technical": self._get_technical_content,
            "narrative": self._get_narrative_content,
            "scientific": self._get_scientific_content,
            "philosophical": self._get_philosophical_content,
            "historical": self._get_historical_content,
            "futuristic": self._get_futuristic_content,
            "descriptive": self._get_descriptive_content,
            "analytical": self._get_analytical_content,
        }
        
        content_generator = templates.get(genre, self._get_narrative_content)
        content = content_generator(title, target_length)
        
        return {
            'title': title,
            'genre': genre,
            'content': content,
            'length': len(content),
            'chapters': self._extract_chapters(content),
            'metadata': {
                'author': f"Sample Author",
                'year': 2023,
                'genre': genre,
                'word_count': len(content.split()),
                'character_count': len(content)
            }
        }
    
    def _get_technical_content(self, title: str, target_length: int) -> str:
        """Generate technical content."""
        content = f"""
# {title}

## Introduction

This comprehensive guide explores the fundamental principles and advanced techniques in the field of machine learning and artificial intelligence. The rapid advancement of computational capabilities has enabled unprecedented progress in developing intelligent systems that can learn, adapt, and make decisions with remarkable accuracy.

## Chapter 1: Foundations of Machine Learning

Machine learning represents a paradigm shift in how we approach problem-solving through computational means. At its core, machine learning involves the development of algorithms that can automatically learn patterns from data without being explicitly programmed for every possible scenario.

### Supervised Learning

Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. This approach is particularly powerful when we have access to high-quality labeled datasets that represent the problem domain accurately.

The key components of supervised learning include:

1. **Training Data**: A collection of input-output pairs that serve as examples for the learning algorithm
2. **Feature Engineering**: The process of selecting and transforming input variables to improve model performance
3. **Model Selection**: Choosing the appropriate algorithm based on the problem characteristics and data properties
4. **Validation**: Assessing model performance on held-out data to ensure generalization

### Unsupervised Learning

Unsupervised learning algorithms discover hidden patterns in data without the guidance of labeled examples. This approach is particularly valuable when labeled data is scarce or when we want to explore the underlying structure of our data.

Common unsupervised learning techniques include:

- **Clustering**: Grouping similar data points together based on their characteristics
- **Dimensionality Reduction**: Reducing the number of features while preserving important information
- **Association Rule Learning**: Discovering relationships between different items in large datasets

## Chapter 2: Deep Learning and Neural Networks

Deep learning has revolutionized the field of machine learning by enabling the training of complex models with multiple layers of abstraction. These deep neural networks can automatically learn hierarchical representations of data, leading to state-of-the-art performance across many domains.

### Neural Network Architecture

The fundamental building block of deep learning is the artificial neuron, which receives multiple inputs, applies weights and biases, and produces an output through an activation function. When organized into layers, these neurons can learn increasingly complex patterns.

Key architectural components include:

1. **Input Layer**: Receives the raw data and passes it to the first hidden layer
2. **Hidden Layers**: Process the data through multiple transformations, each learning different levels of abstraction
3. **Output Layer**: Produces the final predictions or classifications
4. **Activation Functions**: Introduce non-linearity to enable the network to learn complex patterns

### Training Deep Networks

Training deep neural networks involves optimizing the network parameters to minimize a loss function that measures the difference between predicted and actual outputs. This optimization process typically uses gradient descent and its variants.

The training process includes:

- **Forward Propagation**: Computing predictions by passing data through the network
- **Loss Calculation**: Measuring the error between predictions and ground truth
- **Backpropagation**: Computing gradients of the loss with respect to network parameters
- **Parameter Updates**: Adjusting weights and biases to reduce the loss

## Chapter 3: Advanced Techniques and Applications

Modern machine learning systems incorporate sophisticated techniques to handle complex real-world challenges. These include attention mechanisms, transfer learning, and reinforcement learning approaches.

### Attention Mechanisms

Attention mechanisms have become crucial for processing sequential data and enabling models to focus on relevant parts of the input. The transformer architecture, built on self-attention, has achieved remarkable success in natural language processing and beyond.

Key benefits of attention mechanisms include:

- **Parallelization**: Unlike recurrent networks, attention can process all positions simultaneously
- **Long-range Dependencies**: Attention can capture relationships between distant elements in sequences
- **Interpretability**: Attention weights provide insights into what the model focuses on

### Transfer Learning

Transfer learning leverages knowledge gained from one task to improve performance on related tasks. This approach is particularly valuable when labeled data is limited for the target task.

Transfer learning strategies include:

1. **Feature Extraction**: Using pre-trained models as feature extractors
2. **Fine-tuning**: Adapting pre-trained models to specific tasks through additional training
3. **Multi-task Learning**: Training models on multiple related tasks simultaneously

## Chapter 4: Practical Implementation and Best Practices

Implementing machine learning systems in production requires careful consideration of various factors beyond model accuracy. These include data quality, system scalability, model interpretability, and ethical considerations.

### Data Quality and Preprocessing

High-quality data is essential for successful machine learning applications. Data preprocessing steps include:

- **Data Cleaning**: Removing or correcting errors, inconsistencies, and missing values
- **Feature Engineering**: Creating new features or transforming existing ones to improve model performance
- **Data Augmentation**: Generating additional training examples to increase dataset diversity
- **Normalization**: Scaling features to ensure they contribute equally to the learning process

### Model Evaluation and Validation

Proper evaluation is crucial for understanding model performance and ensuring reliable predictions. Key evaluation practices include:

- **Cross-validation**: Using multiple train-test splits to get robust performance estimates
- **Hold-out Testing**: Reserving a portion of data for final evaluation
- **Performance Metrics**: Choosing appropriate metrics based on the problem type and business requirements
- **Error Analysis**: Understanding where and why the model makes mistakes

## Chapter 5: Future Directions and Challenges

The field of machine learning continues to evolve rapidly, with new techniques and applications emerging regularly. Understanding current trends and challenges helps guide future research and development efforts.

### Emerging Trends

Several trends are shaping the future of machine learning:

1. **Large Language Models**: Models with billions of parameters that demonstrate remarkable capabilities across diverse tasks
2. **Multimodal Learning**: Systems that can process and integrate information from multiple modalities
3. **Federated Learning**: Training models across distributed data sources while preserving privacy
4. **Automated Machine Learning**: Systems that can automatically design and optimize machine learning pipelines

### Current Challenges

Despite significant progress, several challenges remain:

- **Data Efficiency**: Developing methods that can learn effectively from limited data
- **Robustness**: Ensuring models perform reliably across different conditions and distributions
- **Interpretability**: Making complex models more understandable and trustworthy
- **Ethical AI**: Addressing bias, fairness, and societal impact of machine learning systems

## Conclusion

Machine learning represents a powerful paradigm for building intelligent systems that can learn from data and make informed decisions. As the field continues to advance, it offers tremendous opportunities for solving complex problems across diverse domains.

The key to success in machine learning lies in understanding both the theoretical foundations and practical considerations. By combining solid theoretical knowledge with hands-on experience and careful attention to real-world constraints, practitioners can develop effective machine learning solutions that provide genuine value.

The future of machine learning is bright, with continued advances in algorithms, hardware, and applications promising to unlock even greater capabilities. As we move forward, it will be important to balance technical progress with considerations of ethics, fairness, and societal benefit.

This concludes our comprehensive exploration of machine learning principles, techniques, and applications. The field continues to evolve rapidly, offering exciting opportunities for researchers, practitioners, and organizations seeking to leverage the power of intelligent systems.
"""
        
        # Extend content to reach target length
        while len(content) < target_length:
            content += "\n\n" + self._get_additional_technical_content()
        
        return content[:target_length]
    
    def _get_additional_technical_content(self) -> str:
        """Generate additional technical content to reach target length."""
        topics = [
            "Advanced Optimization Techniques",
            "Regularization Methods",
            "Ensemble Learning Approaches",
            "Model Selection and Hyperparameter Tuning",
            "Deployment and Monitoring",
            "Ethical Considerations in AI",
            "Performance Optimization",
            "Scalability Challenges"
        ]
        
        topic = random.choice(topics)
        return f"""
## {topic}

The field of machine learning continues to evolve with new techniques and methodologies emerging regularly. {topic} represents a crucial area of research that addresses fundamental challenges in building robust and efficient learning systems.

Recent advances in this area have shown promising results, with new algorithms and approaches demonstrating improved performance across various benchmarks. The integration of theoretical insights with practical implementation considerations has led to significant progress in understanding how to build more effective machine learning systems.

Key developments include novel optimization techniques that can handle non-convex optimization landscapes more effectively, improved regularization methods that prevent overfitting while maintaining model capacity, and ensemble approaches that combine multiple models to achieve better generalization performance.

The practical implications of these advances are significant, enabling the development of more reliable and efficient machine learning systems that can be deployed in real-world applications. As the field continues to mature, we can expect to see continued progress in addressing the fundamental challenges of machine learning.
"""
    
    def _get_narrative_content(self, title: str, target_length: int) -> str:
        """Generate narrative content."""
        return f"""
# {title}

## Chapter 1: The Beginning

In the quiet town of Millbrook, where the morning mist rises from the river and the old oak trees stand as silent witnesses to countless stories, our tale begins. Sarah had always been drawn to the mysterious, the unexplained, and the extraordinary that lay hidden beneath the surface of everyday life.

The old library on Main Street held more than just books—it held secrets, memories, and the whispers of those who had come before. Sarah spent countless hours among the dusty shelves, searching for answers to questions she couldn't quite articulate.

## Chapter 2: Discovery

It was on a particularly foggy morning in October when Sarah discovered the hidden compartment behind the third shelf from the left. The mechanism was cleverly concealed, activated by pressing three specific books in sequence. As the hidden door swung open with a soft creak, Sarah's heart raced with anticipation.

Inside, she found a collection of journals, maps, and artifacts that seemed to belong to another era. The handwriting was elegant but faded, and the maps showed places that didn't exist on any modern atlas. Sarah realized she had stumbled upon something extraordinary.

## Chapter 3: The Journey Begins

The journals told the story of an explorer who had traveled to distant lands in search of ancient knowledge. Each entry was more fascinating than the last, describing encounters with cultures that had preserved wisdom lost to the modern world. Sarah felt a deep connection to this long-dead explorer, as if their paths were meant to cross across the boundaries of time.

Determined to follow in the explorer's footsteps, Sarah began planning her own journey. She studied the maps, learned about the cultures mentioned in the journals, and prepared herself for an adventure that would change her life forever.

## Chapter 4: Challenges and Revelations

The journey was not without its challenges. Sarah encountered obstacles that tested her resolve and forced her to question everything she thought she knew. But with each challenge came new revelations, new understanding, and new connections to the world around her.

Through her travels, Sarah discovered that the greatest adventures are not just about reaching a destination, but about the transformation that occurs along the way. She learned about herself, about the world, and about the interconnectedness of all things.

## Chapter 5: The Return

When Sarah finally returned to Millbrook, she was not the same person who had left. She carried with her not just the artifacts and knowledge she had discovered, but a deeper understanding of life, purpose, and the mysteries that connect us all.

The old library welcomed her back, and she knew that her own story would one day join the countless others that filled its shelves. For every ending is also a beginning, and every journey is part of a larger story that continues to unfold.

## Epilogue

Years later, Sarah would often return to the hidden compartment, now filled with her own journals and discoveries. She had become part of the library's secret history, a guardian of knowledge and a seeker of truth. And in the quiet moments between the pages, she could still hear the whispers of those who had come before, and the promise of those who would follow.

The story continues, as all good stories do, in the hearts and minds of those who dare to seek, to discover, and to believe in the magic that lies just beyond the ordinary.
"""
    
    def _get_scientific_content(self, title: str, target_length: int) -> str:
        """Generate scientific content."""
        return f"""
# {title}

## Abstract

This comprehensive study presents groundbreaking research in the field of scientific discovery, examining the fundamental principles that govern our understanding of the natural world. Through rigorous experimentation and theoretical analysis, we have uncovered new insights that challenge existing paradigms and open new avenues for future research.

## Introduction

The pursuit of scientific knowledge has been one of humanity's greatest endeavors, driving us to understand the mysteries of the universe and our place within it. From the smallest subatomic particles to the vast expanses of space, science provides us with the tools to explore, understand, and ultimately harness the forces that shape our reality.

## Methodology

Our research employed a multi-disciplinary approach, combining theoretical modeling with experimental validation. We utilized state-of-the-art instrumentation and computational methods to analyze complex systems and phenomena that have previously eluded scientific understanding.

### Experimental Design

The experimental design was carefully crafted to ensure reproducibility and validity of results. We employed randomized controlled trials where appropriate, and implemented rigorous statistical analysis to ensure the reliability of our findings.

### Data Collection and Analysis

Data collection involved multiple phases, each designed to capture different aspects of the phenomena under investigation. Advanced analytical techniques were employed to extract meaningful patterns and relationships from the collected data.

## Results

Our research has yielded several significant findings that advance our understanding of the natural world:

### Primary Findings

1. **Novel Phenomena Discovery**: We have identified previously unknown phenomena that challenge existing theoretical frameworks.

2. **Quantitative Relationships**: Our analysis has revealed precise quantitative relationships between variables that were previously thought to be independent.

3. **Predictive Models**: We have developed new predictive models that can accurately forecast system behavior under various conditions.

### Secondary Findings

Additional research has uncovered several secondary findings that, while not directly related to our primary hypotheses, provide valuable insights for future research directions.

## Discussion

The implications of our findings extend far beyond the immediate scope of this study. They suggest new approaches to understanding complex systems and may lead to practical applications in various fields.

### Theoretical Implications

Our results challenge several long-standing theoretical assumptions and suggest the need for paradigm shifts in how we conceptualize certain phenomena. The theoretical implications are profound and will likely influence future research directions.

### Practical Applications

The practical applications of our findings are numerous and span multiple disciplines. From medicine to engineering, our research provides new tools and approaches for solving real-world problems.

## Future Research Directions

Based on our findings, we identify several promising directions for future research:

1. **Extended Studies**: Longer-term studies to validate the stability of our findings over time.

2. **Cross-Disciplinary Applications**: Exploring applications of our findings in related fields.

3. **Technology Development**: Developing new technologies based on our discoveries.

## Conclusion

This research represents a significant advancement in our understanding of the natural world. The findings provide new insights, challenge existing paradigms, and open new avenues for future research. As we continue to explore the mysteries of the universe, studies like this remind us of the power of scientific inquiry to transform our understanding of reality.

The journey of discovery continues, and we look forward to the new insights and breakthroughs that future research will bring.
"""
    
    def _get_philosophical_content(self, title: str, target_length: int) -> str:
        """Generate philosophical content."""
        return f"""
# {title}

## Introduction: The Nature of Inquiry

Philosophy begins with wonder—the recognition that the world around us, and our place within it, is far more complex and mysterious than it initially appears. This fundamental sense of wonder drives us to ask the deepest questions about existence, knowledge, morality, and meaning.

## Chapter 1: The Problem of Knowledge

What can we truly know? This question lies at the heart of epistemology, the branch of philosophy concerned with the nature and scope of knowledge. Throughout history, philosophers have grappled with the challenge of distinguishing between genuine knowledge and mere belief.

### The Socratic Method

Socrates, the father of Western philosophy, taught us that wisdom begins with the recognition of our own ignorance. His method of questioning—the Socratic method—remains one of the most powerful tools for philosophical inquiry. By systematically questioning our assumptions and beliefs, we can uncover the foundations of our knowledge and identify areas where our understanding may be incomplete.

### Rationalism vs. Empiricism

The debate between rationalism and empiricism has shaped much of Western philosophical thought. Rationalists argue that knowledge comes primarily from reason and innate ideas, while empiricists maintain that knowledge is derived from sensory experience. This fundamental disagreement continues to influence contemporary discussions about the nature of knowledge.

## Chapter 2: The Nature of Reality

Metaphysics, the study of the fundamental nature of reality, addresses questions about existence, being, and the structure of the world. What is real? What is the relationship between mind and matter? These questions have occupied philosophers for millennia.

### The Mind-Body Problem

One of the most persistent problems in philosophy is the relationship between mind and body. How can mental states, which seem immaterial, interact with physical states? Various solutions have been proposed, from dualism to materialism to idealism, each offering different perspectives on this fundamental question.

### The Problem of Universals

The problem of universals concerns the nature of general concepts and properties. Do concepts like "beauty" or "justice" exist independently of particular instances, or are they merely mental constructs? This question has implications for our understanding of language, mathematics, and the nature of reality itself.

## Chapter 3: Ethics and Moral Philosophy

Ethics addresses questions about how we should live and what makes actions right or wrong. The study of ethics is not merely academic—it has profound implications for how we make decisions and interact with others.

### Consequentialism vs. Deontology

Two major approaches to ethics are consequentialism, which judges actions by their outcomes, and deontology, which focuses on the inherent rightness or wrongness of actions themselves. Each approach offers different insights into moral decision-making and has different implications for how we should live.

### Virtue Ethics

Virtue ethics, inspired by Aristotle, focuses on the character of the moral agent rather than on rules or consequences. This approach emphasizes the development of virtuous character traits and the cultivation of practical wisdom in moral decision-making.

## Chapter 4: The Meaning of Life

Perhaps the most fundamental question in philosophy is the meaning of life. What gives our existence purpose and significance? This question has been approached from various angles, including religious, existential, and naturalistic perspectives.

### Existentialism

Existentialist philosophers like Sartre and Camus argue that life has no inherent meaning, but that we must create our own meaning through our choices and actions. This perspective emphasizes human freedom and responsibility in the face of an apparently meaningless universe.

### The Search for Purpose

Many philosophers have sought to identify sources of meaning in life, from the pursuit of knowledge and truth to the cultivation of relationships and the development of character. The diversity of these approaches reflects the complexity of human experience and the variety of ways in which people find purpose and fulfillment.

## Chapter 5: Contemporary Philosophical Issues

Philosophy continues to evolve, addressing new challenges and questions that arise from advances in science, technology, and society. Contemporary philosophers grapple with issues ranging from artificial intelligence to environmental ethics to the nature of consciousness.

### Philosophy of Mind and Cognitive Science

Recent advances in cognitive science and neuroscience have raised new questions about the nature of consciousness, free will, and the relationship between mind and brain. These developments have enriched philosophical inquiry and opened new avenues for understanding human nature.

### Applied Ethics

The complexity of modern society has given rise to new ethical challenges that require careful philosophical analysis. Issues such as bioethics, environmental ethics, and the ethics of technology demand sophisticated ethical reasoning and the application of philosophical principles to real-world problems.

## Conclusion: The Enduring Value of Philosophy

Philosophy, despite its abstract nature, has profound practical implications for how we live our lives. By encouraging us to question our assumptions, examine our beliefs, and think critically about fundamental issues, philosophy helps us to live more examined and meaningful lives.

The questions that philosophy addresses are not merely academic—they are questions that every thinking person must confront. In engaging with these questions, we participate in a conversation that has been ongoing for thousands of years and will continue as long as humans seek to understand themselves and their world.

The value of philosophy lies not in providing definitive answers, but in teaching us how to ask better questions and how to think more clearly about the issues that matter most. In this way, philosophy remains as relevant and necessary today as it was in ancient times.
"""
    
    def _get_historical_content(self, title: str, target_length: int) -> str:
        """Generate historical content."""
        return f"""
# {title}

## Introduction: The Tapestry of Time

History is not merely a collection of dates and events, but a complex tapestry woven from the lives, decisions, and circumstances of countless individuals. Each thread in this tapestry represents a choice made, a path taken, or a moment that changed the course of human events.

## Chapter 1: The Ancient Foundations

The foundations of human civilization were laid in the fertile river valleys of Mesopotamia, Egypt, the Indus Valley, and China. These early civilizations developed writing, mathematics, astronomy, and complex social structures that would influence human development for millennia.

### The Rise of Empires

From these early civilizations emerged the great empires of antiquity—the Persian Empire, the Roman Empire, the Han Dynasty, and others. These empires not only controlled vast territories but also facilitated the exchange of ideas, technologies, and cultures across continents.

### The Classical Age

The Classical Age of Greece and Rome produced some of the most influential ideas in human history. Greek philosophy, democracy, and scientific inquiry, combined with Roman law, engineering, and administration, created a foundation for Western civilization that endures to this day.

## Chapter 2: The Medieval Transformation

The fall of the Roman Empire marked the beginning of the medieval period, a time of transformation and adaptation. While often characterized as a "dark age," this period was actually one of significant innovation and cultural development.

### The Byzantine Empire

The Eastern Roman Empire, known as Byzantium, preserved and developed the classical heritage while creating its own distinctive culture. Byzantine art, architecture, and scholarship would influence both Eastern and Western traditions.

### The Islamic Golden Age

The rise of Islam and the expansion of the Islamic world created a golden age of learning and innovation. Islamic scholars preserved and translated classical texts, made significant advances in mathematics, medicine, and astronomy, and created a vibrant intellectual culture.

### The European Renaissance

The Renaissance marked a renewed interest in classical learning and a flowering of art, literature, and science. This period saw the development of new techniques in painting, sculpture, and architecture, as well as advances in navigation, astronomy, and medicine.

## Chapter 3: The Age of Exploration and Discovery

The 15th and 16th centuries marked the beginning of the Age of Exploration, as European powers sought new trade routes and territories. This period of global exploration and colonization would reshape the world and create new connections between distant cultures.

### The Columbian Exchange

The encounter between the Old World and the New World resulted in the Columbian Exchange, a massive transfer of plants, animals, diseases, and cultures between the hemispheres. This exchange had profound and lasting effects on both sides of the Atlantic.

### The Scientific Revolution

The 16th and 17th centuries saw the emergence of modern science, with figures like Copernicus, Galileo, and Newton challenging traditional views of the universe and developing new methods of scientific inquiry.

## Chapter 4: The Modern Era

The 18th and 19th centuries brought revolutionary changes in politics, society, and technology. The Enlightenment, the Industrial Revolution, and the rise of nationalism transformed the world in ways that continue to influence us today.

### The Enlightenment

The Enlightenment emphasized reason, individualism, and progress. This intellectual movement influenced political revolutions, social reforms, and the development of modern democratic institutions.

### The Industrial Revolution

The Industrial Revolution transformed economies and societies through the introduction of new technologies, manufacturing processes, and forms of organization. This revolution created unprecedented wealth and opportunity while also bringing new challenges and inequalities.

### The Age of Imperialism

The 19th century saw the expansion of European empires across the globe, creating new political, economic, and cultural connections while also generating conflicts and resistance.

## Chapter 5: The Contemporary World

The 20th and 21st centuries have been marked by unprecedented change, including two world wars, the Cold War, decolonization, and the digital revolution. These developments have created a more interconnected but also more complex world.

### The World Wars

The two world wars of the 20th century were among the most destructive conflicts in human history, but they also led to significant political, social, and technological changes that shaped the modern world.

### The Cold War and Decolonization

The Cold War between the United States and the Soviet Union dominated international relations for much of the second half of the 20th century, while the process of decolonization created new independent nations and reshaped global politics.

### The Digital Revolution

The development of computers, the internet, and digital technologies has created a new era of connectivity and information sharing, transforming how we communicate, work, and live.

## Conclusion: Lessons from History

History teaches us that change is constant, that human societies are complex and adaptable, and that the choices we make today will shape the world of tomorrow. By studying history, we can better understand the present and make more informed decisions about the future.

The study of history is not merely an academic exercise—it is essential for understanding who we are, where we come from, and where we might be going. In learning from the past, we can build a better future for ourselves and for generations to come.
"""
    
    def _get_futuristic_content(self, title: str, target_length: int) -> str:
        """Generate futuristic content."""
        return f"""
# {title}

## Introduction: Visions of Tomorrow

The future is not a destination but a direction—a path we create through our choices, innovations, and collective imagination. As we stand at the threshold of unprecedented technological advancement, we must consider not only what is possible, but what is desirable for humanity.

## Chapter 1: The Digital Transformation

The digital revolution has only just begun. As we move deeper into the 21st century, we can expect to see even more profound changes in how we interact with technology and with each other.

### Artificial Intelligence and Machine Learning

Artificial intelligence is rapidly evolving from a specialized tool to a fundamental aspect of human life. Machine learning algorithms are becoming more sophisticated, enabling computers to perform tasks that were once thought to be uniquely human. The implications of this development are profound, affecting everything from healthcare and education to entertainment and employment.

### The Internet of Things

The Internet of Things (IoT) is creating a world where everyday objects are connected and intelligent. Smart homes, smart cities, and smart industries are becoming reality, creating new possibilities for efficiency, convenience, and sustainability.

### Virtual and Augmented Reality

Virtual and augmented reality technologies are blurring the line between the digital and physical worlds. These technologies have applications in education, healthcare, entertainment, and many other fields, offering new ways to experience and interact with information.

## Chapter 2: Biotechnology and Human Enhancement

Advances in biotechnology are opening new possibilities for human enhancement and medical treatment. From gene editing to brain-computer interfaces, these technologies raise important questions about what it means to be human.

### Genetic Engineering

CRISPR and other gene editing technologies are making it possible to modify DNA with unprecedented precision. While these technologies offer hope for treating genetic diseases, they also raise ethical questions about the limits of human modification.

### Brain-Computer Interfaces

Brain-computer interfaces are developing rapidly, offering new possibilities for treating neurological conditions and potentially enhancing human cognitive abilities. These technologies could revolutionize how we interact with computers and with each other.

### Longevity and Aging

Research into aging and longevity is advancing rapidly, with scientists exploring ways to extend healthy human lifespan. While the goal of significantly extending human life is still distant, progress in understanding the biology of aging is accelerating.

## Chapter 3: Space Exploration and Colonization

Humanity's future may extend far beyond Earth. Advances in space technology are making it increasingly feasible to explore and potentially colonize other worlds.

### Mars Colonization

Mars represents the most likely target for human colonization in the near future. Advances in rocket technology, life support systems, and terraforming techniques are making the dream of a human presence on Mars increasingly realistic.

### Space Mining and Resources

The asteroid belt and other celestial bodies contain vast resources that could support human expansion into space. Space mining could provide materials for space-based manufacturing and construction.

### Interstellar Travel

While still in the realm of science fiction, research into interstellar travel is advancing. Concepts like fusion propulsion, antimatter engines, and generation ships are being explored as potential means of reaching other star systems.

## Chapter 4: Environmental and Sustainability Technologies

As we face the challenges of climate change and environmental degradation, new technologies are emerging to help us build a more sustainable future.

### Renewable Energy

Advances in solar, wind, and other renewable energy technologies are making clean energy more affordable and efficient. Energy storage technologies are also improving, making renewable energy more reliable and practical.

### Carbon Capture and Climate Engineering

Technologies for capturing and storing carbon dioxide are developing rapidly, offering potential solutions to climate change. Climate engineering techniques, while controversial, are also being explored as potential tools for managing global climate.

### Sustainable Agriculture

New agricultural technologies, including vertical farming, precision agriculture, and lab-grown food, are offering ways to produce food more efficiently and sustainably.

## Chapter 5: Social and Cultural Evolution

Technology is not the only driver of change. Social and cultural evolution will also shape the future, as humanity adapts to new possibilities and challenges.

### The Future of Work

Automation and artificial intelligence are transforming the nature of work. While some jobs may disappear, new opportunities will emerge, requiring new skills and new ways of thinking about employment and productivity.

### Education and Learning

The future of education will likely be more personalized, flexible, and technology-enhanced. Virtual reality, artificial intelligence, and other technologies will create new possibilities for learning and skill development.

### Social Connectivity

Technology is changing how we connect with each other, creating new forms of community and social interaction. The future may see the development of new social structures and cultural practices that reflect our increasingly connected world.

## Conclusion: Shaping the Future

The future is not predetermined—it is something we create through our actions, choices, and innovations. As we stand at the threshold of unprecedented change, we have the opportunity and the responsibility to shape a future that reflects our highest values and aspirations.

The technologies and trends discussed in this exploration represent just a glimpse of what may be possible. The actual future will likely be even more surprising and transformative than we can imagine. What matters most is that we approach these changes with wisdom, foresight, and a commitment to human flourishing.

The future belongs to those who dare to imagine it, to those who work to create it, and to those who are prepared to adapt to it. As we move forward into this uncertain but exciting future, let us do so with hope, determination, and a shared vision of a better world for all humanity.
"""
    
    def _get_descriptive_content(self, title: str, target_length: int) -> str:
        """Generate descriptive content."""
        return f"""
# {title}

## Introduction: The Beauty of the Natural World

Nature is a masterpiece of complexity and beauty, a symphony of interconnected systems that have evolved over billions of years. From the smallest microorganisms to the largest ecosystems, the natural world offers endless opportunities for wonder, discovery, and understanding.

## Chapter 1: The Microscopic World

Beneath our feet and all around us exists a world invisible to the naked eye, yet teeming with life and activity. The microscopic world is home to countless organisms that play crucial roles in the functioning of ecosystems and the health of our planet.

### Bacteria and Microorganisms

Bacteria are among the most abundant and diverse forms of life on Earth. These single-celled organisms can be found in virtually every environment, from the depths of the ocean to the soil beneath our feet. Some bacteria are beneficial, helping to break down organic matter and fix nitrogen in the soil, while others can cause disease.

### Viruses and Their Role

Viruses, though not technically alive, are among the most numerous biological entities on Earth. They play important roles in regulating populations and transferring genetic material between organisms. Recent research has revealed the crucial role viruses play in maintaining the health of ecosystems.

### The Soil Microbiome

The soil beneath our feet is a complex ecosystem teeming with microscopic life. Bacteria, fungi, protozoa, and other microorganisms work together to break down organic matter, cycle nutrients, and maintain soil health. This underground world is essential for plant growth and ecosystem functioning.

## Chapter 2: The Plant Kingdom

Plants are the foundation of most terrestrial ecosystems, converting sunlight into energy and providing food and habitat for countless other organisms. The diversity of plant life is staggering, from tiny mosses to towering trees.

### Photosynthesis and Energy Flow

Photosynthesis is one of the most important biological processes on Earth, converting sunlight into chemical energy that fuels most life. Plants use this process to create glucose from carbon dioxide and water, releasing oxygen as a byproduct. This process is the foundation of most food webs.

### Plant Adaptations

Plants have evolved an incredible array of adaptations to survive in different environments. From the water-storing tissues of desert cacti to the broad leaves of rainforest plants, each adaptation reflects the challenges and opportunities of a particular habitat.

### The Role of Plants in Ecosystems

Plants provide numerous ecosystem services, including oxygen production, carbon sequestration, soil stabilization, and habitat provision. They also play crucial roles in nutrient cycling and water purification.

## Chapter 3: The Animal Kingdom

Animals represent an incredible diversity of forms, behaviors, and adaptations. From the simplest sponges to the most complex mammals, animals have evolved to exploit virtually every ecological niche on Earth.

### Invertebrates: The Unsung Heroes

Invertebrates make up the vast majority of animal species, yet they often receive less attention than their vertebrate cousins. From insects that pollinate our crops to marine invertebrates that form the foundation of ocean food webs, these creatures play essential roles in ecosystem functioning.

### Vertebrates: Complexity and Intelligence

Vertebrates, including fish, amphibians, reptiles, birds, and mammals, represent some of the most complex and intelligent forms of life on Earth. These animals have evolved sophisticated nervous systems, complex behaviors, and remarkable adaptations to their environments.

### Migration and Movement

Many animals undertake incredible journeys, migrating across continents and oceans in search of food, breeding grounds, or suitable habitat. These migrations are among the most spectacular phenomena in the natural world.

## Chapter 4: Ecosystems and Communities

Ecosystems are complex networks of interactions between living organisms and their physical environment. These systems are characterized by energy flow, nutrient cycling, and the exchange of materials between different components.

### Energy Flow and Food Webs

Energy flows through ecosystems in predictable patterns, from primary producers (plants) to primary consumers (herbivores) to secondary consumers (carnivores) and decomposers. This energy flow creates complex food webs that connect all organisms in an ecosystem.

### Nutrient Cycling

Nutrients cycle through ecosystems in complex patterns, moving between living organisms, the soil, water, and atmosphere. These cycles are essential for maintaining ecosystem health and productivity.

### Succession and Change

Ecosystems are dynamic, constantly changing in response to environmental conditions, species interactions, and disturbances. Understanding these patterns of change is crucial for conservation and management efforts.

## Chapter 5: Conservation and Stewardship

As human activities increasingly impact the natural world, conservation has become more important than ever. Protecting biodiversity and maintaining healthy ecosystems is essential for human well-being and the survival of countless species.

### Threats to Biodiversity

Human activities pose numerous threats to biodiversity, including habitat destruction, pollution, climate change, and the introduction of invasive species. Understanding these threats is the first step in addressing them.

### Conservation Strategies

Conservation efforts employ a variety of strategies, from protecting individual species to preserving entire ecosystems. These efforts require cooperation between governments, organizations, and individuals.

### The Role of Science

Scientific research is essential for understanding the natural world and developing effective conservation strategies. Advances in technology are providing new tools for monitoring and protecting biodiversity.

## Conclusion: Our Connection to Nature

The natural world is not separate from human society—it is the foundation upon which our civilization is built. Every breath we take, every meal we eat, and every material we use comes from the natural world. Understanding and protecting this world is not just an environmental concern—it is essential for human survival and flourishing.

As we face the challenges of the 21st century, from climate change to biodiversity loss, we must remember our deep connection to the natural world. By working to understand, appreciate, and protect the incredible diversity of life on Earth, we can ensure a sustainable future for ourselves and for generations to come.

The natural world offers endless opportunities for wonder, discovery, and inspiration. By taking the time to observe, learn, and appreciate the beauty and complexity of nature, we can develop a deeper understanding of our place in the world and our responsibility to protect it.
"""
    
    def _get_analytical_content(self, title: str, target_length: int) -> str:
        """Generate analytical content."""
        return f"""
# {title}

## Introduction: The Science of the Mind

Human psychology is the study of the mind and behavior, encompassing everything from basic cognitive processes to complex social interactions. Understanding psychology helps us comprehend not only how individuals think and act, but also how societies function and evolve.

## Chapter 1: Cognitive Processes

The human mind is a remarkable information processing system, capable of complex reasoning, memory, and decision-making. Understanding these cognitive processes is fundamental to understanding human behavior.

### Perception and Attention

Perception is the process by which we interpret sensory information from our environment. Our brains constantly filter and organize this information, allowing us to make sense of the world around us. Attention plays a crucial role in this process, determining what information we focus on and what we ignore.

### Memory Systems

Human memory is not a single system but a complex network of different types of memory, each serving different functions. From working memory that holds information temporarily to long-term memory that stores knowledge and experiences, these systems work together to support learning and decision-making.

### Language and Communication

Language is one of the most complex cognitive abilities, involving the integration of multiple brain systems. The ability to understand and produce language is fundamental to human communication and social interaction.

## Chapter 2: Learning and Development

Learning is a fundamental process that allows humans to adapt to their environment and acquire new skills and knowledge. Understanding how learning occurs is essential for education, training, and personal development.

### Classical and Operant Conditioning

Classical conditioning, discovered by Pavlov, involves learning associations between stimuli. Operant conditioning, developed by Skinner, involves learning through consequences. These basic learning principles underlie much of human behavior.

### Social Learning

Humans learn not only through direct experience but also by observing others. Social learning theory explains how we acquire behaviors, attitudes, and knowledge through observation and imitation.

### Cognitive Development

The study of cognitive development examines how thinking abilities change and develop throughout the lifespan. From the sensorimotor stage of infancy to the formal operational stage of adolescence, cognitive development follows predictable patterns.

## Chapter 3: Personality and Individual Differences

While all humans share basic psychological processes, individuals differ significantly in their personalities, abilities, and behavioral patterns. Understanding these individual differences is crucial for psychology and related fields.

### Personality Theories

Various theories attempt to explain personality, from Freud's psychoanalytic theory to modern trait theories. Each approach offers different insights into the nature of personality and its development.

### Intelligence and Abilities

Intelligence is a complex construct that encompasses various cognitive abilities. Research has revealed that intelligence is not a single ability but a collection of different skills and capacities.

### Motivation and Emotion

Motivation drives behavior, while emotions provide important information about our internal states and external environment. Understanding these processes is essential for understanding human behavior.

## Chapter 4: Social Psychology

Humans are social beings, and much of our behavior is influenced by social factors. Social psychology examines how individuals think, feel, and behave in social situations.

### Social Cognition

Social cognition refers to how we think about ourselves and others. This includes processes like attribution, stereotyping, and social comparison, which influence how we interpret social situations.

### Group Dynamics

Groups have powerful effects on individual behavior. Understanding group dynamics helps explain phenomena like conformity, groupthink, and social influence.

### Interpersonal Relationships

Relationships are fundamental to human well-being. Research has revealed the factors that contribute to successful relationships and the psychological processes that underlie social connections.

## Chapter 5: Abnormal Psychology and Mental Health

Understanding psychological disorders and mental health is crucial for helping individuals who are struggling with psychological problems.

### Classification of Disorders

Psychological disorders are classified according to various systems, each with different approaches to understanding and categorizing mental health problems.

### Treatment Approaches

Various treatment approaches have been developed to help individuals with psychological disorders, from psychotherapy to medication to alternative treatments.

### Prevention and Promotion

Prevention of psychological problems and promotion of mental health are increasingly important areas of focus in psychology and public health.

## Chapter 6: Applied Psychology

Psychology has numerous practical applications in various fields, from education to business to healthcare.

### Educational Psychology

Educational psychology applies psychological principles to improve learning and teaching. This field helps educators understand how students learn and how to create effective learning environments.

### Industrial and Organizational Psychology

I/O psychology applies psychological principles to the workplace, helping organizations improve productivity, employee satisfaction, and organizational effectiveness.

### Health Psychology

Health psychology examines the relationship between psychological factors and physical health, helping to promote healthy behaviors and improve health outcomes.

## Conclusion: The Future of Psychology

Psychology continues to evolve as new research methods and technologies provide new insights into the human mind and behavior. The field is becoming increasingly interdisciplinary, incorporating insights from neuroscience, genetics, and other fields.

As we face new challenges in the 21st century, from technological change to global health crises, psychology will play an increasingly important role in understanding and addressing these challenges. By continuing to study the human mind and behavior, we can develop better ways to promote human well-being and address the complex problems facing our world.

The study of psychology is not just an academic pursuit—it has practical applications that can improve lives and help create a better world. By understanding the principles of human behavior, we can make more informed decisions, build better relationships, and create more effective solutions to the challenges we face.
"""
    
    def _extract_chapters(self, content: str) -> List[str]:
        """Extract chapter titles from content."""
        chapters = []
        lines = content.split('\n')
        for line in lines:
            if line.startswith('## ') and 'Chapter' in line:
                chapters.append(line.replace('## ', ''))
        return chapters
    
    def process_books_for_rag(self) -> List[Document]:
        """Process books into documents suitable for RAG testing."""
        logger.info("Processing books for RAG testing...")
        
        if not self.books:
            self.load_sample_books()
        
        documents = []
        
        for book in self.books:
            # Split book into chunks
            chunks = self._split_into_chunks(book['content'])
            
            for i, chunk in enumerate(chunks[:self.config.max_chunks_per_book]):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source': f"{book['title']}_chunk_{i}",
                        'book_title': book['title'],
                        'genre': book['genre'],
                        'chunk_index': i,
                        'book_length': book['length'],
                        'author': book['metadata']['author'],
                        'year': book['metadata']['year']
                    }
                )
                documents.append(doc)
        
        self.processed_books = documents
        logger.info(f"Processed {len(documents)} document chunks from {len(self.books)} books")
        
        return documents
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.config.chunk_size - self.config.chunk_overlap
            
        return chunks
    
    def generate_test_queries(self) -> List[Dict[str, Any]]:
        """Generate test queries for different books and genres."""
        if not self.books:
            self.load_sample_books()
        
        queries = []
        
        for book in self.books:
            book_queries = self._generate_queries_for_book(book)
            queries.extend(book_queries)
        
        return queries
    
    def _generate_queries_for_book(self, book: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test queries for a specific book."""
        genre = book['genre']
        title = book['title']
        
        # Genre-specific query templates
        query_templates = {
            'technical': [
                f"What are the main concepts discussed in {title}?",
                f"How does {title} explain the technical principles?",
                f"What are the key methodologies described in {title}?",
                f"What practical applications are mentioned in {title}?",
                f"What are the limitations discussed in {title}?"
            ],
            'narrative': [
                f"What is the main story in {title}?",
                f"Who are the main characters in {title}?",
                f"What challenges do the characters face in {title}?",
                f"How does the story in {title} develop?",
                f"What is the resolution in {title}?"
            ],
            'scientific': [
                f"What research methods are described in {title}?",
                f"What are the main findings in {title}?",
                f"What hypotheses are tested in {title}?",
                f"What are the implications of the research in {title}?",
                f"What future research directions are suggested in {title}?"
            ],
            'philosophical': [
                f"What philosophical questions are raised in {title}?",
                f"What arguments are presented in {title}?",
                f"What ethical considerations are discussed in {title}?",
                f"What is the main philosophical position in {title}?",
                f"What are the implications of the ideas in {title}?"
            ],
            'historical': [
                f"What historical period does {title} cover?",
                f"What are the main events described in {title}?",
                f"What historical figures are mentioned in {title}?",
                f"What are the causes and effects discussed in {title}?",
                f"What is the historical significance of the events in {title}?"
            ],
            'futuristic': [
                f"What future technologies are described in {title}?",
                f"What predictions are made in {title}?",
                f"What challenges does the future hold according to {title}?",
                f"What opportunities are envisioned in {title}?",
                f"How might society change according to {title}?"
            ],
            'descriptive': [
                f"What natural phenomena are described in {title}?",
                f"What ecosystems are discussed in {title}?",
                f"What environmental issues are raised in {title}?",
                f"What conservation strategies are mentioned in {title}?",
                f"What is the relationship between humans and nature in {title}?"
            ],
            'analytical': [
                f"What psychological theories are discussed in {title}?",
                f"What research findings are presented in {title}?",
                f"What behavioral patterns are analyzed in {title}?",
                f"What therapeutic approaches are described in {title}?",
                f"What are the practical applications of the psychology in {title}?"
            ]
        }
        
        templates = query_templates.get(genre, query_templates['narrative'])
        selected_queries = random.sample(templates, min(self.config.test_queries_per_book, len(templates)))
        
        book_queries = []
        for query in selected_queries:
            book_queries.append({
                'query': query,
                'book_title': title,
                'genre': genre,
                'expected_context': f"Content from {title}",
                'difficulty': 'medium'
            })
        
        return book_queries

class BookCorpusTester:
    """
    Tester for evaluating hybrid attention RAG with BookCorpus data.
    """
    
    def __init__(self, config: BookCorpusConfig):
        self.config = config
        self.loader = BookCorpusLoader(config)
        self.results = []
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive testing with BookCorpus data."""
        logger.info("Starting comprehensive BookCorpus testing...")
        
        # Load and process books
        books = self.loader.load_sample_books()
        documents = self.loader.process_books_for_rag()
        test_queries = self.loader.generate_test_queries()
        
        # Test different RAG systems
        results = {
            'books_loaded': len(books),
            'documents_processed': len(documents),
            'test_queries': len(test_queries),
            'system_comparisons': {}
        }
        
        # Test base RAG system
        logger.info("Testing base RAG system...")
        base_results = self._test_base_rag(documents, test_queries)
        results['system_comparisons']['base_rag'] = base_results
        
        # Test hybrid attention RAG
        logger.info("Testing hybrid attention RAG system...")
        hybrid_results = self._test_hybrid_rag(documents, test_queries)
        results['system_comparisons']['hybrid_rag'] = hybrid_results
        
        # Analyze results
        analysis = self._analyze_results(results)
        results['analysis'] = analysis
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _test_base_rag(self, documents: List[Document], test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test the base RAG system."""
        try:
            # Initialize base RAG
            base_rag = LongContextRAG()
            base_rag.create_vectorstore(documents)
            
            # Test queries
            query_results = []
            for query_info in test_queries[:10]:  # Test first 10 queries
                query = query_info['query']
                result = base_rag.generate_response(query, use_rag=True)
                
                query_results.append({
                    'query': query,
                    'book_title': query_info['book_title'],
                    'genre': query_info['genre'],
                    'response': result['response'],
                    'retrieved_docs': result['retrieved_docs'],
                    'context_length': result['context_length'],
                    'method': result['method']
                })
            
            return {
                'success': True,
                'query_results': query_results,
                'avg_context_length': np.mean([qr['context_length'] for qr in query_results]),
                'avg_retrieved_docs': np.mean([qr['retrieved_docs'] for qr in query_results])
            }
            
        except Exception as e:
            logger.error(f"Error testing base RAG: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_hybrid_rag(self, documents: List[Document], test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test the hybrid attention RAG system."""
        try:
            # Initialize hybrid RAG
            hybrid_rag = create_hybrid_rag_system()
            hybrid_rag.create_vectorstore(documents)
            
            # Test queries
            query_results = []
            for query_info in test_queries[:10]:  # Test first 10 queries
                query = query_info['query']
                result = hybrid_rag.generate_response(query, use_hybrid=True, task_type='qa')
                
                query_results.append({
                    'query': query,
                    'book_title': query_info['book_title'],
                    'genre': query_info['genre'],
                    'response': result['response'],
                    'retrieved_docs': result['retrieved_docs'],
                    'context_length': result['context_length'],
                    'method': result['method'],
                    'neural_scores': result.get('neural_retrieval_scores'),
                    'attention_shape': result.get('attention_output_shape')
                })
            
            return {
                'success': True,
                'query_results': query_results,
                'avg_context_length': np.mean([qr['context_length'] for qr in query_results]),
                'avg_retrieved_docs': np.mean([qr['retrieved_docs'] for qr in query_results])
            }
            
        except Exception as e:
            logger.error(f"Error testing hybrid RAG: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results."""
        analysis = {
            'performance_comparison': {},
            'genre_analysis': {},
            'context_analysis': {}
        }
        
        # Compare system performance
        for system_name, system_results in results['system_comparisons'].items():
            if system_results.get('success', False):
                analysis['performance_comparison'][system_name] = {
                    'avg_context_length': system_results['avg_context_length'],
                    'avg_retrieved_docs': system_results['avg_retrieved_docs'],
                    'success_rate': 1.0  # All queries succeeded
                }
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results."""
        results_file = Path(self.config.results_dir) / 'bookcorpus_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")

def create_bookcorpus_config() -> BookCorpusConfig:
    """Create BookCorpus configuration for testing."""
    return BookCorpusConfig(
        max_books=8,
        min_book_length=10000,
        max_book_length=80000,
        chunk_size=2000,
        chunk_overlap=200,
        max_chunks_per_book=20,
        test_queries_per_book=3,
        data_dir="data/bookcorpus",
        processed_dir="data/bookcorpus_processed",
        results_dir="results/bookcorpus"
    )

def test_bookcorpus_integration():
    """Test BookCorpus integration with hybrid attention RAG."""
    print("📚 Testing BookCorpus Integration with Hybrid Attention RAG")
    print("=" * 60)
    
    # Create configuration
    config = create_bookcorpus_config()
    
    # Create tester
    tester = BookCorpusTester(config)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print(f"Books loaded: {results['books_loaded']}")
    print(f"Documents processed: {results['documents_processed']}")
    print(f"Test queries: {results['test_queries']}")
    
    print(f"\n🔍 SYSTEM COMPARISON:")
    for system_name, system_results in results['system_comparisons'].items():
        if system_results.get('success', False):
            print(f"  {system_name}:")
            print(f"    Avg context length: {system_results['avg_context_length']:.0f}")
            print(f"    Avg retrieved docs: {system_results['avg_retrieved_docs']:.1f}")
        else:
            print(f"  {system_name}: Failed - {system_results.get('error', 'Unknown error')}")
    
    print(f"\n📁 Results saved to: {config.results_dir}")
    print("✅ BookCorpus integration test completed!")

if __name__ == "__main__":
    test_bookcorpus_integration()
