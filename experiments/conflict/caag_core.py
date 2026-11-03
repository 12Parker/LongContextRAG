"""
CAAG contradiction detector, extracted VERBATIM from
hybrid/narrativeqa_hybrid_rag_contradiction_aware.py (lines 56-423, commit 89e545d).

Why a copy: the original module imports langchain, chromadb and `chunking_evaluation`
at import time, so the detector could not be used without the full RAG stack.
Do not edit the logic here; tests/test_caag_parity.py checks it stays identical
to the original source.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConflictStrategy(Enum):
    """Strategies for handling detected contradictions."""
    FLAG = "flag"           # Flag contradictions but keep all docs
    REMOVE = "remove"       # Remove lower-confidence contradicting doc
    SEPARATE = "separate"   # Separate contradictory docs for explicit handling
    ARBITRATE = "arbitrate" # Use LLM to arbitrate between contradictions


@dataclass
class ContradictionPair:
    """Represents a detected contradiction between two document chunks."""
    doc_idx_1: int
    doc_idx_2: int
    contradiction_score: float
    contradiction_type: str  # 'negation', 'temporal', 'factual', 'perspective'
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return (f"ContradictionPair(docs=[{self.doc_idx_1}, {self.doc_idx_2}], "
                f"score={self.contradiction_score:.3f}, type={self.contradiction_type})")


@dataclass 
class RetrievalResult:
    """Enhanced retrieval result with contradiction metadata."""
    documents: List[Dict[str, Any]]
    contradictions: List[ContradictionPair]
    contradiction_groups: List[Set[int]]  # Groups of mutually contradicting docs
    strategy_applied: Optional[ConflictStrategy]
    removed_indices: List[int] = field(default_factory=list)
    arbitration_needed: bool = False


class ContradictionDetector:
    """
    Detects contradictions between retrieved document chunks.
    
    This implements a multi-signal approach to contradiction detection:
    1. Semantic similarity + negation asymmetry (lexical)
    2. Named entity conflicts (factual)
    3. Temporal marker conflicts (temporal)
    4. Sentiment/stance polarity (perspective)
    
    The detector can be extended with learned components for better accuracy.
    """
    
    # Negation and contrast words
    NEGATION_WORDS = {
        'not', 'never', 'no', 'none', 'nobody', 'nothing', 'nowhere',
        "didn't", "doesn't", "don't", "wasn't", "weren't", "isn't", "aren't",
        "hadn't", "hasn't", "haven't", "won't", "wouldn't", "couldn't", "shouldn't",
        'neither', 'nor', 'without', 'lack', 'lacks', 'lacking', 'failed', 'refuse',
        'refused', 'deny', 'denied', 'denies'
    }
    
    CONTRAST_WORDS = {
        'but', 'however', 'although', 'though', 'yet', 'despite', 'nevertheless',
        'nonetheless', 'whereas', 'while', 'contrary', 'opposite', 'instead',
        'rather', 'unlike', 'conversely', 'on the other hand', 'in contrast'
    }
    
    # Temporal markers for detecting temporal conflicts
    TEMPORAL_MARKERS = {
        'before': -1, 'after': 1, 'during': 0, 'while': 0,
        'previously': -1, 'later': 1, 'earlier': -1, 'afterwards': 1,
        'first': -1, 'then': 1, 'finally': 1, 'initially': -1,
        'already': -1, 'yet': 1, 'still': 0, 'now': 0,
        'yesterday': -1, 'tomorrow': 1, 'today': 0
    }
    
    def __init__(self,
                 embeddings: Any,
                 similarity_threshold: float = 0.65,
                 contradiction_threshold: float = 0.35,
                 use_semantic: bool = True,
                 use_negation: bool = True,
                 use_entity: bool = True,
                 use_temporal: bool = True):
        """
        Initialize the contradiction detector.
        
        Args:
            embeddings: Embedding model for semantic similarity
            similarity_threshold: Min similarity to consider docs as related
            contradiction_threshold: Min score to flag as contradiction
            use_semantic: Enable semantic similarity analysis
            use_negation: Enable negation pattern detection
            use_entity: Enable named entity conflict detection
            use_temporal: Enable temporal marker conflict detection
        """
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.use_semantic = use_semantic
        self.use_negation = use_negation
        self.use_entity = use_entity
        self.use_temporal = use_temporal
        
        # Optional: spaCy for better NER (lazy load)
        self._nlp = None
    
    def _get_nlp(self):
        """Lazy load spaCy model for NER."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                logger.warning("spaCy not available. Entity-based detection disabled.")
                self._nlp = False
        return self._nlp if self._nlp else None
    
    def _compute_semantic_similarity(self, 
                                      texts: List[str]) -> np.ndarray:
        """Compute pairwise semantic similarity matrix."""
        if not texts:
            return np.array([])
        
        # Get embeddings
        doc_embeddings = self.embeddings.embed_documents(texts)
        doc_embeddings = np.array(doc_embeddings)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        normalized = doc_embeddings / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = normalized @ normalized.T
        return similarity_matrix
    
    def _count_negations(self, text: str) -> Tuple[int, int]:
        """Count negation and contrast words in text."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        negation_count = len(words & self.NEGATION_WORDS)
        contrast_count = sum(1 for phrase in self.CONTRAST_WORDS 
                            if phrase in text_lower)
        
        return negation_count, contrast_count
    
    def _extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract named entities from text."""
        nlp = self._get_nlp()
        if nlp is None:
            return {}
        
        doc = nlp(text)
        entities = defaultdict(set)
        
        for ent in doc.ents:
            entities[ent.label_].add(ent.text.lower())
        
        return dict(entities)
    
    def _check_entity_conflicts(self,
                                 text_i: str,
                                 text_j: str) -> Tuple[float, Dict]:
        """Check for conflicting named entity information."""
        entities_i = self._extract_entities(text_i)
        entities_j = self._extract_entities(text_j)
        
        if not entities_i or not entities_j:
            return 0.0, {}
        
        conflicts = {}
        conflict_score = 0.0
        
        # Check for same entity type with different values in similar context
        common_types = set(entities_i.keys()) & set(entities_j.keys())
        
        for ent_type in common_types:
            vals_i = entities_i[ent_type]
            vals_j = entities_j[ent_type]
            
            # If there are different values for same entity type
            if vals_i != vals_j and not (vals_i & vals_j):
                # This could indicate a conflict (e.g., different dates, numbers)
                if ent_type in ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'MONEY', 'PERCENT']:
                    conflicts[ent_type] = {'text_1': vals_i, 'text_2': vals_j}
                    conflict_score += 0.3
        
        return min(conflict_score, 1.0), conflicts
    
    def _check_temporal_conflicts(self,
                                   text_i: str,
                                   text_j: str) -> Tuple[float, Dict]:
        """Check for temporal ordering conflicts."""
        text_i_lower = text_i.lower()
        text_j_lower = text_j.lower()
        
        # Find temporal markers in each text
        markers_i = {word: val for word, val in self.TEMPORAL_MARKERS.items() 
                    if word in text_i_lower}
        markers_j = {word: val for word, val in self.TEMPORAL_MARKERS.items() 
                    if word in text_j_lower}
        
        if not markers_i or not markers_j:
            return 0.0, {}
        
        # Check for conflicting temporal ordering
        # E.g., text_i says "before X" and text_j says "after X" for same event
        conflict_evidence = {}
        
        # Simple heuristic: if one text has mostly "before" markers
        # and the other has mostly "after" markers about similar content
        avg_i = np.mean(list(markers_i.values())) if markers_i else 0
        avg_j = np.mean(list(markers_j.values())) if markers_j else 0
        
        if avg_i * avg_j < 0:  # Different temporal directions
            conflict_evidence = {
                'text_1_markers': markers_i,
                'text_2_markers': markers_j,
                'direction_conflict': True
            }
            return 0.4, conflict_evidence
        
        return 0.0, {}
    
    def detect_contradictions(self,
                               docs: List[Dict[str, Any]]) -> List[ContradictionPair]:
        """
        Detect contradictions between retrieved documents.
        
        Args:
            docs: List of document dictionaries with 'text' field
            
        Returns:
            List of ContradictionPair objects
        """
        if len(docs) < 2:
            return []
        
        texts = [d.get('text', d.get('page_content', '')) for d in docs]
        contradictions = []
        
        # Compute semantic similarity matrix
        if self.use_semantic:
            similarity_matrix = self._compute_semantic_similarity(texts)
        else:
            similarity_matrix = np.ones((len(texts), len(texts)))
        
        # Check each pair
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                semantic_sim = similarity_matrix[i, j]
                
                # Only check for contradictions if documents are related
                if semantic_sim < self.similarity_threshold:
                    continue
                
                contradiction_signals = []
                total_score = 0.0
                evidence = {'semantic_similarity': float(semantic_sim)}
                
                # 1. Negation asymmetry check
                if self.use_negation:
                    neg_i, contrast_i = self._count_negations(texts[i])
                    neg_j, contrast_j = self._count_negations(texts[j])
                    
                    negation_diff = abs(neg_i - neg_j)
                    contrast_diff = abs(contrast_i - contrast_j)
                    
                    if negation_diff >= 2 or contrast_diff >= 1:
                        negation_score = min((negation_diff * 0.15 + 
                                            contrast_diff * 0.2), 0.5)
                        total_score += negation_score
                        contradiction_signals.append('negation')
                        evidence['negation'] = {
                            'text_1': {'negations': neg_i, 'contrasts': contrast_i},
                            'text_2': {'negations': neg_j, 'contrasts': contrast_j}
                        }
                
                # 2. Entity conflict check
                if self.use_entity:
                    entity_score, entity_conflicts = self._check_entity_conflicts(
                        texts[i], texts[j]
                    )
                    if entity_score > 0:
                        total_score += entity_score
                        contradiction_signals.append('entity')
                        evidence['entity_conflicts'] = entity_conflicts
                
                # 3. Temporal conflict check
                if self.use_temporal:
                    temporal_score, temporal_conflicts = self._check_temporal_conflicts(
                        texts[i], texts[j]
                    )
                    if temporal_score > 0:
                        total_score += temporal_score
                        contradiction_signals.append('temporal')
                        evidence['temporal_conflicts'] = temporal_conflicts
                
                # Weight by semantic similarity (more similar = more likely real contradiction)
                final_score = total_score * semantic_sim
                
                if final_score >= self.contradiction_threshold:
                    # Determine primary contradiction type
                    if 'entity' in contradiction_signals:
                        ctype = 'factual'
                    elif 'temporal' in contradiction_signals:
                        ctype = 'temporal'
                    elif 'negation' in contradiction_signals:
                        ctype = 'negation'
                    else:
                        ctype = 'perspective'
                    
                    contradictions.append(ContradictionPair(
                        doc_idx_1=i,
                        doc_idx_2=j,
                        contradiction_score=final_score,
                        contradiction_type=ctype,
                        evidence=evidence
                    ))
        
        # Sort by contradiction score (highest first)
        contradictions.sort(key=lambda x: x.contradiction_score, reverse=True)
        
        return contradictions
    
    def find_contradiction_groups(self,
                                   contradictions: List[ContradictionPair],
                                   num_docs: int) -> List[Set[int]]:
        """
        Group documents that are mutually contradicting using union-find.
        
        Args:
            contradictions: List of contradiction pairs
            num_docs: Total number of documents
            
        Returns:
            List of sets, each containing indices of mutually contradicting docs
        """
        if not contradictions:
            return []
        
        # Union-Find implementation
        parent = list(range(num_docs))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union contradicting documents
        for cp in contradictions:
            union(cp.doc_idx_1, cp.doc_idx_2)
        
        # Group by root
        groups = defaultdict(set)
        contradiction_indices = set()
        for cp in contradictions:
            contradiction_indices.add(cp.doc_idx_1)
            contradiction_indices.add(cp.doc_idx_2)
        
        for idx in contradiction_indices:
            root = find(idx)
            groups[root].add(idx)
        
        # Return only groups with more than one member
        return [g for g in groups.values() if len(g) > 1]


