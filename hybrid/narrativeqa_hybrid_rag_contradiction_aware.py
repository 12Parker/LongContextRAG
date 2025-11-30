#!/usr/bin/env python3
"""
NarrativeQA Hybrid RAG System with Contradiction-Aware Attention Gating

This extends the Hybrid RAG system with a novel contradiction detection mechanism
that identifies potentially conflicting information in retrieved passages and
handles them appropriately during generation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from pathlib import Path
import sys
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 not installed. Install with: pip install rank-bm25")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import config
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import tiktoken

logger = logging.getLogger(__name__)


# ============================================================================
# CONTRADICTION DETECTION MODULE
# ============================================================================

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


class ContradictionAwareRetriever:
    """
    Wrapper that adds contradiction-aware gating to any base retriever.
    
    This implements the core CAAG (Contradiction-Aware Attention Gating) mechanism:
    1. Retrieve documents using base retriever
    2. Detect contradictions using multi-signal analysis
    3. Apply conflict resolution strategy
    4. Return enhanced results with contradiction metadata
    """
    
    def __init__(self,
                 base_retriever: 'HybridRetriever',
                 embeddings: Any,
                 strategy: ConflictStrategy = ConflictStrategy.FLAG,
                 contradiction_threshold: float = 0.35,
                 similarity_threshold: float = 0.65):
        """
        Initialize the contradiction-aware retriever.
        
        Args:
            base_retriever: The underlying retriever (HybridRetriever)
            embeddings: Embedding model
            strategy: How to handle detected contradictions
            contradiction_threshold: Threshold for flagging contradictions
            similarity_threshold: Min similarity to consider docs related
        """
        self.base_retriever = base_retriever
        self.embeddings = embeddings
        self.strategy = strategy
        
        self.detector = ContradictionDetector(
            embeddings=embeddings,
            similarity_threshold=similarity_threshold,
            contradiction_threshold=contradiction_threshold
        )
        
        # Statistics tracking
        self.stats = {
            'total_retrievals': 0,
            'retrievals_with_contradictions': 0,
            'total_contradictions_detected': 0,
            'contradictions_by_type': defaultdict(int)
        }
    
    def retrieve(self,
                 query: str,
                 k: int = 10,
                 return_scores: bool = True) -> RetrievalResult:
        """
        Retrieve documents with contradiction detection and handling.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            return_scores: Whether to include retrieval scores
            
        Returns:
            RetrievalResult with documents and contradiction metadata
        """
        self.stats['total_retrievals'] += 1
        
        # Get base retrieval results
        base_results = self.base_retriever.retrieve(
            query, k=k, return_scores=return_scores
        )
        
        # Detect contradictions
        contradictions = self.detector.detect_contradictions(base_results)
        
        # Find contradiction groups
        contradiction_groups = self.detector.find_contradiction_groups(
            contradictions, len(base_results)
        )
        
        # Update statistics
        if contradictions:
            self.stats['retrievals_with_contradictions'] += 1
            self.stats['total_contradictions_detected'] += len(contradictions)
            for cp in contradictions:
                self.stats['contradictions_by_type'][cp.contradiction_type] += 1
        
        # Apply strategy
        result = self._apply_strategy(
            base_results, contradictions, contradiction_groups
        )
        
        return result
    
    def _apply_strategy(self,
                        docs: List[Dict[str, Any]],
                        contradictions: List[ContradictionPair],
                        groups: List[Set[int]]) -> RetrievalResult:
        """Apply the configured conflict resolution strategy."""
        
        if not contradictions:
            return RetrievalResult(
                documents=docs,
                contradictions=[],
                contradiction_groups=[],
                strategy_applied=None
            )
        
        if self.strategy == ConflictStrategy.FLAG:
            return self._strategy_flag(docs, contradictions, groups)
        
        elif self.strategy == ConflictStrategy.REMOVE:
            return self._strategy_remove(docs, contradictions, groups)
        
        elif self.strategy == ConflictStrategy.SEPARATE:
            return self._strategy_separate(docs, contradictions, groups)
        
        elif self.strategy == ConflictStrategy.ARBITRATE:
            return self._strategy_arbitrate(docs, contradictions, groups)
        
        # Default: flag
        return self._strategy_flag(docs, contradictions, groups)
    
    def _strategy_flag(self,
                       docs: List[Dict[str, Any]],
                       contradictions: List[ContradictionPair],
                       groups: List[Set[int]]) -> RetrievalResult:
        """Flag contradictions but keep all documents."""
        # Add contradiction metadata to documents
        contradiction_map = defaultdict(list)
        for cp in contradictions:
            contradiction_map[cp.doc_idx_1].append({
                'with_doc': cp.doc_idx_2,
                'score': cp.contradiction_score,
                'type': cp.contradiction_type
            })
            contradiction_map[cp.doc_idx_2].append({
                'with_doc': cp.doc_idx_1,
                'score': cp.contradiction_score,
                'type': cp.contradiction_type
            })
        
        for idx, doc in enumerate(docs):
            if idx in contradiction_map:
                doc['has_contradiction'] = True
                doc['contradictions'] = contradiction_map[idx]
            else:
                doc['has_contradiction'] = False
                doc['contradictions'] = []
        
        return RetrievalResult(
            documents=docs,
            contradictions=contradictions,
            contradiction_groups=groups,
            strategy_applied=ConflictStrategy.FLAG
        )
    
    def _strategy_remove(self,
                         docs: List[Dict[str, Any]],
                         contradictions: List[ContradictionPair],
                         groups: List[Set[int]]) -> RetrievalResult:
        """Remove lower-scoring document from each contradiction pair."""
        to_remove = set()
        
        for cp in contradictions:
            # Get retrieval scores
            score_1 = docs[cp.doc_idx_1].get('scores', {}).get('combined', 0)
            score_2 = docs[cp.doc_idx_2].get('scores', {}).get('combined', 0)
            
            # Remove the one with lower retrieval score
            if score_1 < score_2:
                to_remove.add(cp.doc_idx_1)
            else:
                to_remove.add(cp.doc_idx_2)
        
        filtered_docs = [d for idx, d in enumerate(docs) if idx not in to_remove]
        
        return RetrievalResult(
            documents=filtered_docs,
            contradictions=contradictions,
            contradiction_groups=groups,
            strategy_applied=ConflictStrategy.REMOVE,
            removed_indices=list(to_remove)
        )
    
    def _strategy_separate(self,
                           docs: List[Dict[str, Any]],
                           contradictions: List[ContradictionPair],
                           groups: List[Set[int]]) -> RetrievalResult:
        """Separate contradictory documents for explicit handling."""
        # Find all indices involved in contradictions
        contradiction_indices = set()
        for cp in contradictions:
            contradiction_indices.add(cp.doc_idx_1)
            contradiction_indices.add(cp.doc_idx_2)
        
        # Split documents
        main_docs = []
        for idx, doc in enumerate(docs):
            if idx not in contradiction_indices:
                doc['is_contradictory'] = False
                main_docs.append(doc)
            else:
                doc['is_contradictory'] = True
                doc['contradiction_group'] = None
                # Find which group this doc belongs to
                for gidx, group in enumerate(groups):
                    if idx in group:
                        doc['contradiction_group'] = gidx
                        break
                main_docs.append(doc)  # Keep but mark
        
        return RetrievalResult(
            documents=main_docs,
            contradictions=contradictions,
            contradiction_groups=groups,
            strategy_applied=ConflictStrategy.SEPARATE,
            arbitration_needed=True
        )
    
    def _strategy_arbitrate(self,
                            docs: List[Dict[str, Any]],
                            contradictions: List[ContradictionPair],
                            groups: List[Set[int]]) -> RetrievalResult:
        """Mark for LLM arbitration (actual arbitration happens in generation)."""
        # Similar to separate, but with stronger signal for arbitration
        result = self._strategy_separate(docs, contradictions, groups)
        result.strategy_applied = ConflictStrategy.ARBITRATE
        result.arbitration_needed = True
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval and contradiction statistics."""
        return {
            **self.stats,
            'contradiction_rate': (
                self.stats['retrievals_with_contradictions'] / 
                max(self.stats['total_retrievals'], 1)
            ),
            'avg_contradictions_per_retrieval': (
                self.stats['total_contradictions_detected'] /
                max(self.stats['total_retrievals'], 1)
            )
        }


# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """
    Hybrid retriever combining BM25 (keyword) and Dense (semantic) retrieval.
    """
    
    def __init__(self, 
                 documents: List[str],
                 vectorstore: Any,
                 embeddings: Any,
                 alpha: float = 0.5,
                 k1: float = 1.5,
                 b: float = 0.75):
        """
        Initialize hybrid retriever.
        
        Args:
            documents: List of document chunks (text)
            vectorstore: Dense vector store (Chroma)
            embeddings: Embedding model
            alpha: Weight for combining scores (0=BM25 only, 1=Dense only, 0.5=equal)
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling document length normalization
        """
        self.documents = documents
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.alpha = alpha
        
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available. Falling back to dense retrieval only.")
            self.bm25 = None
            self.alpha = 1.0
        else:
            logger.info("Initializing BM25 index...")
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
            logger.info(f"BM25 index created with {len(documents)} documents")
        
        self.doc_metadata = {}
        for i, doc in enumerate(documents):
            self.doc_metadata[i] = {
                'chunk_id': i,
                'text': doc
            }
    
    def retrieve(self, 
                 query: str, 
                 k: int = 10,
                 return_scores: bool = False) -> List[Dict[str, Any]]:
        """Retrieve top-k documents using hybrid BM25 + Dense retrieval."""
        start_time = time.time()
        
        # Get BM25 scores
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
        else:
            bm25_scores = np.zeros(len(self.documents))
        
        # Get dense retrieval scores
        dense_k = min(k * 3, len(self.documents))
        dense_results = self.vectorstore.similarity_search_with_score(
            query, k=dense_k
        )
        
        dense_scores = np.zeros(len(self.documents))
        for doc, score in dense_results:
            chunk_id = doc.metadata.get('chunk_id', -1)
            if chunk_id >= 0 and chunk_id < len(dense_scores):
                dense_scores[chunk_id] = np.exp(-score)
        
        # Normalize scores
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        dense_scores_norm = self._normalize_scores(dense_scores)
        
        # Combine scores
        if self.bm25 is not None:
            combined_scores = (
                (1 - self.alpha) * bm25_scores_norm +
                self.alpha * dense_scores_norm
            )
        else:
            combined_scores = dense_scores_norm
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:
                doc_info = {
                    'chunk_id': idx,
                    'text': self.documents[idx],
                    'metadata': self.doc_metadata[idx]
                }
                
                if return_scores:
                    doc_info['scores'] = {
                        'combined': float(combined_scores[idx]),
                        'bm25': float(bm25_scores_norm[idx]),
                        'dense': float(dense_scores_norm[idx])
                    }
                
                results.append(doc_info)
        
        retrieval_time = time.time() - start_time
        logger.debug(f"Hybrid retrieval took {retrieval_time*1000:.1f}ms")
        
        return results
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)
        
        return (scores - min_score) / (max_score - min_score)


# ============================================================================
# MAIN RAG SYSTEM WITH CONTRADICTION AWARENESS
# ============================================================================

class NarrativeQAHybridRAG:
    """
    Hybrid RAG system for NarrativeQA with Contradiction-Aware Attention Gating.
    
    This extends the base hybrid RAG with:
    1. Contradiction detection between retrieved passages
    2. Multiple strategies for handling conflicts (flag, remove, separate, arbitrate)
    3. Enhanced prompts that explicitly handle contradictory information
    4. Provenance tracking for answer attribution
    """
    
    def __init__(self, 
                 max_context_tokens: int = 50000,
                 chunk_size: int = 1500,
                 chunk_overlap: int = 200,
                 top_k_results: int = 10,
                 db_path: str = "./narrativeqa_hybrid_vectordb",
                 story_text: str = None,
                 retrieval_mode: str = 'hybrid',
                 hybrid_alpha: float = 0.5,
                 retriever_checkpoint: str = None,
                 # New contradiction-aware parameters
                 enable_contradiction_detection: bool = True,
                 conflict_strategy: str = 'flag',
                 contradiction_threshold: float = 0.35,
                 similarity_threshold: float = 0.65):
        """
        Initialize the NarrativeQA Hybrid RAG system with contradiction awareness.
        
        Args:
            max_context_tokens: Maximum tokens for context
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k_results: Number of top results to retrieve
            db_path: Path to vector database
            story_text: The actual story text to use
            retrieval_mode: 'hybrid', 'neural', 'dense', or 'bm25'
            hybrid_alpha: Weight for hybrid retrieval
            retriever_checkpoint: Path to trained neural retriever
            enable_contradiction_detection: Enable CAAG module
            conflict_strategy: 'flag', 'remove', 'separate', or 'arbitrate'
            contradiction_threshold: Threshold for detecting contradictions
            similarity_threshold: Min similarity to consider docs related
        """
        self.max_context_tokens = max_context_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_results = top_k_results
        self.db_path = db_path
        self.story_text = story_text
        self.retrieval_mode = retrieval_mode.lower()
        self.hybrid_alpha = hybrid_alpha
        self.retriever_checkpoint = retriever_checkpoint
        
        # Contradiction detection settings
        self.enable_contradiction_detection = enable_contradiction_detection
        self.conflict_strategy = ConflictStrategy(conflict_strategy.lower())
        self.contradiction_threshold = contradiction_threshold
        self.similarity_threshold = similarity_threshold
        
        # Validate retrieval mode
        valid_modes = ['hybrid', 'neural', 'dense', 'bm25']
        if self.retrieval_mode not in valid_modes:
            logger.warning(f"Invalid retrieval mode '{self.retrieval_mode}'. Using 'hybrid'.")
            self.retrieval_mode = 'hybrid'
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.4
        )
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize semantic chunker
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name="text-embedding-3-large"
        )
        
        self.text_splitter = ClusterSemanticChunker(
            embedding_function=embedding_function,
            max_chunk_size=chunk_size,
            length_function=len
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize components
        self.vectorstore = None
        self.document_embeddings = None
        self.documents = []
        self.hybrid_retriever = None
        self.contradiction_retriever = None
        
        self._initialize_vectordb()
        self._initialize_retrievers()
    
    def _initialize_vectordb(self):
        """Initialize the vector database with NarrativeQA stories."""
        print("Initializing NarrativeQA Vector Database...")
        print("Creating fresh database...")
        self._create_vectordb()
    
    def _create_vectordb(self):
        """Create vector database from NarrativeQA stories."""
        print("Loading NarrativeQA story...")
        
        try:
            if self.story_text:
                story = self.story_text
                print(f"  Using provided story text ({len(story)} characters)")
            else:
                from datasets import load_dataset
                dataset = load_dataset("narrativeqa", split="train")
                story_data = dataset[0].get('document', '')
                if isinstance(story_data, dict):
                    story = story_data.get('text', '')
                else:
                    story = str(story_data)
                print(f"  Loaded story from dataset ({len(story)} characters)")
            
            if not story:
                raise ValueError("No story text available")
            
            # Split story into chunks
            chunks = self.text_splitter.split_text(story)
            print(f"  Created {len(chunks)} chunks from story")
            
            self.documents = chunks
            
            documents = []
            metadatas = []
            
            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    'story_id': 'current_story',
                    'chunk_id': j,
                    'total_chunks': len(chunks),
                    'story_index': 0
                })
            
            print("Creating vector database...")
            self.vectorstore = Chroma.from_texts(
                texts=documents,
                metadatas=metadatas,
                embedding=self.embeddings,
                persist_directory=self.db_path
            )
            
            print(f"Vector database created at {self.db_path}")
            
        except Exception as e:
            print(f"Error creating vector database: {e}")
            raise
    
    def _initialize_retrievers(self):
        """Initialize retrieval components including contradiction detection."""
        print(f"Initializing retriever (mode: {self.retrieval_mode})...")
        
        if self.retrieval_mode in ['hybrid', 'bm25']:
            if not BM25_AVAILABLE:
                print("  BM25 not available. Install with: pip install rank-bm25")
                print("  Falling back to dense retrieval only")
                self.retrieval_mode = 'dense'
            else:
                print(f"  Building BM25 index for {len(self.documents)} documents...")
                self.hybrid_retriever = HybridRetriever(
                    documents=self.documents,
                    vectorstore=self.vectorstore,
                    embeddings=self.embeddings,
                    alpha=self.hybrid_alpha if self.retrieval_mode == 'hybrid' else 0.0
                )
                print(f"  Hybrid retriever initialized (alpha={self.hybrid_alpha})")
        
        # Initialize contradiction-aware retriever
        if self.enable_contradiction_detection and self.hybrid_retriever:
            print(f"  Initializing Contradiction-Aware Attention Gating...")
            print(f"     Strategy: {self.conflict_strategy.value}")
            print(f"     Contradiction threshold: {self.contradiction_threshold}")
            
            self.contradiction_retriever = ContradictionAwareRetriever(
                base_retriever=self.hybrid_retriever,
                embeddings=self.embeddings,
                strategy=self.conflict_strategy,
                contradiction_threshold=self.contradiction_threshold,
                similarity_threshold=self.similarity_threshold
            )
            print(f"  Contradiction detection enabled")
        else:
            print(f"  Contradiction detection: {'disabled' if not self.enable_contradiction_detection else 'not available'}")
        
        print(f"Retriever initialized")
    
    def retrieve_documents(self, 
                          query: str, 
                          k: int = None,
                          expand_context: bool = True,
                          context_window: int = 1,
                          detect_contradictions: bool = None) -> Tuple[List[Dict[str, Any]], Optional[RetrievalResult]]:
        """
        Retrieve relevant documents with optional contradiction detection.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            expand_context: Whether to include neighboring chunks
            context_window: Number of neighboring chunks to include
            detect_contradictions: Override for contradiction detection
            
        Returns:
            Tuple of (documents list, RetrievalResult with contradiction info)
        """
        if k is None:
            k = self.top_k_results
        
        if detect_contradictions is None:
            detect_contradictions = self.enable_contradiction_detection
        
        start_time = time.time()
        retrieval_result = None
        
        # Use contradiction-aware retrieval if enabled
        if detect_contradictions and self.contradiction_retriever:
            retrieval_result = self.contradiction_retriever.retrieve(
                query, k=k, return_scores=True
            )
            docs = [{
                'page_content': d.get('text', d.get('page_content', '')),
                'metadata': d.get('metadata', {}),
                'scores': d.get('scores', {}),
                'has_contradiction': d.get('has_contradiction', False),
                'contradictions': d.get('contradictions', []),
                'is_contradictory': d.get('is_contradictory', False),
                'contradiction_group': d.get('contradiction_group')
            } for d in retrieval_result.documents]
            
            if retrieval_result.contradictions:
                logger.info(
                    f"Detected {len(retrieval_result.contradictions)} contradictions "
                    f"in {len(retrieval_result.contradiction_groups)} groups"
                )
        
        elif self.retrieval_mode in ['hybrid', 'bm25'] and self.hybrid_retriever:
            results = self.hybrid_retriever.retrieve(query, k=k, return_scores=True)
            docs = [{
                'page_content': r['text'],
                'metadata': r['metadata'],
                'scores': r.get('scores', {})
            } for r in results]
        
        else:
            # Dense retrieval fallback
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            docs = [{
                'page_content': doc.page_content,
                'metadata': doc.metadata,
                'scores': {'dense': float(score)}
            } for doc, score in results]
        
        # Expand with context if enabled
        if expand_context and context_window > 0:
            docs = self._expand_with_neighbors(docs, window=context_window)
        
        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieved {len(docs)} documents in {retrieval_time*1000:.1f}ms")
        
        return docs, retrieval_result
    
    def _expand_with_neighbors(self, 
                               docs: List[Dict[str, Any]], 
                               window: int = 1) -> List[Dict[str, Any]]:
        """Expand retrieved chunks with neighboring chunks."""
        expanded_indices = set()
        
        for doc in docs:
            chunk_id = doc['metadata'].get('chunk_id', -1)
            if chunk_id >= 0:
                for offset in range(-window, window + 1):
                    neighbor_id = chunk_id + offset
                    if 0 <= neighbor_id < len(self.documents):
                        expanded_indices.add(neighbor_id)
        
        original_ids = [doc['metadata'].get('chunk_id', -1) for doc in docs]
        expanded_docs = []
        seen = set()
        
        for chunk_id in original_ids:
            if chunk_id in expanded_indices and chunk_id not in seen:
                # Preserve original doc info if available
                original_doc = next(
                    (d for d in docs if d['metadata'].get('chunk_id') == chunk_id), 
                    None
                )
                if original_doc:
                    expanded_docs.append(original_doc)
                else:
                    expanded_docs.append({
                        'page_content': self.documents[chunk_id],
                        'metadata': {'chunk_id': chunk_id},
                        'scores': {}
                    })
                seen.add(chunk_id)
        
        for chunk_id in sorted(expanded_indices):
            if chunk_id not in seen:
                expanded_docs.append({
                    'page_content': self.documents[chunk_id],
                    'metadata': {'chunk_id': chunk_id},
                    'scores': {}
                })
                seen.add(chunk_id)
        
        return expanded_docs
    
    def _build_context_with_contradictions(self,
                                            docs: List[Dict[str, Any]],
                                            retrieval_result: Optional[RetrievalResult]) -> str:
        """
        Build context string with contradiction annotations.
        
        Contradictory passages are marked for explicit handling by the LLM.
        """
        context_parts = []
        
        for i, doc in enumerate(docs):
            chunk_id = doc['metadata'].get('chunk_id', i)
            scores = doc.get('scores', {})
            
            # Build header with scores
            header_parts = [f"Chunk {chunk_id}"]
            
            if scores:
                score_strs = [f"{k}={v:.3f}" for k, v in scores.items()]
                header_parts.append(f"[{', '.join(score_strs)}]")
            
            # Add contradiction warning if detected
            if doc.get('has_contradiction') or doc.get('is_contradictory'):
                contradictions = doc.get('contradictions', [])
                if contradictions:
                    conflicting_chunks = [str(c['with_doc']) for c in contradictions]
                    header_parts.append(
                        f"[POTENTIAL CONFLICT with chunk(s): {', '.join(conflicting_chunks)}]"
                    )
                else:
                    header_parts.append("[POTENTIAL CONFLICT DETECTED]")
            
            header = " ".join(header_parts)
            context_parts.append(f"{header}:\n{doc['page_content']}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self,
                      question: str,
                      context: str,
                      retrieval_result: Optional[RetrievalResult]) -> str:
        """
        Build prompt with contradiction-aware instructions.
        
        If contradictions are detected, the prompt includes explicit instructions
        for handling conflicting information.
        """
        has_contradictions = (
            retrieval_result and 
            retrieval_result.contradictions and 
            len(retrieval_result.contradictions) > 0
        )
        
        if has_contradictions and self.conflict_strategy in [
            ConflictStrategy.FLAG, 
            ConflictStrategy.SEPARATE,
            ConflictStrategy.ARBITRATE
        ]:
            # Build contradiction summary
            contradiction_info = []
            for cp in retrieval_result.contradictions:
                contradiction_info.append(
                    f"- Chunks {cp.doc_idx_1} and {cp.doc_idx_2}: "
                    f"{cp.contradiction_type} conflict (confidence: {cp.contradiction_score:.2f})"
                )
            
            contradiction_section = "\n".join(contradiction_info)
            
            if self.conflict_strategy == ConflictStrategy.ARBITRATE:
                conflict_instruction = """
CONFLICTING INFORMATION DETECTED:
{contradictions}
"""
            else:
                conflict_instruction = """
POTENTIAL CONFLICTS DETECTED:
{contradictions}
"""
            
            conflict_instruction = conflict_instruction.format(
                contradictions=contradiction_section
            )
        else:
            conflict_instruction = ""
        
        prompt = f"""Based on the following story excerpts, answer the question.

Context:
{context}
{conflict_instruction}
Question: {question}

Please provide a concise, direct answer (1-2 sentences maximum). Focus on the key facts from the story excerpts.

Important instructions:
- Carefully analyze ALL provided excerpts - the answer may be present even if worded differently
- Look for synonyms, paraphrases, and related concepts
- Consider context and implications, not just exact word matches
- Only say the excerpts don't contain information if you are absolutely certain
- If you find partial or related information, include it in your answer"""

        return prompt
    
    def generate_response(self, 
                         question: str,
                         expand_context: bool = False,
                         context_window: int = 1,
                         detect_contradictions: bool = None) -> Dict[str, Any]:
        """
        Generate a response using contradiction-aware hybrid RAG.
        
        Args:
            question: The question to answer
            expand_context: Whether to include neighboring chunks
            context_window: Number of neighboring chunks to include
            detect_contradictions: Override contradiction detection setting
            
        Returns:
            Dictionary containing response and metadata including contradiction info
        """
        start_time = time.time()
        
        try:
            print(f"Retrieving relevant chunks for: {question[:50]}...")
            
            # Retrieve documents with contradiction detection
            docs, retrieval_result = self.retrieve_documents(
                question, 
                k=self.top_k_results,
                expand_context=expand_context,
                context_window=context_window,
                detect_contradictions=detect_contradictions
            )
            
            print(f"  Retrieved {len(docs)} chunks")
            
            # Log contradiction info
            if retrieval_result and retrieval_result.contradictions:
                print(f"  Detected {len(retrieval_result.contradictions)} potential contradictions")
                for cp in retrieval_result.contradictions:
                    print(f"     - Chunks {cp.doc_idx_1} vs {cp.doc_idx_2}: "
                          f"{cp.contradiction_type} (score: {cp.contradiction_score:.2f})")
            
            if not docs:
                print("  No documents retrieved")
                return {
                    'response': "I couldn't find relevant information to answer your question.",
                    'context': '',
                    'retrieved_docs': 0,
                    'context_length': 0,
                    'context_tokens': 0,
                    'response_time': time.time() - start_time,
                    'method': f'hybrid_rag_{self.retrieval_mode}',
                    'contradictions_detected': 0,
                    'contradiction_groups': 0
                }
            
            # Build context with contradiction annotations
            context = self._build_context_with_contradictions(docs, retrieval_result)
            
            # Check context length
            context_tokens = len(self.tokenizer.encode(context))
            if context_tokens > self.max_context_tokens:
                words = context.split()
                max_words = int(self.max_context_tokens * 0.8)
                context = " ".join(words[:max_words]) + "..."
                context_tokens = len(self.tokenizer.encode(context))
            
            # Build prompt with contradiction awareness
            prompt = self._build_prompt(question, context, retrieval_result)
            
            # Generate response
            response = self.llm.invoke(prompt)
            generated_answer = response.content
            
            elapsed_time = time.time() - start_time
            
            print(f"  Response generated in {elapsed_time:.2f}s")
            print(f"  Context tokens: {context_tokens}")
            
            # Build result with contradiction metadata
            result = {
                'response': generated_answer,
                'context': context,
                'retrieved_docs': len(docs),
                'context_length': len(context),
                'context_tokens': context_tokens,
                'response_time': elapsed_time,
                'method': f'hybrid_rag_{self.retrieval_mode}',
                'retrieval_mode': self.retrieval_mode,
                'context_expanded': expand_context,
                'contradiction_detection_enabled': self.enable_contradiction_detection,
                'conflict_strategy': self.conflict_strategy.value
            }
            
            # Add contradiction details
            if retrieval_result:
                result['contradictions_detected'] = len(retrieval_result.contradictions)
                result['contradiction_groups'] = len(retrieval_result.contradiction_groups)
                result['contradictions'] = [
                    {
                        'chunk_1': cp.doc_idx_1,
                        'chunk_2': cp.doc_idx_2,
                        'score': cp.contradiction_score,
                        'type': cp.contradiction_type
                    }
                    for cp in retrieval_result.contradictions
                ]
                if retrieval_result.removed_indices:
                    result['removed_chunks'] = retrieval_result.removed_indices
            else:
                result['contradictions_detected'] = 0
                result['contradiction_groups'] = 0
                result['contradictions'] = []
            
            return result
            
        except Exception as e:
            print(f"  Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return {
                'response': f"Error generating response: {str(e)}",
                'context': '',
                'retrieved_docs': 0,
                'context_length': 0,
                'context_tokens': 0,
                'response_time': time.time() - start_time,
                'method': f'hybrid_rag_{self.retrieval_mode}',
                'error': str(e),
                'contradictions_detected': 0
            }
    
    def get_contradiction_statistics(self) -> Dict[str, Any]:
        """Get statistics about contradiction detection."""
        if self.contradiction_retriever:
            return self.contradiction_retriever.get_statistics()
        return {'message': 'Contradiction detection not enabled'}

def main():
    """Main function for testing the Contradiction-Aware NarrativeQA RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NarrativeQA RAG with Contradiction-Aware Attention Gating"
    )
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--chunk-size", type=int, default=1500, 
                       help="Chunk size for text splitting")
    parser.add_argument("--top-k", type=int, default=10, 
                       help="Number of top results to retrieve")
    parser.add_argument("--db-path", type=str, default="./narrativeqa_hybrid_vectordb", 
                       help="Path to vector database")
    parser.add_argument("--mode", type=str, default="hybrid", 
                       choices=['hybrid', 'neural', 'dense', 'bm25'],
                       help="Retrieval mode")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for hybrid retrieval")
    parser.add_argument("--expand-context", action="store_true",
                       help="Expand retrieved chunks with neighbors")
    parser.add_argument("--context-window", type=int, default=1,
                       help="Number of neighboring chunks to include")
    
    # Contradiction detection arguments
    parser.add_argument("--no-contradiction-detection", action="store_true",
                       help="Disable contradiction detection")
    parser.add_argument("--conflict-strategy", type=str, default="flag",
                       choices=['flag', 'remove', 'separate', 'arbitrate'],
                       help="Strategy for handling contradictions")
    parser.add_argument("--contradiction-threshold", type=float, default=0.35,
                       help="Threshold for detecting contradictions")
    parser.add_argument("--similarity-threshold", type=float, default=0.65,
                       help="Min similarity to consider docs related")
    
    args = parser.parse_args()
    
    # Check BM25 availability
    if args.mode in ['hybrid', 'bm25'] and not BM25_AVAILABLE:
        print("Error: BM25 mode requires rank-bm25 package")
        print("Install with: pip install rank-bm25")
        return
    
    # Initialize RAG system
    print("Initializing NarrativeQA RAG with Contradiction-Aware Attention Gating...")
    print(f"  Retrieval Mode: {args.mode}")
    print(f"  Contradiction Detection: {'Disabled' if args.no_contradiction_detection else 'Enabled'}")
    print(f"  Conflict Strategy: {args.conflict_strategy}")
    
    if args.mode == 'hybrid':
        print(f"  Alpha: {args.alpha} (BM25={1-args.alpha:.1f}, Dense={args.alpha:.1f})")
    
    rag = NarrativeQAHybridRAG(
        chunk_size=args.chunk_size,
        top_k_results=args.top_k,
        db_path=args.db_path,
        retrieval_mode=args.mode,
        hybrid_alpha=args.alpha,
        enable_contradiction_detection=not args.no_contradiction_detection,
        conflict_strategy=args.conflict_strategy,
        contradiction_threshold=args.contradiction_threshold,
        similarity_threshold=args.similarity_threshold
    )
    
    def print_result(result: Dict[str, Any]):
        """Print formatted result."""
        print(f"\nResponse:")
        print(f"{result['response']}")
        print(f"\nMetadata:")
        print(f"  Mode: {result.get('retrieval_mode', 'unknown')}")
        print(f"  Retrieved docs: {result['retrieved_docs']}")
        print(f"  Context tokens: {result['context_tokens']}")
        print(f"  Response time: {result['response_time']:.2f}s")
        
        if result.get('context_expanded'):
            print(f"  Context expanded: Yes (window={args.context_window})")
        
        # Contradiction info
        print(f"\nContradiction Analysis:")
        print(f"  Detection enabled: {result.get('contradiction_detection_enabled', False)}")
        print(f"  Strategy: {result.get('conflict_strategy', 'N/A')}")
        print(f"  Contradictions found: {result.get('contradictions_detected', 0)}")
        print(f"  Contradiction groups: {result.get('contradiction_groups', 0)}")
        
        if result.get('contradictions'):
            print(f"  Details:")
            for c in result['contradictions']:
                print(f"    - Chunks {c['chunk_1']} vs {c['chunk_2']}: "
                      f"{c['type']} (score: {c['score']:.2f})")
        
        if result.get('removed_chunks'):
            print(f"  Removed chunks: {result['removed_chunks']}")
    
    if args.interactive:
        print("\nInteractive mode - Enter queries (type 'quit' to exit)")
        print("   Type 'stats' to see contradiction detection statistics")
        
        while True:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.lower() == 'stats':
                stats = rag.get_contradiction_statistics()
                print("\nContradiction Detection Statistics:")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
                continue
            
            if not query:
                continue
            
            print(f"\nProcessing: {query}")
            result = rag.generate_response(
                query,
                expand_context=args.expand_context,
                context_window=args.context_window
            )
            print_result(result)
    
    elif args.query:
        print(f"\nProcessing: {args.query}")
        result = rag.generate_response(
            args.query,
            expand_context=args.expand_context,
            context_window=args.context_window
        )
        print_result(result)
    
    else:
        # Default test query
        test_query = "Who is the main character in the story?"
        print(f"\nTesting with: {test_query}")
        result = rag.generate_response(test_query)
        print_result(result)


if __name__ == "__main__":
    main()
