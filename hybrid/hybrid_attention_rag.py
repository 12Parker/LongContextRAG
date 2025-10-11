"""
Hybrid Attention RAG System Implementation

This module implements the novel hybrid attention mechanism combining:
1. Sliding window attention for local relationships
2. Sparse global attention with strategic landmark tokens
3. Retrieval-augmented segments with dynamic passage integration

Architecture:
- Hierarchical processing framework with three stages
- Individual segment transformer processing
- Cross-segment integration for narrative coherence
- Global synthesis combining local/global representations with retrieved knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class AttentionConfig:
    """Configuration for hybrid attention mechanism."""
    # Sliding window attention
    window_size: int = 512
    window_overlap: int = 64
    
    # Sparse global attention
    num_landmark_tokens: int = 32
    landmark_stride: int = 16
    global_attention_heads: int = 4
    
    # Retrieval-augmented segments
    max_retrieved_segments: int = 8
    segment_length: int = 256
    dynamic_integration: bool = True
    
    # Hierarchical processing
    num_segment_layers: int = 6
    num_integration_layers: int = 3
    num_synthesis_layers: int = 2
    
    # Model dimensions
    hidden_size: int = 3072  # Match text-embedding-3-large
    num_attention_heads: int = 12
    intermediate_size: int = 4096  # Adjusted for larger hidden size
    dropout: float = 0.1

class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for capturing local relationships.
    Processes sequences in overlapping windows to maintain local context.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.window_size = config.window_size
        self.window_overlap = config.window_overlap
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Attention projections
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply sliding window attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (optional)
            
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create overlapping windows
        windows = self._create_windows(hidden_states)
        
        # Process each window
        window_outputs = []
        for window in windows:
            window_output = self._process_window(window, attention_mask)
            window_outputs.append(window_output)
        
        # Reconstruct full sequence from windows
        output = self._reconstruct_from_windows(window_outputs, seq_len)
        
        return output
    
    def _create_windows(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Create overlapping windows from the input sequence."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        windows = []
        
        start = 0
        while start < seq_len:
            end = min(start + self.window_size, seq_len)
            window = hidden_states[:, start:end, :]
            windows.append(window)
            start += self.window_size - self.window_overlap
            
        return windows
    
    def _process_window(self, window: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process a single window with standard attention."""
        batch_size, window_len, hidden_size = window.shape
        
        # Multi-head attention
        Q = self.query(window).view(batch_size, window_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(window).view(batch_size, window_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(window).view(batch_size, window_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            # Apply attention mask
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, window_len, hidden_size)
        
        return self.output(context)
    
    def _reconstruct_from_windows(self, window_outputs: List[torch.Tensor], seq_len: int) -> torch.Tensor:
        """Reconstruct full sequence from window outputs."""
        batch_size, _, hidden_size = window_outputs[0].shape
        
        # Initialize output tensor
        output = torch.zeros(batch_size, seq_len, hidden_size, device=window_outputs[0].device)
        counts = torch.zeros(batch_size, seq_len, device=window_outputs[0].device)
        
        # Reconstruct with overlap handling
        start = 0
        for window_output in window_outputs:
            window_len = window_output.shape[1]
            end = min(start + window_len, seq_len)
            actual_len = end - start
            
            # Add window output to corresponding positions
            output[:, start:end, :] += window_output[:, :actual_len, :]
            counts[:, start:end] += 1
            
            start += self.window_size - self.window_overlap
        
        # Average overlapping regions
        output = output / counts.unsqueeze(-1).clamp(min=1)
        
        return output

class SparseGlobalAttention(nn.Module):
    """
    Sparse global attention with strategic landmark tokens.
    Uses landmark tokens to capture long-range dependencies efficiently.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.num_landmarks = config.num_landmark_tokens
        self.landmark_stride = config.landmark_stride
        self.num_heads = config.global_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.global_attention_heads
        
        # Landmark token embeddings
        self.landmark_embeddings = nn.Parameter(torch.randn(config.num_landmark_tokens, config.hidden_size))
        
        # Attention projections
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse global attention with landmark tokens.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Select landmark tokens
        landmark_indices = self._select_landmark_indices(seq_len)
        landmark_tokens = hidden_states[:, landmark_indices, :]  # [batch_size, num_landmarks, hidden_size]
        
        # Add learnable landmark embeddings
        landmark_tokens = landmark_tokens + self.landmark_embeddings.unsqueeze(0)
        
        # Global attention: all tokens attend to landmarks
        global_output = self._global_attention(hidden_states, landmark_tokens)
        
        # Landmark attention: landmarks attend to all tokens
        landmark_output = self._landmark_attention(landmark_tokens, hidden_states)
        
        # Update original sequence with landmark information
        updated_states = self._update_with_landmarks(hidden_states, landmark_output, landmark_indices)
        
        return updated_states + global_output
    
    def _select_landmark_indices(self, seq_len: int) -> torch.Tensor:
        """Select strategic landmark token positions."""
        # Uniform sampling with stride
        indices = torch.arange(0, seq_len, self.landmark_stride, dtype=torch.long)
        
        # Ensure we don't exceed sequence length
        indices = indices[indices < seq_len]
        
        # If we need more landmarks, add random ones
        if len(indices) < self.num_landmarks:
            remaining = self.num_landmarks - len(indices)
            additional = torch.randperm(seq_len)[:remaining]
            indices = torch.cat([indices, additional])
        
        # If we have too many, sample uniformly
        if len(indices) > self.num_landmarks:
            indices = indices[torch.linspace(0, len(indices)-1, self.num_landmarks, dtype=torch.long)]
        
        return indices
    
    def _global_attention(self, hidden_states: torch.Tensor, landmark_tokens: torch.Tensor) -> torch.Tensor:
        """All tokens attend to landmark tokens."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_landmarks = landmark_tokens.shape[1]
        
        # Multi-head attention
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(landmark_tokens).view(batch_size, num_landmarks, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(landmark_tokens).view(batch_size, num_landmarks, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return self.output(context)
    
    def _landmark_attention(self, landmark_tokens: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Landmark tokens attend to all tokens."""
        batch_size, num_landmarks, hidden_size = landmark_tokens.shape
        seq_len = hidden_states.shape[1]
        
        # Multi-head attention
        Q = self.query(landmark_tokens).view(batch_size, num_landmarks, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_landmarks, hidden_size)
        
        return self.output(context)
    
    def _update_with_landmarks(self, hidden_states: torch.Tensor, landmark_output: torch.Tensor, landmark_indices: torch.Tensor) -> torch.Tensor:
        """Update original sequence with landmark information."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        updated_states = hidden_states.clone()
        
        # Update landmark positions with enhanced representations
        for i, idx in enumerate(landmark_indices):
            if idx < seq_len:
                updated_states[:, idx, :] = landmark_output[:, i, :]
        
        return updated_states

class RetrievalAugmentedSegments(nn.Module):
    """
    Retrieval-augmented segments with dynamic passage integration.
    Integrates retrieved knowledge dynamically based on processing state.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.max_segments = config.max_retrieved_segments
        self.segment_length = config.segment_length
        self.hidden_size = config.hidden_size
        
        # Dynamic integration components
        self.segment_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Integration attention
        self.integration_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Dynamic query generation
        self.query_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
            nn.Tanh()
        )
        
        # Segment relevance scoring
        self.relevance_scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor, retrieved_segments: List[torch.Tensor], 
                processing_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Integrate retrieved segments dynamically.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            retrieved_segments: List of [batch_size, segment_len, hidden_size]
            processing_state: [batch_size, hidden_size] (optional)
            
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        if not retrieved_segments:
            return hidden_states
        
        # Encode retrieved segments
        encoded_segments = self._encode_segments(retrieved_segments)
        
        # Dynamic query generation
        if processing_state is not None:
            dynamic_query = self.query_generator(processing_state)
        else:
            # Use mean of hidden states as processing state
            dynamic_query = hidden_states.mean(dim=1)
        
        # Score segment relevance
        relevant_segments = self._score_and_select_segments(encoded_segments, dynamic_query)
        
        # Integrate segments with main sequence
        integrated_states = self._integrate_segments(hidden_states, relevant_segments)
        
        return integrated_states
    
    def _encode_segments(self, segments: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode retrieved segments."""
        encoded_segments = []
        for segment in segments:
            encoded = self.segment_encoder(segment)
            encoded_segments.append(encoded)
        return encoded_segments
    
    def _score_and_select_segments(self, segments: List[torch.Tensor], query: torch.Tensor) -> List[torch.Tensor]:
        """Score and select most relevant segments."""
        if not segments:
            return []
        
        # Calculate relevance scores
        scores = []
        for segment in segments:
            # Use mean of segment as segment representation
            segment_repr = segment.mean(dim=1)  # [batch_size, hidden_size]
            
            # Concatenate query and segment representation
            combined = torch.cat([query, segment_repr], dim=-1)
            
            # Calculate relevance score
            score = self.relevance_scorer(combined)
            scores.append(score)
        
        # Select top segments
        scores = torch.cat(scores, dim=-1)  # [batch_size, num_segments]
        top_indices = torch.topk(scores, min(self.max_segments, len(segments)), dim=-1).indices
        
        # Select relevant segments
        relevant_segments = []
        for i in range(top_indices.shape[1]):
            idx = top_indices[:, i]
            relevant_segments.append(segments[idx[0]])  # Simplified for batch processing
        
        return relevant_segments
    
    def _integrate_segments(self, hidden_states: torch.Tensor, segments: List[torch.Tensor]) -> torch.Tensor:
        """Integrate selected segments with main sequence."""
        if not segments:
            return hidden_states
        
        # Concatenate all segments
        all_segments = torch.cat(segments, dim=1)  # [batch_size, total_segment_len, hidden_size]
        
        # Cross-attention between main sequence and segments
        attended_states, _ = self.integration_attention(
            query=hidden_states,
            key=all_segments,
            value=all_segments
        )
        
        # Residual connection
        return hidden_states + attended_states

class HybridAttentionRAG(nn.Module):
    """
    Complete hybrid attention RAG system implementing the three-stage hierarchical framework.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Individual components
        self.sliding_window_attention = SlidingWindowAttention(config)
        self.sparse_global_attention = SparseGlobalAttention(config)
        self.retrieval_augmented_segments = RetrievalAugmentedSegments(config)
        
        # Hierarchical processing layers
        self.segment_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_segment_layers)
        ])
        
        self.integration_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_integration_layers)
        ])
        
        self.synthesis_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_synthesis_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, retrieved_segments: List[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the hierarchical processing framework.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            retrieved_segments: List of retrieved segments
            attention_mask: [batch_size, seq_len] (optional)
            
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        # Stage 1: Individual segment transformer processing
        segment_output = self._process_segments(hidden_states, attention_mask)
        
        # Stage 2: Cross-segment integration for narrative coherence
        integration_output = self._integrate_segments(segment_output, retrieved_segments)
        
        # Stage 3: Global synthesis
        synthesis_output = self._global_synthesis(integration_output)
        
        return self.layer_norm(synthesis_output)
    
    def _process_segments(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stage 1: Process individual segments with hybrid attention."""
        x = hidden_states
        
        for layer in self.segment_layers:
            # Apply sliding window attention
            sliding_output = self.sliding_window_attention(x, attention_mask)
            
            # Apply sparse global attention
            global_output = self.sparse_global_attention(sliding_output)
            
            # Apply transformer layer
            x = layer(global_output, src_key_padding_mask=attention_mask)
        
        return x
    
    def _integrate_segments(self, hidden_states: torch.Tensor, retrieved_segments: List[torch.Tensor] = None) -> torch.Tensor:
        """Stage 2: Cross-segment integration for narrative coherence."""
        x = hidden_states
        
        # Apply retrieval-augmented segments
        if retrieved_segments:
            x = self.retrieval_augmented_segments(x, retrieved_segments)
        
        # Apply integration layers
        for layer in self.integration_layers:
            x = layer(x)
        
        return x
    
    def _global_synthesis(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Stage 3: Global synthesis combining local/global representations."""
        x = hidden_states
        
        # Apply synthesis layers
        for layer in self.synthesis_layers:
            x = layer(x)
        
        return x

# Example usage and testing
def create_sample_data(batch_size: int = 2, seq_len: int = 1024, hidden_size: int = 768) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Create sample data for testing."""
    # Main sequence
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Retrieved segments
    retrieved_segments = [
        torch.randn(batch_size, 256, hidden_size),
        torch.randn(batch_size, 128, hidden_size),
        torch.randn(batch_size, 192, hidden_size)
    ]
    
    return hidden_states, retrieved_segments

def test_hybrid_attention_rag():
    """Test the hybrid attention RAG system."""
    print("🧪 Testing Hybrid Attention RAG System")
    print("=" * 50)
    
    # Create configuration
    config = AttentionConfig(
        window_size=256,
        num_landmark_tokens=16,
        max_retrieved_segments=4,
        hidden_size=512,
        num_attention_heads=8
    )
    
    # Create model
    model = HybridAttentionRAG(config)
    
    # Create sample data
    hidden_states, retrieved_segments = create_sample_data(
        batch_size=1, seq_len=512, hidden_size=config.hidden_size
    )
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Number of retrieved segments: {len(retrieved_segments)}")
    
    # Forward pass
    with torch.no_grad():
        output = model(hidden_states, retrieved_segments)
    
    print(f"Output shape: {output.shape}")
    print("✅ Hybrid Attention RAG system test completed successfully!")

if __name__ == "__main__":
    test_hybrid_attention_rag()
