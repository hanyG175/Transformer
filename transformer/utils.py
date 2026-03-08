"""
Utility functions for working with Transformers.

This module provides helper functions for:
- Creating transformer models with different configurations
- Data processing utilities
- Attention visualization
- Analysis and debugging tools
"""

import numpy as np


class TransformerConfig:
    """
    Configuration class for Transformer models.
    
    Stores all hyperparameters and allows easy creation of different models.
    """
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, 
                 n_layers=6, max_seq_length=2048, dropout=0.1):
        """
        Initialize configuration.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_layers: Number of encoder/decoder layers
            max_seq_length: Maximum sequence length
            dropout: Dropout rate (for reference, not used in this implementation)
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.dropout = dropout
    
    def __repr__(self):
        return (
            f"TransformerConfig("
            f"vocab_size={self.vocab_size}, "
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"d_ff={self.d_ff}, "
            f"n_layers={self.n_layers}, "
            f"max_seq_length={self.max_seq_length})"
        )
    
    def get_param_count(self):
        """
        Estimate total parameter count for the model.
        
        Returns:
            total_params: Approximate number of parameters
        """
        d_model = self.d_model
        n_layers = self.n_layers
        vocab_size = self.vocab_size
        
        # Embeddings
        embedding_params = vocab_size * d_model
        
        # Per encoder/decoder layer:
        # - Attention: 4 * d_model^2 (Q, K, V, O projections)
        # - Feed-forward: 2 * d_model * d_ff = 8 * d_model^2
        # - Layer norms: 2 * 2 * d_model (4 layer norms)
        params_per_layer = 4 * d_model**2 + 8 * d_model**2 + 8 * d_model
        
        # Encoder + Decoder
        transformer_params = 2 * n_layers * params_per_layer
        
        # Output projection
        output_params = d_model * vocab_size
        
        total = embedding_params + transformer_params + output_params
        return total
    
    @classmethod
    def base_model(cls, vocab_size):
        """Create base model configuration (original paper)."""
        return cls(
            vocab_size=vocab_size,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_layers=6
        )
    
    @classmethod
    def large_model(cls, vocab_size):
        """Create large model configuration."""
        return cls(
            vocab_size=vocab_size,
            d_model=1024,
            n_heads=16,
            d_ff=4096,
            n_layers=24
        )
    
    @classmethod
    def small_model(cls, vocab_size):
        """Create small model configuration."""
        return cls(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            d_ff=1024,
            n_layers=3
        )


def create_positional_encoding_heatmap(d_model, seq_len):
    """
    Create data for visualizing positional encodings.
    
    The positional encoding has interesting structure that can be visualized
    as a heatmap showing how positions differ.
    
    Args:
        d_model: Model dimension
        seq_len: Sequence length
    
    Returns:
        encoding_matrix: Array of shape (seq_len, d_model) with encodings
    """
    from transformer.embeddings import PositionalEncoding
    
    pe = PositionalEncoding(d_model, max_seq_length=max(seq_len, 512))
    return pe.get_encoding(seq_len)


def compute_attention_entropy(attention_weights):
    """
    Compute entropy of attention weights to measure focus/diversity.
    
    High entropy: Attention distributed evenly (uncertain)
    Low entropy: Attention focused on few positions (certain)
    
    Args:
        attention_weights: Array of shape (batch, n_heads, seq_len, seq_len)
    
    Returns:
        entropy: Array of shape (batch, n_heads, seq_len) with entropy per position
    """
    # Avoid log(0)
    weights_safe = np.maximum(attention_weights, 1e-10)
    
    # Shannon entropy: -sum(p * log(p))
    entropy = -np.sum(attention_weights * np.log(weights_safe), axis=-1)
    
    return entropy


def create_sequence_pair(vocab_size, src_len, tgt_len, start_token=1, end_token=2):
    """
    Create a random sequence pair for demonstration.
    
    Args:
        vocab_size: Vocabulary size
        src_len: Source sequence length
        tgt_len: Target sequence length
        start_token: Token ID for sequence start
        end_token: Token ID for sequence end
    
    Returns:
        src_ids: Source sequence of shape (1, src_len)
        tgt_ids: Target sequence of shape (1, tgt_len)
    """
    src_ids = np.random.randint(3, vocab_size, (1, src_len))
    src_ids[0, 0] = start_token
    src_ids[0, -1] = end_token
    
    tgt_ids = np.random.randint(3, vocab_size, (1, tgt_len))
    tgt_ids[0, 0] = start_token
    tgt_ids[0, -1] = end_token
    
    return src_ids, tgt_ids


def softmax_temperature(logits, temperature=1.0):
    """
    Apply temperature scaling to logits before softmax.
    
    Temperature controls the "sharpness" of the probability distribution:
    - Temperature < 1: Sharper distribution (more confident)
    - Temperature = 1: Standard softmax
    - Temperature > 1: Softer distribution (more uncertain)
    
    Args:
        logits: Array of logits
        temperature: Temperature value
    
    Returns:
        probabilities: Softmax with temperature applied
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return probabilities


def top_k_sampling(logits, k=10, temperature=1.0):
    """
    Top-k sampling for text generation.
    
    Only samples from the k most likely tokens. This eliminates very unlikely
    tokens while still allowing diversity.
    
    Args:
        logits: Array of logits from model output
        k: Number of top tokens to consider
        temperature: Temperature for probability scaling
    
    Returns:
        sampled_indices: Sampled token indices
    """
    # Apply temperature
    probs = softmax_temperature(logits.reshape(-1), temperature)
    
    # Find top-k
    top_k_indices = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_indices]
    
    # Renormalize
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    
    # Sample
    sampled_idx = np.random.choice(top_k_indices, p=top_k_probs)
    
    return sampled_idx


def nucleus_sampling(logits, p=0.9, temperature=1.0):
    """
    Nucleus (top-p) sampling for text generation.
    
    Samples from the smallest set of tokens with cumulative probability >= p.
    This adapts the number of tokens based on the probability distribution.
    
    Args:
        logits: Array of logits from model output
        p: Cumulative probability threshold (e.g., 0.9 for top 90%)
        temperature: Temperature for probability scaling
    
    Returns:
        sampled_indices: Sampled token indices
    """
    # Apply temperature
    probs = softmax_temperature(logits.reshape(-1), temperature)
    
    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative probabilities
    cumsum_probs = np.cumsum(sorted_probs)
    
    # Find cutoff index where cumsum >= p
    cutoff_idx = np.searchsorted(cumsum_probs, p, side='left')
    cutoff_idx = min(cutoff_idx + 1, len(sorted_indices) - 1)
    
    # Keep only top-p tokens
    nucleus_indices = sorted_indices[:cutoff_idx]
    nucleus_probs = sorted_probs[:cutoff_idx]
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
    
    # Sample
    sampled_idx = np.random.choice(nucleus_indices, p=nucleus_probs)
    
    return sampled_idx


def analyze_model_config(config):
    """
    Print detailed analysis of a configuration.
    
    Args:
        config: TransformerConfig object
    """
    print(f"\nTransformer Configuration Analysis")
    print("=" * 50)
    print(f"Configuration: {config}")
    print(f"\nModel Dimensions:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff} (ratio: {config.d_ff / config.d_model:.1f}x d_model)")
    print(f"  n_heads: {config.n_heads}")
    print(f"  d_k = d_v: {config.d_model // config.n_heads}")
    
    print(f"\nModel Structure:")
    print(f"  Encoder layers: {config.n_layers}")
    print(f"  Decoder layers: {config.n_layers}")
    print(f"  Total layers: {2 * config.n_layers}")
    
    print(f"\nVocabulary:")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Max sequence length: {config.max_seq_length}")
    
    param_count = config.get_param_count()
    print(f"\nEstimated Parameters: {param_count:,.0f}")
    if param_count < 1e6:
        print(f"  ({param_count / 1e3:.1f}K)")
    elif param_count < 1e9:
        print(f"  ({param_count / 1e6:.1f}M)")
    else:
        print(f"  ({param_count / 1e9:.1f}B)")
    
    print(f"\nAttention Details:")
    print(f"  Each attention head size: {config.d_model // config.n_heads}")
    print(f"  Total attention projections per layer:")
    print(f"    Input: {config.d_model} -> {config.n_heads} * {config.d_model // config.n_heads}")
    print(f"    Output: {config.n_heads} * {config.d_model // config.n_heads} -> {config.d_model}")


if __name__ == "__main__":
    print("Transformer Utilities Module")
    print("=" * 50)
    
    # Example usage
    config = TransformerConfig.base_model(vocab_size=50000)
    analyze_model_config(config)
    
    print("\n" + "=" * 50)
    print("Available configurations:")
    small = TransformerConfig.small_model(50000)
    print(f"  Small: {small.get_param_count() / 1e6:.1f}M parameters")
    
    base = TransformerConfig.base_model(50000)
    print(f"  Base: {base.get_param_count() / 1e6:.1f}M parameters")
    
    large = TransformerConfig.large_model(50000)
    print(f"  Large: {large.get_param_count() / 1e6:.1f}M parameters")
