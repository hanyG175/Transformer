"""
Transformer Encoder and Decoder Layers

This module implements individual transformer layers that form the basis
of the full Transformer architecture.

Mathematical Foundation:
========================

TRANSFORMER LAYER STRUCTURE
---------------------------

Each encoder/decoder layer consists of two main sublayers:

1. MULTI-HEAD ATTENTION SUBLAYER
2. FEED-FORWARD NETWORK SUBLAYER

These are connected using RESIDUAL CONNECTIONS and LAYER NORMALIZATION.

RESIDUAL CONNECTIONS (Skip Connections)
---------------------------------------

Residual connections enable training of very deep networks by creating
shortcuts that allow gradients to flow directly from later layers to
earlier ones.

sublayer_output = sublayer_input + sublayer(sublayer_input)

Benefits:
- Improve gradient flow during backpropagation
- Allow networks to learn identity mapping
- Enable training of much deeper models
- Improve convergence speed

Mathematically, this means the layer learns a residual function F(x),
and the actual transformation is x + F(x). This is easier to optimize
than learning F(x) directly.

LAYER NORMALIZATION
-------------------

Layer normalization is applied before each sublayer:

LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

Where:
- x: Input vector
- μ = (1/d) Σ_i x_i: Mean over features
- σ² = (1/d) Σ_i (x_i - μ)²: Variance over features
- d: Feature dimension
- ε: Small constant for numerical stability (typically 1e-6)
- γ, β: Learnable scale and shift parameters

Why layer norm before the sublayer? (PreNorm variant)
- Better gradient flow
- More stable training
- Improves generalization

Alternative: PostNorm applies normalization after residual connection
- Original Transformer uses PostNorm
- Modern architectures often use PreNorm

ENCODER LAYER
=============

EncoderLayer consists of:

1. Multi-Head Self-Attention:
   - Q, K, V all come from the same input
   - Allows positions to attend to all other positions
   - No masking (bidirectional context)

2. Feed-Forward Network:
   - Positions processed independently
   - Learns complex transformations through two-layer MLP

Forward pass:
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))

DECODER LAYER
=============

DecoderLayer consists of:

1. Multi-Head Masked Self-Attention:
   - Q, K, V from same input
   - Causal mask prevents attending to future positions
   - Ensures autoregressive generation (each position only sees past)

2. Multi-Head Cross-Attention:
   - Q from decoder state
   - K, V from encoder output
   - Allows decoder to attend to source sequence
   - No masking (can see entire source)

3. Feed-Forward Network:
   - Same as encoder

Forward pass:
x = x + MaskedMultiHeadAttention(LayerNorm(x))
x = x + CrossMultiHeadAttention(LayerNorm(x), encoder_output)
x = x + FeedForward(LayerNorm(x))

CAUSAL MASKING
--------------

For autoregressive decoding, we prevent positions from attending to
future positions using a causal mask.

Causal mask for sequence length 4:
    [[0, -∞, -∞, -∞],
     [0,  0, -∞, -∞],
     [0,  0,  0, -∞],
     [0,  0,  0,  0]]

Position i can only attend to positions j where j ≤ i.

This enforces the autoregressive property: when predicting token at
position i, the model can only see tokens at positions 0 to i-1.
"""

import numpy as np
from .attention_impl import MultiHeadAttention
from .feedforward import FeedForwardNetwork


def layer_norm(x, eps=1e-6):
    """
    Layer normalization.
    
    LayerNorm(x) = (x - μ) / √(σ² + ε)
    
    Normalizes each sample independently over features.
    
    Args:
        x: Input array of shape (..., d_model)
        eps: Small constant for numerical stability
    
    Returns:
        normalized: Normalized array of same shape as input
    """
    # Compute mean and variance over last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    normalized = (x - mean) / np.sqrt(variance + eps)
    
    return normalized


def create_causal_mask(seq_len):
    """
    Create a causal (autoregressive) attention mask.
    
    A causal mask ensures that position i can only attend to positions
    j where j <= i (i.e., current and past positions only).
    
    For a sequence of length 4, the mask is:
    [[0,    -inf, -inf, -inf],
     [0,    0,    -inf, -inf],
     [0,    0,    0,    -inf],
     [0,    0,    0,    0   ]]
    
    This prevents information from flowing from future to past, enabling
    autoregressive generation.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        mask: Causal mask of shape (seq_len, seq_len) with -inf for masked
              positions
    """
    # Create lower triangular matrix of ones
    # tril gives 1s on and below diagonal, 0s above
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    # Invert: 0s where attention allowed (diag and below), 1s where blocked (above)
    mask = 1 - mask
    
    # Convert 1s to large negative values, keep 0s as 0s
    return -mask * 1e9


def create_padding_mask(seq):
    """
    Create a padding mask for variable length sequences.
    
    Padding tokens should not contribute to attention. This mask
    marks padding positions with -inf.
    
    Args:
        seq: Sequence of shape (batch_size, seq_len) with padding token index 0
    
    Returns:
        mask: Padding mask of shape (batch_size, 1, 1, seq_len) suitable
              for broadcasting in attention computation
    """
    # Assume padding token is 0
    mask = (seq == 0).astype(np.float32)
    # Expand dimensions for broadcasting: (batch, seq_len) -> (batch, 1, 1, seq_len)
    mask = mask.reshape(mask.shape[0], 1, 1, -1)
    # Convert to -inf for masked positions
    mask = mask * (-1e9)
    return mask


class TransformerEncoderLayer:
    """
    Single Transformer encoder layer.
    
    Combines:
    1. Multi-head self-attention
    2. Feed-forward network
    
    With residual connections and layer normalization.
    
    Attributes:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
    """
    
    def __init__(self, d_model, n_heads, d_ff):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feedforward = FeedForwardNetwork(d_model, d_ff)
    
    def forward(self, x, mask=None):
        """
        Forward pass through encoder layer.
        
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Output of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        # Normalize input
        x_norm = layer_norm(x)
        # Apply attention
        attn_output, _ = self.attention.forward(x_norm, x_norm, x_norm, mask)
        # Add residual
        x = x + attn_output
        
        # Feed-forward with residual connection
        # Normalize
        x_norm = layer_norm(x)
        # Apply feed-forward
        ff_output = self.feedforward.forward(x_norm)
        # Add residual
        x = x + ff_output
        
        return x


class TransformerDecoderLayer:
    """
    Single Transformer decoder layer.
    
    Combines:
    1. Masked multi-head self-attention (for autoregressive property)
    2. Multi-head cross-attention (to encoder output)
    3. Feed-forward network
    
    With residual connections and layer normalization.
    
    Attributes:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
    """
    
    def __init__(self, d_model, n_heads, d_ff):
        """
        Initialize decoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Two attention layers: self-attention and cross-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feedforward = FeedForwardNetwork(d_model, d_ff)
    
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Forward pass through decoder layer.
        
        x = x + MaskedSelfAttention(LayerNorm(x))
        x = x + CrossAttention(LayerNorm(x), encoder_output)
        x = x + FeedForward(LayerNorm(x))
        
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Padding mask for cross-attention
        
        Returns:
            output: Output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual
        x_norm = layer_norm(x)
        self_attn_output, _ = self.self_attention.forward(
            x_norm, x_norm, x_norm, self_attn_mask
        )
        x = x + self_attn_output
        
        # Cross-attention with residual
        x_norm = layer_norm(x)
        cross_attn_output, _ = self.cross_attention.forward(
            x_norm,  # Q from decoder
            encoder_output,  # K from encoder
            encoder_output,  # V from encoder
            cross_attn_mask
        )
        x = x + cross_attn_output
        
        # Feed-forward with residual
        x_norm = layer_norm(x)
        ff_output = self.feedforward.forward(x_norm)
        x = x + ff_output
        
        return x
