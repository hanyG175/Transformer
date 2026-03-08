"""
Implementation of Attention Mechanisms

This module contains the actual implementation of:
1. Scaled Dot-Product Attention
2. Multi-Head Attention

Both mechanisms are core to the Transformer architecture.
"""

import numpy as np
import sys


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax((Q K^T) / √d_k) V
    
    Mathematical derivation:
    
    1. Compute compatibility scores: scores = Q K^T / √d_k
       - Q ∈ ℝ^(n × d_k): Query matrix
       - K ∈ ℝ^(m × d_k): Key matrix
       - Scores shape: (n, m)
       - Scaling by √d_k stabilizes gradients
    
    2. Apply mask (optional): Set masked positions to -∞
       scores_masked = scores + mask
    
    3. Apply softmax: weights = softmax(scores_masked)
       - Converts scores to probability distribution
       - weights ∈ ℝ^(n × m), each row sums to 1
    
    4. Apply to values: output = weights V
       - Weighted sum of values
       - Output shape: (n, d_v)
    
    Args:
        Q: Query matrix of shape (batch, n_heads, seq_len, d_k)
        K: Key matrix of shape (batch, n_heads, seq_len, d_k)
        V: Value matrix of shape (batch, n_heads, seq_len, d_v)
        mask: Optional mask of shape (batch, 1, seq_len, seq_len) or
              (1, 1, seq_len, seq_len). Values to mask should be very
              negative (e.g., -1e9)
    
    Returns:
        output: Attention output of shape (batch, n_heads, seq_len, d_v)
        attention_weights: Attention weights of shape
                          (batch, n_heads, seq_len, seq_len)
    """
    # Get the dimension of keys for scaling
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores
    # scores = Q K^T / √d_k
    # Q shape: (..., seq_len_q, d_k)
    # K shape: (..., seq_len_k, d_k)
    # K^T shape: (..., d_k, seq_len_k)
    # scores shape: (..., seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    # Step 2: Apply mask if provided
    if mask is not None:
        # Replace masked positions with a large negative number
        # This will become ~0 after softmax
        scores = scores + mask
    
    # Step 3: Apply softmax to get attention weights
    # softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    attention_weights = softmax(scores, axis=-1)
    
    # Handle NaN values that might result from masked positions
    attention_weights = np.nan_to_num(attention_weights, nan=0.0)
    
    # Step 4: Weight values with attention
    # output = weights V
    # attention_weights shape: (..., seq_len_q, seq_len_k)
    # V shape: (..., seq_len_k, d_v)
    # output shape: (..., seq_len_q, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    
    softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    
    Subtracting the maximum improves numerical stability by preventing
    overflow when exponentiating large values.
    
    Args:
        x: Input array
        axis: Axis along which to apply softmax
    
    Returns:
        softmax_x: Softmax applied along axis
    """
    # Subtract maximum for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # Compute exp
    exp_x = np.exp(x_shifted)
    
    # Normalize by sum
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    """
    Multi-head attention mechanism.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    This allows the model to jointly attend to information from different
    representation subspaces.
    
    Mathematical foundation:
    
    1. Linear projections: 
       Q_i = Q W_i^Q,  K_i = K W_i^K,  V_i = V W_i^V
       where W matrices have shape (d_model, d_k/d_v)
    
    2. Apply attention independently on each head:
       head_i = Attention(Q_i, K_i, V_i)
    
    3. Concatenate and project:
       MultiHead = Concat(head_1, ..., head_h) W^O
       where W^O has shape (h*d_v, d_model)
    
    Benefits:
    - Different heads can attend to different parts of input
    - Creates h different "representation subspaces"
    - Increased model capacity without much computational cost
    - Provides ensemble-like benefits
    
    Typical settings:
    - h = 8 or 16 attention heads
    - d_model = 512
    - d_k = d_v = d_model / h = 512 / 8 = 64
    
    Attributes:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_k: Dimension of each head (d_model // n_heads)
        d_v: Value dimension (d_model // n_heads)
    """
    
    def __init__(self, d_model, n_heads):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Total model dimension
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Initialize projection matrices
        # W_Q, W_K, W_V: (d_model, d_model) - will be split into heads
        # W_O: (d_model, d_model) - output projection
        self.W_Q = self._initialize_weight(d_model, d_model)
        self.W_K = self._initialize_weight(d_model, d_model)
        self.W_V = self._initialize_weight(d_model, d_model)
        self.W_O = self._initialize_weight(d_model, d_model)
    
    def _initialize_weight(self, in_dim, out_dim):
        """
        Xavier/Glorot initialization.
        
        W ~ U(-√(6/(in+out)), √(6/(in+out)))
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
        
        Returns:
            weight: Initialized weight matrix of shape (in_dim, out_dim)
        """
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, (in_dim, out_dim))
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            Q: Query matrix of shape (batch_size, seq_len, d_model)
            K: Key matrix of shape (batch_size, seq_len, d_model)
            V: Value matrix of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: List of attention weights for each head
        """
        batch_size = Q.shape[0]
        
        # Step 1: Linear projections in batch form
        # Project Q, K, V and split into multiple heads
        Q_proj = np.matmul(Q, self.W_Q)  # (batch, seq_len, d_model)
        K_proj = np.matmul(K, self.W_K)  # (batch, seq_len, d_model)
        V_proj = np.matmul(V, self.W_V)  # (batch, seq_len, d_model)
        
        # Reshape and transpose for multi-head processing
        # Split the d_model dimension into (n_heads, d_k/d_v)
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) 
        #                            -> (batch, n_heads, seq_len, d_k)
        Q_heads = self._split_heads(Q_proj, batch_size)
        K_heads = self._split_heads(K_proj, batch_size)
        V_heads = self._split_heads(V_proj, batch_size)
        
        # Step 2: Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask
        )
        
        # Step 3: Concatenate heads
        # (batch, n_heads, seq_len, d_v) -> (batch, seq_len, n_heads, d_v)
        #                                  -> (batch, seq_len, d_model)
        concat_output = self._concat_heads(attn_output, batch_size)
        
        # Step 4: Final linear transformation
        output = np.matmul(concat_output, self.W_O)
        
        return output, attn_weights
    
    def _split_heads(self, x, batch_size):
        """
        Split the last dimension into (n_heads, d_k or d_v).
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            batch_size: Batch size
        
        Returns:
            output: Shape (batch_size, n_heads, seq_len, d_k or d_v)
        """
        seq_len = x.shape[1]
        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k)
        x_reshaped = x.reshape(batch_size, seq_len, self.n_heads, -1)
        # Transpose: (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        return x_reshaped.transpose(0, 2, 1, 3)
    
    def _concat_heads(self, x, batch_size):
        """
        Concatenate multiple heads.
        
        Args:
            x: Input of shape (batch_size, n_heads, seq_len, d_v)
            batch_size: Batch size
        
        Returns:
            output: Shape (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[2]
        # Transpose: (batch, n_heads, seq_len, d_v) -> (batch, seq_len, n_heads, d_v)
        x_trans = x.transpose(0, 2, 1, 3)
        # Reshape: (batch, seq_len, n_heads, d_v) -> (batch, seq_len, d_model)
        return x_trans.reshape(batch_size, seq_len, self.d_model)
