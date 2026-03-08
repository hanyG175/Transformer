"""
Embedding and Positional Encoding Components

This module implements:
1. Token embeddings: Converting discrete tokens to continuous vectors
2. Positional encodings: Injecting position information into the model

Mathematical Foundation:
========================

1. TOKEN EMBEDDINGS
-------------------

Token embeddings are learned representations that map each token in the
vocabulary to a continuous vector space of dimension d_model.

E(token) ∈ ℝ^(d_model)

Where:
  - E is the embedding matrix of shape (vocab_size, d_model)
  - Each row corresponds to the embedding vector for that token
  - Embeddings are initialized randomly and learned during training

Embedding matrix: E ∈ ℝ^(vocab_size × d_model)
For token i, the embedding is: e_i = E[i, :]

2. POSITIONAL ENCODING
----------------------

Since the Transformer has no recurrence or convolution, it must inject
information about the relative or absolute position of the tokens in
the sequence. We use sinusoidal positional encodings.

The positional encoding is computed as:

PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Where:
  - pos ∈ {0, 1, 2, ...}: Position in the sequence
  - i ∈ {0, 1, ..., d_model/2 - 1}: Dimension index (halved because each
    position gets one sin and one cos)
  - d_model: Model dimension (assumed to be even)

Why sinusoidal encodings?

1. PERIODICITY: Different frequencies capture different positional patterns.
   Dimension 2i uses frequency 1/10000^(2i/d_model), creating a geometric
   progression of frequencies.

2. ABSOLUTE POSITION: The pattern PE(pos) uniquely identifies each position.

3. RELATIVE POSITION: The relative position encoding PE(pos+k) can be
   represented as a linear function of PE(pos):
   PE(pos+k) = f(PE(pos), k)
   
   This means the model can learn to attend based on relative distances.

4. EXTRAPOLATION: The model can potentially extrapolate to sequences longer
   than training sequences by learning the periodic pattern.

EXAMPLE:
--------

For d_model = 4, position = 0:
PE(0, 0) = sin(0 / 10000^(0/4)) = sin(0) = 0
PE(0, 1) = cos(0 / 10000^(0/4)) = cos(0) = 1
PE(0, 2) = sin(0 / 10000^(2/4)) = sin(0) = 0
PE(0, 3) = cos(0 / 10000^(2/4)) = cos(0) = 1

For position = 1:
PE(1, 0) = sin(1 / 10000^0) = sin(1) ≈ 0.841
PE(1, 1) = cos(1 / 10000^0) = cos(1) ≈ 0.540
PE(1, 2) = sin(1 / 10000^0.5) ≈ sin(0.01) ≈ 0.01
PE(1, 3) = cos(1 / 10000^0.5) ≈ cos(0.01) ≈ 1.0

The final input embeddings are: X = Embedding(token) + PE(position)
"""

import numpy as np


class TokenEmbedding:
    """
    Token embedding layer.
    
    Maps discrete token indices to continuous vector representations.
    
    Attributes:
        vocab_size: Number of tokens in vocabulary
        d_model: Dimension of embedding vectors
        embedding_matrix: The embedding lookup table
    """
    
    def __init__(self, vocab_size, d_model):
        """
        Initialize token embeddings.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Initialize embeddings with Xavier/Glorot initialization
        # E ~ N(0, 1/d_model)
        self.embedding_matrix = np.random.normal(
            0, 1.0 / np.sqrt(d_model), (vocab_size, d_model)
        )
    
    def embed(self, token_ids):
        """
        Embed token indices.
        
        Args:
            token_ids: Array of shape (batch_size, seq_len) or (seq_len,)
                      containing token indices
        
        Returns:
            embeddings: Array of shape (..., seq_len, d_model) containing
                       embedding vectors
        """
        return self.embedding_matrix[token_ids]


class PositionalEncoding:
    """
    Positional encoding using sinusoidal functions.
    
    Provides position information to the model through sine and cosine
    functions of different frequencies.
    
    Attributes:
        d_model: Model dimension
        max_seq_length: Maximum sequence length for precomputation
    """
    
    def __init__(self, d_model, max_seq_length=2048):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension (must be even)
            max_seq_length: Maximum sequence length to precompute
        """
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Precompute positional encodings
        self.pe = self._compute_positional_encodings(max_seq_length, d_model)
    
    def _compute_positional_encodings(self, max_len, d_model):
        """
        Compute sinusoidal positional encodings.
        
        Mathematical formulation:
        PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
        
        Returns:
            pe: Array of shape (max_len, d_model) containing positional encodings
        """
        pe = np.zeros((max_len, d_model))
        
        # pos: array of shape (max_len, 1) representing positions [0, max_len)
        pos = np.arange(0, max_len).reshape(-1, 1)  # Shape: (max_len, 1)
        
        # Compute the base: 10000^(2i / d_model)
        # i ranges from 0 to d_model/2 - 1
        # Create dimension indices: [0, 2, 4, ..., d_model-2]
        dim_indices = np.arange(0, d_model, 2)  # Shape: (d_model/2,)
        
        # angle_rates: 10000^(2i / d_model) for all i
        # Shape: (d_model/2,)
        angle_rates = 1.0 / np.power(10000, dim_indices / d_model)
        
        # Compute angles: pos / 10000^(2i / d_model)
        # pos shape: (max_len, 1), angle_rates shape: (d_model/2,)
        # Broadcasting gives shape: (max_len, d_model/2)
        angles = pos * angle_rates
        
        # Apply sine to even indices
        pe[:, 0::2] = np.sin(angles)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = np.cos(angles)
        
        return pe  # Shape: (max_len, d_model)
    
    def get_encoding(self, seq_length):
        """
        Get positional encodings for a sequence.
        
        Args:
            seq_length: Length of sequence
        
        Returns:
            encodings: Array of shape (seq_length, d_model)
        """
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds max length "
                f"{self.max_seq_length}"
            )
        return self.pe[:seq_length, :]
