"""
Complete Transformer Architecture

This module implements the full Transformer model, composed of stacked
encoder and decoder layers.

TRANSFORMER ARCHITECTURE OVERVIEW
==================================

The Transformer is an encoder-decoder architecture:

INPUT → [ENCODER STACK] → ENCODER OUTPUT
         ↓
DECODER INPUT + ENCODER OUTPUT → [DECODER STACK] → OUTPUT

ENCODER
-------
- Stack of N identical encoder layers
- Each layer has multi-head attention + feed-forward
- Each input position can attend to all positions (no masking)
- Processes entire input sequence in parallel

DECODER
-------
- Stack of N identical decoder layers
- Each layer has masked self-attention + cross-attention + feed-forward
- Masked self-attention prevents attending to future positions
- Cross-attention allows attending to encoder output
- Processes output sequence one position at a time (autoregressive)

COMPLETE ARCHITECTURE FLOW
==========================

Encoder:
1. Input tokens → Token Embedding
2. Add Positional Encoding
3. Pass through N encoder layers
4. Output: Contextualized representations

Decoder (Training):
1. Target tokens → Token Embedding
2. Add Positional Encoding
3. For each of N decoder layers:
   a. Masked self-attention on decoder input
   b. Cross-attention to encoder output
   c. Feed-forward
4. Output: Next token logits (via final linear layer + softmax)

Decoder (Inference):
1. Start with single token (e.g., <START>)
2. Generate tokens one at a time
3. Each new token attends to all previously generated tokens
4. Stop when <END> token is generated

MATHEMATICAL NOTATION
=====================

Encoder forward pass:
x = embedding(tokens) + positional_encoding(positions)
for i in 1..N:
    x = encoder_layer_i(x)
encoder_output = x

Decoder forward pass (training):
y = embedding(target_tokens) + positional_encoding(positions)
for i in 1..N:
    y = decoder_layer_i(y, encoder_output)
output_logits = linear_projection(y)  # to vocab_size
predictions = softmax(output_logits)

Decoder forward pass (inference):
outputs = []
y = embedding([START_TOKEN]) + positional_encoding([0])
for t in 1..max_length:
    for i in 1..N:
        y = decoder_layer_i(y, encoder_output)
    logits = linear_projection(y[:, -1, :])  # Last token logits
    next_token = argmax(logits)
    outputs.append(next_token)
    if next_token == END_TOKEN:
        break
    y = concatenate(y, embedding([next_token]) + PE([t]))
return outputs

PARAMETERS & HYPERPARAMETERS
=============================

Standard configuration (base model):
- d_model = 512: Model dimension
- n_heads = 8: Number of attention heads
- d_ff = 2048: Feed-forward hidden dimension (4 * d_model)
- n_layers = 6: Number of encoder/decoder layers
- dropout = 0.1: Dropout rate
- vocab_size: Size of vocabulary
- max_seq_length = 2048: Maximum sequence length

Large configuration:
- d_model = 1024
- n_heads = 16
- d_ff = 4096
- n_layers = 24
- vocab_size = 50000

COMPUTATIONAL COMPLEXITY
========================

Per layer:
- Attention: O(n² * d_model) where n is sequence length
- Feed-forward: O(n * d_model * d_ff) = O(n * d_model²) since d_ff ≈ 4*d_model
- Total per layer: O(n² * d_model + n * d_model²)

For long sequences, attention dominates (quadratic in sequence length).

Full model (N layers):
- Time: O(N * (n² * d_model + n * d_model²))
- Space: O(N * n * d_model) for activations + weights

Key insight: Parallelizable across sequence positions in encoder,
but sequential in decoder (autoregressive).
"""

import numpy as np
from .embeddings import TokenEmbedding, PositionalEncoding
from .layers import TransformerEncoderLayer, TransformerDecoderLayer, create_causal_mask


class TransformerEncoder:
    """
    Transformer Encoder.
    
    Stack of encoder layers that process input sequence in parallel.
    
    Attributes:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        n_layers: Number of encoder layers
        embedding: Token embedding layer
        positional_encoding: Positional encoding layer
        layers: Stack of encoder layers
    """
    
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_length=2048):
        """
        Initialize Transformer Encoder.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_layers: Number of encoder layers
            max_seq_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack of encoder layers
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
    
    def forward(self, token_ids, mask=None):
        """
        Forward pass through encoder.
        
        Args:
            token_ids: Input tokens of shape (batch_size, seq_len)
            mask: Optional padding mask
        
        Returns:
            output: Encoded representations of shape (batch_size, seq_len, d_model)
        """
        # Get sequence length
        seq_len = token_ids.shape[1]
        
        # Embed tokens
        x = self.embedding.embed(token_ids)
        
        # Add positional encoding
        pe = self.positional_encoding.get_encoding(seq_len)
        x = x + pe[np.newaxis, :, :]  # Broadcast over batch dimension
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer.forward(x, mask)
        
        return x


class TransformerDecoder:
    """
    Transformer Decoder.
    
    Stack of decoder layers that process output sequence autoregressively.
    
    Attributes:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        n_layers: Number of decoder layers
        embedding: Token embedding layer
        positional_encoding: Positional encoding layer
        layers: Stack of decoder layers
        output_projection: Linear projection to vocabulary size
    """
    
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_length=2048):
        """
        Initialize Transformer Decoder.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            n_layers: Number of decoder layers
            max_seq_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack of decoder layers
        self.layers = [
            TransformerDecoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        
        # Output projection to vocabulary
        limit = np.sqrt(6.0 / (d_model + vocab_size))
        self.output_projection = np.random.uniform(
            -limit, limit, (d_model, vocab_size)
        )
    
    def forward(self, token_ids, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Forward pass through decoder.
        
        Args:
            token_ids: Target tokens of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Padding mask for cross-attention
        
        Returns:
            logits: Output logits of shape (batch_size, tgt_seq_len, vocab_size)
        """
        # Get sequence length
        seq_len = token_ids.shape[1]
        
        # Embed tokens
        x = self.embedding.embed(token_ids)
        
        # Add positional encoding
        pe = self.positional_encoding.get_encoding(seq_len)
        x = x + pe[np.newaxis, :, :]
        
        # Create causal mask if not provided
        if self_attn_mask is None:
            self_attn_mask = create_causal_mask(seq_len)
            self_attn_mask = self_attn_mask[np.newaxis, np.newaxis, :, :]  # Broadcast
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer.forward(x, encoder_output, self_attn_mask, cross_attn_mask)
        
        # Project to vocabulary
        logits = np.matmul(x, self.output_projection)
        
        return logits


class Transformer:
    """
    Complete Transformer model (Encoder-Decoder).
    
    The full sequence-to-sequence Transformer architecture as described
    in "Attention Is All You Need" (Vaswani et al., 2017).
    
    Typical applications:
    - Neural Machine Translation (NMT)
    - Abstractive Summarization
    - Question Answering
    - Any sequence-to-sequence task
    
    Attributes:
        encoder: Transformer encoder
        decoder: Transformer decoder
        vocab_size: Size of vocabulary
    """
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, 
                 n_layers=6, max_seq_length=2048):
        """
        Initialize Transformer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (default: 512 for base model)
            n_heads: Number of attention heads (default: 8)
            d_ff: Feed-forward hidden dimension (default: 2048 = 4 * d_model)
            n_layers: Number of encoder/decoder layers (default: 6)
            max_seq_length: Maximum sequence length (default: 2048)
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_length
        )
        
        self.decoder = TransformerDecoder(
            vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_length
        )
    
    def forward(self, src_ids, tgt_ids):
        """
        Forward pass through Transformer.
        
        Args:
            src_ids: Source tokens of shape (batch_size, src_seq_len)
            tgt_ids: Target tokens of shape (batch_size, tgt_seq_len)
        
        Returns:
            logits: Output logits of shape (batch_size, tgt_seq_len, vocab_size)
        """
        # Encode source
        encoder_output = self.encoder.forward(src_ids)
        
        # Decode with encoder output
        logits = self.decoder.forward(tgt_ids, encoder_output)
        
        return logits
    
    def encode(self, src_ids):
        """
        Just encode the source sequence.
        
        Useful for inspection or reuse.
        
        Args:
            src_ids: Source tokens of shape (batch_size, src_seq_len)
        
        Returns:
            encoder_output: Encoded representations of shape 
                           (batch_size, src_seq_len, d_model)
        """
        return self.encoder.forward(src_ids)
    
    def decode(self, tgt_ids, encoder_output):
        """
        Just decode given encoder output.
        
        Useful for autoregressive generation.
        
        Args:
            tgt_ids: Target tokens of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
        
        Returns:
            logits: Output logits of shape (batch_size, tgt_seq_len, vocab_size)
        """
        return self.decoder.forward(tgt_ids, encoder_output)
