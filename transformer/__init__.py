"""
Transformer Architecture Implementation from Scratch

This package provides a complete implementation of the Transformer architecture
as described in "Attention Is All You Need" (Vaswani et al., 2017).

Core components:
- Attention mechanisms (Scaled Dot-Product Attention, Multi-Head Attention)
- Positional encodings
- Feedforward networks
- Transformer encoder and decoder stacks
"""

from .attention_impl import MultiHeadAttention
from .embeddings import TokenEmbedding, PositionalEncoding
from .feedforward import FeedForwardNetwork
from .layers import TransformerEncoderLayer, TransformerDecoderLayer
from .transformer import Transformer, TransformerEncoder, TransformerDecoder

__all__ = [
    'MultiHeadAttention',
    'TokenEmbedding',
    'PositionalEncoding',
    'FeedForwardNetwork',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'Transformer',
    'TransformerEncoder',
    'TransformerDecoder',
]
