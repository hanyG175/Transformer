"""
Attention Mechanisms

This module implements the core attention mechanisms that form the foundation
of the Transformer architecture.

Mathematical Foundation:
========================

1. SCALED DOT-PRODUCT ATTENTION
-------------------------------

The attention function maps a query (Q) and a set of key-value pairs (K, V)
to an output, where queries, keys, values, and outputs are all vectors.

The output is computed as a weighted sum of the values, where the weight
assigned to each value is determined by the compatibility function of the
query with the corresponding key.

Attention(Q, K, V) = softmax(Q K^T / √d_k) V

Where:
  - Q ∈ ℝ^(n × d_k): Query matrix, n queries each of dimension d_k
  - K ∈ ℝ^(m × d_k): Key matrix, m keys each of dimension d_k
  - V ∈ ℝ^(m × d_v): Value matrix, m values each of dimension d_v
  - d_k: Dimension of keys (scaled factor)
  - n, m: Sequence lengths

The scaling factor √d_k prevents the dot products from growing too large,
keeping gradients stable during backpropagation. Large dot products can cause
softmax to have an extremely sharp distribution with very small gradients.

2. MULTI-HEAD ATTENTION
-----------------------

Rather than performing single attention function, it's beneficial to linearly
project the queries, keys and values h times with different, learned linear
projections to d_k, d_k and d_v dimensions, respectively.

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Parameters:
  - W_i^Q ∈ ℝ^(d_model × d_k): Learned projection matrices for queries
  - W_i^K ∈ ℝ^(d_model × d_k): Learned projection matrices for keys
  - W_i^V ∈ ℝ^(d_model × d_v): Learned projection matrices for values
  - W^O ∈ ℝ^(h·d_v × d_model): Output projection matrix
  - h: Number of attention heads
  - d_model: Model dimension

Benefits of multi-head attention:
1. It allows the model to attend to information from different representation
   subspaces at different positions
2. Different heads can learn to focus on different parts of the input
3. It increases the expressiveness of the attention mechanism

MASKING
-------

For autoregressive decoding, we prevent positions from attending to subsequent
positions. The mask prevents information flow from future tokens. We set
attention weights to -∞ (or very large negative values) for positions we want
to mask out, before applying softmax.

Masked Attention(Q, K, V) = softmax((Q K^T / √d_k + M) / √d_k) V

Where M is a mask matrix with M_ij = 0 if attention allowed, -∞ otherwise.
