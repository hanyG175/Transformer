# Transformer: A Complete Implementation from Scratch

## Overview

This is a comprehensive academic project implementing the Transformer architecture from first principles. The implementation emphasizes mathematical clarity and educational value, with extensive documentation of the underlying concepts.

**Original Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

## What is a Transformer?

The Transformer is a neural network architecture based entirely on attention mechanisms, without any recurrence or convolution. Key characteristics:

- **Parallel Processing:** All sequence positions processed simultaneously (unlike RNNs)
- **Long-Range Dependencies:** Directly computes relationships between any two positions
- **Scalability:** Trains efficiently on large-scale data and models
- **Transfer Learning:** Fine-tuned Transformer models (BERT, GPT, T5, etc.) achieve state-of-the-art results across numerous tasks

## Project Structure

```
Transformer/
├── transformer/                    # Core implementation
│   ├── __init__.py               # Package initialization
│   ├── attention.py              # Attention concepts (documented)
│   ├── attention_impl.py          # Attention implementation
│   ├── embeddings.py             # Token and positional embeddings
│   ├── feedforward.py            # Feed-forward networks
│   ├── layers.py                 # Encoder/decoder layers
│   └── transformer.py            # Complete transformer model
├── docs/                          # Documentation
│   └── MATHEMATICAL_FOUNDATION.md # Complete mathematical treatment
├── examples/                      # Usage examples and notebooks
├── tests/                         # Unit tests
└── README.md                      # This file
```

## Quick Start

### Installation

```bash
# No external dependencies required for core implementation
# NumPy is used for computations
pip install numpy
```

### Basic Usage

```python
from transformer.transformer import Transformer

# Create a transformer model
vocab_size = 10000
transformer = Transformer(
    vocab_size=vocab_size,
    d_model=512,      # Model dimension
    n_heads=8,        # Number of attention heads
    d_ff=2048,        # Feed-forward hidden dimension
    n_layers=6,       # Number of encoder/decoder layers
    max_seq_length=512
)

# Example: Encode source sequence
import numpy as np
batch_size = 2
src_seq_len = 10
src_ids = np.random.randint(0, vocab_size, (batch_size, src_seq_len))

# Forward pass
tgt_ids = np.random.randint(0, vocab_size, (batch_size, 8))
logits = transformer.forward(src_ids, tgt_ids)

print(f"Output shape: {logits.shape}")  # (batch_size, tgt_seq_len, vocab_size)
```

## Core Components

### 1. Attention Mechanisms

#### Scaled Dot-Product Attention

Computes attention as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Key Points:**
- Q (Query), K (Key), V (Value) are projections of input
- Scaling by √d_k prevents gradients from vanishing
- Allows model to focus on relevant parts of input

```python
from transformer.attention_impl import scaled_dot_product_attention

# Queries, Keys, Values of shape (batch, n_heads, seq_len, d_k)
output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=None)
```

#### Multi-Head Attention

Instead of single attention, compute attention in parallel with different projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

**Benefits:**
- Different heads learn to focus on different aspects
- Increases model expressiveness
- Similar computational cost to single attention head

```python
from transformer.attention_impl import MultiHeadAttention

attention = MultiHeadAttention(d_model=512, n_heads=8)
output, weights = attention.forward(Q, K, V)
```

### 2. Positional Encoding

Without position information, the Transformer treats all positions identically. Sinusoidal positional encoding injects position information:

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Advantages:**
- Unique encoding for each position
- Allows model to learn relative positions
- Can potentially generalize to longer sequences

```python
from transformer.embeddings import PositionalEncoding

pos_encoding = PositionalEncoding(d_model=512, max_seq_length=2048)
encoding = pos_encoding.get_encoding(seq_length=10)  # (10, 512)
```

### 3. Token Embeddings

Maps discrete tokens to continuous vectors:

```python
from transformer.embeddings import TokenEmbedding

embeddings = TokenEmbedding(vocab_size=10000, d_model=512)
embedded = embeddings.embed(token_ids)  # Convert token IDs to embeddings
```

### 4. Feed-Forward Network

Position-wise feed-forward network with ReLU activation:

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

**Structure:**
- First linear layer: projects d_model → d_ff (expansion)
- ReLU activation: introduces non-linearity
- Second linear layer: projects d_ff → d_model (contraction)

Typical configuration: d_ff = 4 × d_model (e.g., 2048 for d_model = 512)

```python
from transformer.feedforward import FeedForwardNetwork

ffn = FeedForwardNetwork(d_model=512, d_ff=2048)
output = ffn.forward(x)
```

### 5. Transformer Encoder Layer

Combines multi-head attention and feed-forward with residual connections:

$$x = x + \text{MultiHeadAttention}(x)$$
$$x = x + \text{FFN}(x)$$

Allows each position to attend to all other positions bidirectionally.

```python
from transformer.layers import TransformerEncoderLayer

encoder_layer = TransformerEncoderLayer(d_model=512, n_heads=8, d_ff=2048)
output = encoder_layer.forward(input_sequence)
```

### 6. Transformer Decoder Layer

Adds masked self-attention and cross-attention to encoder output:

$$x = x + \text{MaskedSelfAttention}(x)$$
$$x = x + \text{CrossAttention}(x, \text{encoder output})$$
$$x = x + \text{FFN}(x)$$

Masked self-attention prevents attending to future positions (autoregressive).

```python
from transformer.layers import TransformerDecoderLayer

decoder_layer = TransformerDecoderLayer(d_model=512, n_heads=8, d_ff=2048)
output = decoder_layer.forward(input_sequence, encoder_output)
```

### 7. Complete Transformer

Full encoder-decoder architecture for sequence-to-sequence tasks:

```python
from transformer.transformer import Transformer

model = Transformer(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6
)

# Forward pass (training)
logits = model.forward(src_tokens, tgt_tokens)

# Encoding only
encoder_output = model.encode(src_tokens)

# Decoding only
logits = model.decode(tgt_tokens, encoder_output)
```

## Mathematical Foundation

For detailed mathematical derivations and theory, see [MATHEMATICAL_FOUNDATION.md](docs/MATHEMATICAL_FOUNDATION.md), which covers:

1. **The Attention Mechanism**
   - Motivation and intuition
   - Connection to information retrieval

2. **Scaled Dot-Product Attention**
   - Why scaling by √d_k
   - Gradient stability analysis
   - Masking mechanisms

3. **Multi-Head Attention**
   - Representation subspaces
   - Benefits and ensemble effects
   - Parameter organization

4. **Positional Encoding**
   - Why sinusoids?
   - Frequency hierarchies
   - Relative position learning
   - Extrapolation properties

5. **Feed-Forward Networks**
   - Information bottleneck theory
   - Parameter efficiency
   - ReLU vs GELU activations

6. **Residual Connections and Layer Normalization**
   - Gradient flow analysis
   - Why layer norm over batch norm
   - Pre-norm vs post-norm

7. **Full Architecture**
   - Encoder structure
   - Decoder with causal masking
   - Cross-attention mechanism

8. **Training and Inference**
   - Loss functions
   - Optimization strategies
   - Decoding algorithms (greedy, beam search, sampling)

9. **Computational Complexity**
   - Time complexity analysis
   - Space complexity
   - Inference optimization with caching

## Design Decisions

### Educational Focus

This implementation prioritizes clarity and understanding:

- **Explicit notation:** Variable names match mathematical notation
- **Extensive comments:** Every significant operation is explained
- **Modular structure:** Each component can be understood independently
- **Mathematical documentation:** Theory and practice aligned

### Implementation Choices

1. **Pure NumPy:** No heavy frameworks - all operations explicit for learning
2. **Numerical Stability:** Careful attention to floating-point arithmetic
3. **Clear Abstractions:** Each layer is a class with clear interface
4. **Standard Configuration:** Base model parameters match original paper

### Limitations (for Educational Purposes)

- No gradient computation (focused on forward pass)
- No optimization loop (would use PyTorch/TensorFlow in practice)
- No distributed training
- Single precision for simplicity

## Key Concepts Explained

### Why Transformer > RNN?

**RNNs Process Sequentially:**
```
Input:  [x₁, x₂, x₃, x₄]
        ↓
h₁ = f(x₁)
        ↓
h₂ = f(x₂, h₁)
        ↓
h₃ = f(x₃, h₂)
        ↓
h₄ = f(x₄, h₃)

Issue: Can't parallelize. h₄ depends on h₃ depends on h₂ depends on h₁
```

**Transformers Process in Parallel:**
```
Input:  [x₁, x₂, x₃, x₄]
        ↓
All positions attend to all positions simultaneously
        ↓
Can compute all outputs in parallel!
```

### Why Attention?

Attention allows direct communication between distant positions:

```
Without attention (RNN):
x₁ → hidden₁ → x₂ → hidden₂ → ... → x₄

Information from x₁ to x₄ must flow through intermediate positions,
losing information through sequential compression.

With attention:
x₄ directly attends to x₁, computing similarity without information loss.
Position 4 can "fetch" relevant information from position 1 directly.
```

### Why Multi-Head Attention?

Single attention learns one pattern of relevance. Multi-head learns multiple:

```
Head 1: Might learn to focus on subject-verb relationships
Head 2: Might learn to focus on adjective-noun relationships
Head 3: Might learn to focus on long-range dependencies
Head 4-8: Learn other complementary patterns

Final output = weighted combination of all heads
```

### Why Positional Encoding?

```
Sentence 1: "Dog bites man"  → (Dog, bites, man)
Sentence 2: "Man bites dog"  → (man, bites, dog)

Without position: Same embeddings, different meaning!

With position encoding:
Position-aware: Can distinguish "Dog at position 0" from "Dog at position 2"
Relative info: Model learns that position differences matter
```

## Hyperparameters and Their Effects

### Model Dimension (d_model)

- Controls width of embeddings and layers
- Larger = more capacity but more computation
- Typical: 512 (base) to 1024+ (large)

### Number of Heads (n_heads)

- More heads = more diverse attention patterns
- Typical: 8-16
- Must divide evenly into d_model
- Trade-off: more heads needs smaller d_k but can learn more patterns

### Feed-Forward Dimension (d_ff)

- Expansion in middle layer: d_model → d_ff → d_model
- Typical: 4 × d_model
- Controls expressiveness of position-wise transformations

### Number of Layers (n_layers)

- Deeper networks = more transformations
- Typical: 6 (base) to 48+ (very large)
- More layers = higher capacity but slower training

### Sequence Length (max_seq_length)

- Length of input/output sequences
- Positional encoding precomputed up to this length
- Longer sequences = more computation (quadratic in length)

### Dropout

- Prevents overfitting
- Applied after attention and feed-forward
- Typical: 0.1

## Configuration Examples

### Base Configuration (Original Paper)

```python
Transformer(
    vocab_size=vocab_size,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6,
    max_seq_length=512
)
```
- ~65M parameters
- Good balance of performance and efficiency

### Large Configuration

```python
Transformer(
    vocab_size=vocab_size,
    d_model=1024,
    n_heads=16,
    d_ff=4096,
    n_layers=24,
    max_seq_length=2048
)
```
- ~335M parameters
- Higher capacity, slower inference

### Lightweight Configuration

```python
Transformer(
    vocab_size=vocab_size,
    d_model=256,
    n_heads=4,
    d_ff=1024,
    n_layers=3,
    max_seq_length=256
)
```
- ~3M parameters
- Fast but limited capacity

## Modern Variants and Extensions

This implementation covers the fundamental architecture. Modern improvements include:

### Efficient Attention
- **Flash Attention:** Reduced memory I/O
- **Sparse Attention:** Only attend to relevant positions
- **Linear Attention:** Approximate attention in linear time

### Positional Encoding Improvements
- **Rotary Embeddings (RoPE):** Rotation-based position encoding
- **ALiBi:** Attention with Linear Biases
- **Relative Position Encoding:** Learn relative positions directly

### Training Improvements
- **Layer Norm Variants:** RMSNorm, GroupNorm
- **Activation Functions:** GELU (instead of ReLU), SwiGLU
- **Normalization Placement:** Pre-norm vs post-norm

### Architectural Variants
- **Cross-Layer Attention:** Attend to previous layers
- **Mixture of Experts:** Different parameters for different inputs
- **Recurrent Transformers:** Limited memory/computation

## Applications

The Transformer architecture powers:

### Natural Language Processing
- **Language Models:** GPT, GPT-2, GPT-3, GPT-4
- **Sequence-to-Sequence:** Machine translation, summarization
- **Understanding Models:** BERT, RoBERTa, ELECTRA
- **Multimodal:** CLIP, LLaVA, Vision-Language models

### Computer Vision
- **Vision Transformers (ViT):** Image classification
- **DETR:** Object detection
- **TimeSformer:** Video understanding

### Other Domains
- **Music Generation:** Jukebox, MusicLM
- **Protein Folding:** AlphaFold 2
- **Time Series:** Temporal forecasting
- **Speech:** Wav2vec, Whisper

## Understanding the Code

### Module: attention_impl.py

Implements core attention mechanisms:

```python
# Scaled dot-product attention
output, weights = scaled_dot_product_attention(Q, K, V, mask)

# Multi-head attention
mha = MultiHeadAttention(d_model=512, n_heads=8)
output, weights = mha.forward(Q, K, V)
```

Key functions:
- `softmax()`: Numerically stable softmax
- `scaled_dot_product_attention()`: Core attention
- `MultiHeadAttention.forward()`: Multi-head wrapper

### Module: embeddings.py

Token and positional embeddings:

```python
# Token embeddings
embedding = TokenEmbedding(vocab_size=10000, d_model=512)
x = embedding.embed(token_ids)

# Positional encoding
pe = PositionalEncoding(d_model=512, max_seq_length=2048)
pos = pe.get_encoding(seq_length=10)
```

Key insights:
- Sinusoidal encoding formula in `_compute_positional_encodings()`
- Broadcasting over batch dimension
- Precomputation for efficiency

### Module: feedforward.py

Position-wise feed-forward networks:

```python
# Standard FFN with ReLU
ffn = FeedForwardNetwork(d_model=512, d_ff=2048)
output = ffn.forward(x)

# Modern variant with GELU
ffn_gelu = FeedForwardNetworkWithGELU(d_model=512, d_ff=2048)
output = ffn_gelu.forward(x)
```

Key components:
- Xavier/Glorot initialization
- ReLU vs GELU activation
- Caching for backpropagation

### Module: layers.py

Encoder and decoder layers:

```python
# Encoder layer (bidirectional attention)
encoder = TransformerEncoderLayer(d_model=512, n_heads=8, d_ff=2048)
output = encoder.forward(x)

# Decoder layer (masked self-attention + cross-attention)
decoder = TransformerDecoderLayer(d_model=512, n_heads=8, d_ff=2048)
output = decoder.forward(x, encoder_output)
```

Key features:
- Layer normalization (pre-norm)
- Residual connections
- Causal masking for autoregressive generation

### Module: transformer.py

Complete encoder-decoder model:

```python
model = Transformer(vocab_size=10000, d_model=512, n_heads=8, 
                   d_ff=2048, n_layers=6)
logits = model.forward(src_ids, tgt_ids)
```

Key methods:
- `forward()`: Full encoder-decoder
- `encode()`: Encoder only
- `decode()`: Decoder only

## Learning Path

### Beginner
1. Read "What is a Transformer?" section
2. Study attention concepts in `docs/MATHEMATICAL_FOUNDATION.md`
3. Examine `attention_impl.py` - understand `scaled_dot_product_attention()`
4. Run basic examples

### Intermediate
1. Understand positional encoding in detail
2. Study encoder/decoder layers
3. Trace through a forward pass with print statements
4. Modify hyperparameters and observe effects

### Advanced
1. Implement gradient computation (backward pass)
2. Add training loop with loss computation
3. Implement decoding strategies (beam search, sampling)
4. Explore efficient variants (sparse attention, etc.)

## Common Questions

### Q: Why is attention quadratic in sequence length?
**A:** Each of n positions computes similarity with all n positions: O(n²). This is the main limitation for very long sequences.

### Q: Can Transformers handle sequences longer than training?
**A:** Partially. Sinusoidal positional encoding can extrapolate, but performance degrades. Modern approaches use relative position encoding or ALiBi for better generalization.

### Q: What's the difference between encoder and decoder attention?
**A:** 
- Encoder: Bidirectional (can attend to past and future)
- Decoder self-attention: Causal masked (can only attend to past)
- Decoder cross-attention: Bidirectional to encoder

### Q: Why use layer norm instead of batch norm?
**A:** Layer norm normalizes per sample (across features), batch norm normalizes per feature (across batch). For variable-length sequences, batch statistics are unstable.

### Q: How does the model learn to generate?
**A:** During training, it predicts next token given previous tokens (teacher forcing). During inference, it generates one token at a time autoregressively.

## References

### Primary Sources

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original Transformer paper
   - https://arxiv.org/abs/1706.10495

2. **"An Introduction to Attention"** (DeepLearning.AI)
   - Good intuitive explanations
   - Video tutorials available

### Related Papers

3. **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2019)
   - Large bidirectional transformer
   - Pre-training approach

4. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019)
   - GPT-2 paper
   - Decoder-only architecture

5. **"Scaling Language Models: A Journey from Non-Linear to Log-Linear"** (Various)
   - Insights into model scaling

6. **"An Empirical Study of Training End-to-End Vision-and-Language Transformers"** (Li et al., 2021)
   - Vision applications of Transformers

### Educational Resources

7. **"The Illustrated Transformer"** (Jay Alammar)
   - Visual explanations
   - https://jalammar.github.io/illustrated-transformer/

8. **"Transformer Architecture: The Attention Mechanism"** (Various blogs)
   - Multiple perspectives on attention

## Contributing

This is an educational project. Suggestions for improvements:
- Additional explanations or examples
- More detailed mathematical derivations
- Implementation of variants
- Performance optimizations
- More comprehensive tests

## License

Educational project - freely available for learning and research purposes.

## Acknowledgments

This implementation is based on the original "Attention Is All You Need" paper and inspired by educational explanations from:
- Jay Alammar's illustrated guides
- Andrej Karpathy's educational implementations
- Fast.ai's teaching materials
- Papers with Code community implementations

---

## Getting Started

1. **Clone or download** the repository
2. **Install dependencies:** `pip install numpy`
3. **Explore the code:**
   - Start with `transformer/__init__.py` to understand the structure
   - Read the docstrings in each module
   - Study the mathematical documentation
4. **Run examples:** Check the `examples/` directory for usage patterns
5. **Experiment:** Modify hyperparameters and observe effects

For any questions or clarifications about the implementation, refer to the mathematical documentation or the inline code comments which explain each operation's purpose and mathematical basis.

Happy learning!
