# Project Architecture Overview

## Project Structure

```
Transformer/
├── transformer/                          # Core implementation package
│   ├── __init__.py                      # Package exports
│   ├── attention.py                     # Attention theory and concepts
│   ├── attention_impl.py                # Attention implementation
│   ├── embeddings.py                    # Token and positional embeddings
│   ├── feedforward.py                   # Feed-forward networks
│   ├── layers.py                        # Encoder/decoder layers
│   ├── transformer.py                   # Complete transformer model
│   └── utils.py                         # Utility functions and helpers
│
├── docs/                                # Documentation
│   └── MATHEMATICAL_FOUNDATION.md       # Complete mathematical treatment
│
├── examples/                            # Usage examples
│   └── basic_examples.py               # Example scripts and tutorials
│
├── tests/                              # Test suite
│   └── test_transformer.py             # Unit tests
│
├── README.md                           # Main documentation
├── ARCHITECTURE.md                     # This file
└── requirements.txt                    # Dependencies
```

## Module Descriptions

### transformer/attention.py
- **Purpose:** Theory and mathematical documentation of attention mechanisms
- **Contains:** Comprehensive docstrings explaining attention concepts
- **Key Concepts:** Queries, keys, values, scaled dot-product, multi-head attention, masking

### transformer/attention_impl.py
- **Purpose:** Actual implementation of attention mechanisms using NumPy
- **Key Functions:**
  - `softmax()`: Numerically stable softmax
  - `scaled_dot_product_attention()`: Core attention mechanism
  - `MultiHeadAttention`: Multi-head attention wrapper class
- **Key Classes:** `MultiHeadAttention` for parallel attention computation

### transformer/embeddings.py
- **Purpose:** Convert discrete tokens to continuous vectors and add position information
- **Key Classes:**
  - `TokenEmbedding`: Maps token IDs to embedding vectors
  - `PositionalEncoding`: Injects position information using sinusoidal functions
- **Key Methods:**
  - `embed()`: Convert token IDs to embeddings
  - `get_encoding()`: Retrieve positional encodings for a sequence

### transformer/feedforward.py
- **Purpose:** Position-wise feed-forward networks
- **Key Classes:**
  - `FeedForwardNetwork`: Two-layer FFN with ReLU activation
  - `GELU`: Gaussian Error Linear Unit activation
  - `FeedForwardNetworkWithGELU`: Modern FFN variant
- **Architecture:** d_model → d_ff → d_model (expansion then contraction)

### transformer/layers.py
- **Purpose:** Individual encoder and decoder layers with residual connections
- **Key Functions:**
  - `layer_norm()`: Layer normalization
  - `create_causal_mask()`: Create autoregressive attention masks
  - `create_padding_mask()`: Handle variable-length sequences
- **Key Classes:**
  - `TransformerEncoderLayer`: Encoder layer (self-attention + FFN)
  - `TransformerDecoderLayer`: Decoder layer (self-attention + cross-attention + FFN)

### transformer/transformer.py
- **Purpose:** Complete encoder-decoder Transformer model
- **Key Classes:**
  - `TransformerEncoder`: Stack of encoder layers
  - `TransformerDecoder`: Stack of decoder layers
  - `Transformer`: Full sequence-to-sequence model
- **Key Methods:**
  - `forward()`: Full forward pass (encoder + decoder)
  - `encode()`: Encoder only
  - `decode()`: Decoder only

### transformer/utils.py
- **Purpose:** Utility functions and configuration helpers
- **Key Classes:**
  - `TransformerConfig`: Configuration management with presets
- **Key Functions:**
  - `softmax_temperature()`: Temperature scaling for generation
  - `top_k_sampling()`: Restrict to top-k tokens
  - `nucleus_sampling()`: Top-p sampling
  - `analyze_model_config()`: Print configuration analysis

## Data Flow

### Forward Pass (Training)

```
Source tokens → Embedding
                ↓
            Add positional encoding
                ↓
        [ENCODER LAYERS x N]
        - Self-attention
        - Feed-forward
        - Residual & norm
                ↓
        Encoder output (key-value cache)
                ↓
Target tokens → Embedding
                ↓
            Add positional encoding
                ↓
        [DECODER LAYERS x N]
        - Masked self-attention (autoregressive)
        - Cross-attention to encoder
        - Feed-forward
        - Residual & norm
                ↓
        Dense projection to vocab
                ↓
        Output logits (batch, tgt_len, vocab_size)
```

### Forward Pass (Inference)

```
Source tokens → [ENCODER] → Cached encoder output
                ↓
Decoder loop:
    Initialize with START token
    ↓
    For each position:
        Decode with encoder output
        Sample next token
        Add to sequence
        If END token, stop
        Else, continue
    ↓
Return generated sequence
```

## Key Design Patterns

### 1. Modular Architecture
- Each component is self-contained and independently testable
- Clear interfaces between modules
- Easy to replace or modify individual components

### 2. Mathematical Clarity
- Extensive docstrings with mathematical notation
- Variable names match paper notation (Q, K, V, d_model, etc.)
- Comments explain why each operation is performed

### 3. Educational Focus
- Pure NumPy implementation (no external frameworks)
- No gradient computation (focus on forward pass)
- Detailed comments in every module

### 4. Separation of Concerns
- `*_impl.py` modules contain implementation details
- Non-implementation files contain theory and documentation
- Tests verify correctness of each component

## Computational Complexity

### Time Complexity (per forward pass)
- **Attention:** O(n² × d_model) where n = sequence length
- **Feed-Forward:** O(n × d_model²) since d_ff = 4 × d_model
- **Per Layer:** O(n² × d_model) (attention dominates for typical n >> d_model)
- **Full Model:** O(N × L × n² × d_model) where N = num layers, L = num steps

### Space Complexity
- **Activations:** O(batch × n × d_model)
- **Attention Weights:** O(batch × heads × n²)
- **Parameters:** O(N × d_model²)

## Memory and Computation Estimates

### Base Model Configuration (d_model=512, N=6 layers)
- **Parameters:** ~37M (tokens) + ~24M (encoder) + ~24M (decoder) + output projection
- **Total:** ~65-70M parameters
- **Memory (FP32):** ~270MB parameters + activations

### Inference Optimization
- **Key-Value Caching:** Reduces attention from O(n²) to O(n) per token
- **Flash Attention:** Reduces memory I/O (2-4x speedup)
- **Quantization:** Reduce precision from FP32 to INT8/INT4

## Testing Strategy

### Unit Tests (tests/test_transformer.py)
1. Individual components (softmax, attention, embeddings)
2. Layer tests (encoder, decoder, feedforward)
3. Full model test

### Test Coverage
- Shape transformations
- Numerical stability
- Probability distributions
- Gradient-free correctness

### Running Tests
```bash
python tests/test_transformer.py
```

## Implementation Choices Explained

### 1. Pre-Layer Normalization (PreNorm)
```python
# Modern approach:
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```
Benefits: Better gradient flow, enables deeper models

### 2. Sinusoidal Positional Encoding
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Benefits: Unique per position, learns relative positions, generalizes to longer sequences

### 3. Scaled Dot-Product Attention
```python
Attention = softmax(QK^T / √d_k) V
```
Benefits: Numerical stability, prevents gradient saturation

### 4. Multi-Head Attention
```python
MultiHead = Concat(head_1, ..., head_h) W^O
```
Benefits: Multiple representation subspaces, ensemble effect

## Performance Characteristics

### Strengths
- ✓ Highly parallelizable (all positions computed simultaneously)
- ✓ Captures long-range dependencies directly
- ✓ Transfer learning via pre-training
- ✓ Scales to very large models and datasets

### Limitations
- ✗ Quadratic complexity in sequence length (attention bottleneck)
- ✗ Requires absolute position information (positional encoding)
- ✗ Struggles with very long sequences (>4K tokens in practice)
- ✗ Cannot easily handle dynamic sequence lengths

## Modern Improvements (Not Implemented)

1. **Flash Attention:** Reduce memory I/O through block-wise computation
2. **Sparse Attention:** Only attend to relevant positions
3. **Linear Attention:** Approximate attention in linear time
4. **Rotary Embeddings:** Better positional information encoding
5. **ALiBi:** Attention with Linear Biases (position via biases, not embeddings)
6. **GQA/MQA:** Grouped Query Attention (fewer heads for KV cache)

## How to Extend

### Add a New Activation Function
```python
# In feedforward.py
class NewActivation:
    @staticmethod
    def forward(x):
        return new_activation(x)

# Use in FeedForwardNetwork
hidden = NewActivation.forward(hidden)
```

### Add Sparse Attention
```python
# In attention_impl.py
def sparse_attention(Q, K, V, sparsity_pattern):
    # Mask attention to certain patterns
    mask = create_sparse_mask(sparsity_pattern)
    return scaled_dot_product_attention(Q, K, V, mask)
```

### Add Different Normalization
```python
# In layers.py
def group_norm(x, num_groups=32):
    # Group normalization variant
    ...
```

## References for Further Reading

1. **Original Paper:** "Attention Is All You Need" (Vaswani et al., 2017)
2. **Illustrated Guide:** "The Illustrated Transformer" (Jay Alammar)
3. **Vision Transformers:** "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
4. **Efficient Attention:** "FlashAttention: Fast and Memory-Efficient Exact Attention..." (Dao et al., 2022)
5. **Position Encodings:** Various papers on RoPE, ALiBi, T5 Bias

---

This architecture represents a clean, educational implementation of the Transformer suitable for learning and experimentation. For production use, frameworks like PyTorch or TensorFlow with optimized CUDA kernels are recommended.
