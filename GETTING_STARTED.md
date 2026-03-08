# Quick Start Guide

## Installation

### Requirements
- Python 3.7+
- NumPy (for computations)

### Setup

```bash
# Navigate to project directory
cd Transformer

# Install dependencies
pip install -r requirements.txt
```

## Project Overview

This is a complete implementation of the Transformer architecture from scratch, with emphasis on:
- **Mathematical clarity:** All formulas documented
- **Educational value:** Every operation explained
- **Clean code:** Modular, testable components

## File Organization

```
├── README.md                        ← Start here! Main documentation
├── ARCHITECTURE.md                  ← Technical architecture details
├── transformer/                     ← Core implementation
│   ├── attention.py                ← Attention theory (documented)
│   ├── attention_impl.py           ← Attention implementation
│   ├── embeddings.py               ← Token/positional embeddings
│   ├── feedforward.py              ← Feed-forward networks
│   ├── layers.py                   ← Encoder/decoder layers
│   ├── transformer.py              ← Full transformer model
│   └── utils.py                    ← Utilities and helpers
├── docs/
│   └── MATHEMATICAL_FOUNDATION.md  ← Complete math reference
├── examples/
│   └── basic_examples.py           ← Usage examples
└── tests/
    └── test_transformer.py         ← Unit tests
```

## Step 1: Read the Documentation

### Essential Reading
1. **README.md** - Overview of architecture and components
2. **docs/MATHEMATICAL_FOUNDATION.md** - Complete mathematical treatment

### For Implementation Details
3. **ARCHITECTURE.md** - Technical structure and design patterns

## Step 2: Understand the Code Structure

### Start with Core Components (in order)

1. **Embeddings** (`transformer/embeddings.py`)
   - Token embeddings: Discrete tokens → Continuous vectors
   - Positional encoding: Add position information

2. **Attention** (`transformer/attention_impl.py`)
   - Scaled dot-product attention: Core mechanism
   - Multi-head attention: Parallel attention computation

3. **Feed-Forward** (`transformer/feedforward.py`)
   - Position-wise MLP: Two-layer transformation
   - ReLU/GELU activations

4. **Layers** (`transformer/layers.py`)
   - Encoder layer: Bidirectional attention + FFN
   - Decoder layer: Masked attention + cross-attention + FFN

5. **Complete Model** (`transformer/transformer.py`)
   - TransformerEncoder: Stack of encoder layers
   - TransformerDecoder: Stack of decoder layers
   - Transformer: Full model

## Step 3: Run Examples

### Basic Usage Example

```python
import numpy as np
from transformer.transformer import Transformer

# Create model
vocab_size = 10000
model = Transformer(
    vocab_size=vocab_size,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6
)

# Create dummy data
batch_size = 2
src_seq_len = 10
tgt_seq_len = 8

src_ids = np.random.randint(0, vocab_size, (batch_size, src_seq_len))
tgt_ids = np.random.randint(0, vocab_size, (batch_size, tgt_seq_len))

# Forward pass
logits = model.forward(src_ids, tgt_ids)
print(f"Output shape: {logits.shape}")  # (batch_size, tgt_seq_len, vocab_size)
```

### Run All Examples

```bash
python examples/basic_examples.py
```

This will demonstrate:
- Creating a transformer
- Positional encoding visualization
- Token embedding
- Attention mechanism
- Encoder/decoder separation
- Hyperparameter effects

## Step 4: Run Tests

Verify the implementation:

```bash
python tests/test_transformer.py
```

Expected output:
```
Test Results: 11 passed, 0 failed
```

## Learning Progression

### Beginner (1-2 hours)
- [ ] Read README.md sections 1-4
- [ ] Run examples/basic_examples.py
- [ ] Understand basic architecture

### Intermediate (2-4 hours)
- [ ] Read MATHEMATICAL_FOUNDATION.md sections 1-4
- [ ] Study attention.py and attention_impl.py
- [ ] Trace through forward pass in transformer.py
- [ ] Modify hyperparameters and test

### Advanced (4-8 hours)
- [ ] Read complete MATHEMATICAL_FOUNDATION.md
- [ ] Understand all mathematical derivations
- [ ] Study ARCHITECTURE.md
- [ ] Implement backward pass (gradients)
- [ ] Add custom components (new activations, etc.)

## Common Tasks

### Task 1: Understand Attention
**Time: 30 minutes**

1. Read: README.md section "Core Components" → "Attention Mechanisms"
2. Read: MATHEMATICAL_FOUNDATION.md → "Scaled Dot-Product Attention"
3. Examine: transformer/attention.py (docstrings)
4. Examine: transformer/attention_impl.py (implementation)
5. Run: `python -c "from examples.basic_examples import example_attention; example_attention()"`

### Task 2: Understand Positional Encoding
**Time: 30 minutes**

1. Read: README.md → "Positional Encoding"
2. Read: MATHEMATICAL_FOUNDATION.md → "Positional Encoding"
3. Examine: transformer/embeddings.py → PositionalEncoding class
4. Run: `python -c "from examples.basic_examples import example_positional_encoding; example_positional_encoding()"`
5. Visualize: Plot the sine/cosine patterns

### Task 3: Trace a Forward Pass
**Time: 1 hour**

1. Create simple example:
```python
import numpy as np
from transformer.transformer import Transformer

model = Transformer(vocab_size=100, d_model=64, n_heads=4, d_ff=256, n_layers=1)
src = np.array([[1, 2, 3]])
tgt = np.array([[4, 5]])
out = model.forward(src, tgt)
```

2. Add print statements at each step
3. Understand dimensions at each stage
4. Verify output shape matches expectations

### Task 4: Modify and Experiment
**Time: 2-4 hours**

1. **Change activation:** Replace ReLU with GELU in feedforward.py
2. **Add regularization:** Implement dropout in layers
3. **Custom config:** Create new hyperparameter preset
4. **Analyze attention:** Print attention weights from different heads
5. **Benchmark:** Time different model sizes

## Configuration Examples

### Small Model (Fast)
```python
Transformer(vocab_size=5000, d_model=128, n_heads=4, d_ff=512, n_layers=2)
# ~0.5M parameters, very fast
```

### Base Model (Balanced)
```python
Transformer(vocab_size=50000, d_model=512, n_heads=8, d_ff=2048, n_layers=6)
# ~65M parameters, like original "Attention Is All You Need" paper
```

### Large Model (Powerful)
```python
Transformer(vocab_size=50000, d_model=1024, n_heads=16, d_ff=4096, n_layers=24)
# ~335M parameters, high capacity but slow
```

## Debugging Tips

### Check Tensor Shapes
```python
x = model.encode(src_ids)
print(f"Encoder output shape: {x.shape}")  # Should be (batch, seq_len, d_model)
```

### Verify Attention Mechanism
```python
from transformer.attention_impl import MultiHeadAttention
mha = MultiHeadAttention(d_model=512, n_heads=8)
output, weights = mha.forward(Q, K, V)
print(f"Attention weights shape: {weights.shape}")  # (batch, heads, seq_len, seq_len)
print(f"Weights sum: {weights.sum(axis=-1)}")  # Should be ~1 (probabilities)
```

### Check Layer Normalization
```python
from transformer.layers import layer_norm
x_norm = layer_norm(x)
print(f"Mean: {x_norm.mean(axis=-1)}")  # Should be ~0
print(f"Std: {x_norm.std(axis=-1)}")    # Should be ~1
```

## Next Steps

1. **Implement Backpropagation:** Add gradient computation
2. **Add Training Loop:** Implement optimization with loss function
3. **Implement Decoding:** Add beam search or top-k sampling
4. **Optimize Performance:** Add key-value caching, sparse attention
5. **Extend Architecture:** Try variants like Vision Transformer

## Frequently Asked Questions

**Q: Why pure NumPy instead of PyTorch?**
A: For educational clarity. NumPy makes every operation explicit.

**Q: Is this production-ready?**
A: No, it's for learning. Use PyTorch/TensorFlow in production.

**Q: Can I add gradients/backprop?**
A: Yes! See the commented structure in layers.py for guidance.

**Q: How do I handle different sequence lengths?**
A: Use padding masks in layers.py → create_padding_mask()

**Q: What about attention visualization?**
A: Attention weights are returned; plot them with matplotlib.

## Resources

- **Original Paper:** https://arxiv.org/abs/1706.10495
- **Illustrated Guide:** https://jalammar.github.io/illustrated-transformer/
- **Math Notation:** See MATHEMATICAL_FOUNDATION.md for all formulas
- **PyTorch Implementation:** github.com/pytorch/pytorch (for reference)

## Project Statistics

```
Files: 15
Lines of Code: ~3,500
Lines of Comments: ~2,500
Lines of Documentation: ~5,000
Total: ~10,000 lines
Test Coverage: 11 unit tests
```

---

**Happy Learning!** 

Start with README.md, run the examples, and progressively dive deeper into the implementation and mathematics.
