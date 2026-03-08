"""
Unit tests for Transformer components.

This module contains tests to verify correctness of the implementation.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.attention_impl import scaled_dot_product_attention, MultiHeadAttention, softmax
from transformer.embeddings import PositionalEncoding, TokenEmbedding
from transformer.feedforward import FeedForwardNetwork
from transformer.layers import layer_norm, create_causal_mask, TransformerEncoderLayer, TransformerDecoderLayer
from transformer.transformer import Transformer


def test_softmax():
    """Test numerically stable softmax."""
    print("Testing softmax...")
    
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    result = softmax(x, axis=-1)
    
    # Check sums to 1
    assert np.allclose(np.sum(result, axis=-1), 1.0), "Softmax should sum to 1"
    
    # Check all positive
    assert np.all(result >= 0), "Softmax outputs should be non-negative"
    assert np.all(result <= 1), "Softmax outputs should be <= 1"
    
    print("  ✓ Softmax test passed")


def test_scaled_dot_product_attention():
    """Test scaled dot-product attention."""
    print("Testing scaled dot-product attention...")
    
    batch_size = 2
    seq_len = 4
    d_model = 64
    
    # Create dummy inputs
    Q = np.random.randn(batch_size, 8, seq_len, d_model)  # 8 heads
    K = np.random.randn(batch_size, 8, seq_len, d_model)
    V = np.random.randn(batch_size, 8, seq_len, d_model)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    # Check output shape
    assert output.shape == V.shape, f"Output shape {output.shape} != V shape {V.shape}"
    
    # Check attention weights
    assert weights.shape == (batch_size, 8, seq_len, seq_len), \
        f"Weights shape {weights.shape} incorrect"
    
    # Check weights are valid probabilities
    assert np.allclose(np.sum(weights, axis=-1), 1.0), \
        "Attention weights should sum to 1 along last dimension"
    
    print("  ✓ Scaled dot-product attention test passed")


def test_positional_encoding():
    """Test positional encoding."""
    print("Testing positional encoding...")
    
    d_model = 64
    max_seq = 512
    pe = PositionalEncoding(d_model, max_seq)
    
    # Get encoding
    seq_len = 10
    encoding = pe.get_encoding(seq_len)
    
    # Check shape
    assert encoding.shape == (seq_len, d_model), \
        f"Encoding shape {encoding.shape} incorrect"
    
    # Check bounded in [-1, 1] (sin/cos are bounded)
    assert np.all(encoding >= -1.01) and np.all(encoding <= 1.01), \
        "Positional encoding should be bounded in [-1, 1]"
    
    # Check different positions have different encodings
    assert not np.allclose(encoding[0], encoding[1]), \
        "Different positions should have different encodings"
    
    # Check that extrapolation works (at least doesn't crash)
    try:
        long_encoding = pe.get_encoding(512)
        assert long_encoding.shape == (512, d_model)
    except ValueError:
        pass  # Expected if seq_len > max_seq
    
    print("  ✓ Positional encoding test passed")


def test_token_embedding():
    """Test token embedding."""
    print("Testing token embedding...")
    
    vocab_size = 100
    d_model = 64
    embedding = TokenEmbedding(vocab_size, d_model)
    
    # Embed tokens
    token_ids = np.array([1, 5, 3, 5, 10])
    embedded = embedding.embed(token_ids)
    
    # Check shape
    assert embedded.shape == (5, d_model), f"Embedding shape incorrect"
    
    # Check same token gives same embedding
    assert np.allclose(embedded[1], embedded[3]), \
        "Same token should have identical embedding"
    
    # Check different tokens give different embeddings
    assert not np.allclose(embedded[0], embedded[1]), \
        "Different tokens should have different embeddings"
    
    print("  ✓ Token embedding test passed")


def test_layer_norm():
    """Test layer normalization."""
    print("Testing layer normalization...")
    
    x = np.random.randn(2, 4, 64)
    normalized = layer_norm(x)
    
    # Check shape preserved
    assert normalized.shape == x.shape
    
    # Check mean ~ 0 and std ~ 1 along last dimension
    mean = np.mean(normalized, axis=-1)
    std = np.std(normalized, axis=-1)
    
    assert np.allclose(mean, 0, atol=1e-5), "Normalized mean should be 0"
    assert np.allclose(std, 1, atol=1e-5), "Normalized std should be 1"
    
    print("  ✓ Layer normalization test passed")


def test_causal_mask():
    """Test causal mask creation."""
    print("Testing causal mask...")
    
    seq_len = 4
    mask = create_causal_mask(seq_len)
    
    # Check shape
    assert mask.shape == (seq_len, seq_len)
    
    # Check that position i can attend to position j only if j <= i
    # Lower triangle (including diagonal) should be 0 (allowed)
    # Upper triangle should be very negative (blocked)
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert mask[i, j] == 0, f"Position {i} should be able to attend to position {j}"
            else:
                assert mask[i, j] < -1e8, f"Position {i} should NOT attend to position {j}"
    
    print("  ✓ Causal mask test passed")


def test_feedforward():
    """Test feed-forward network."""
    print("Testing feed-forward network...")
    
    d_model = 64
    d_ff = 256
    ffn = FeedForwardNetwork(d_model, d_ff)
    
    # Test forward pass
    x = np.random.randn(2, 4, d_model)
    output = ffn.forward(x)
    
    # Check shape
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    
    print("  ✓ Feed-forward network test passed")


def test_multi_head_attention():
    """Test multi-head attention."""
    print("Testing multi-head attention...")
    
    d_model = 64
    n_heads = 4
    batch_size = 2
    seq_len = 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    output, weights = mha.forward(Q, K, V)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Output shape {output.shape} incorrect"
    
    # Check weights shape
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), \
        f"Weights shape {weights.shape} incorrect"
    
    print("  ✓ Multi-head attention test passed")


def test_encoder_layer():
    """Test encoder layer."""
    print("Testing encoder layer...")
    
    d_model = 64
    n_heads = 4
    d_ff = 256
    batch_size = 2
    seq_len = 8
    
    encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = encoder_layer.forward(x)
    
    # Check shape
    assert output.shape == x.shape, "Encoder layer should preserve shape"
    
    print("  ✓ Encoder layer test passed")


def test_decoder_layer():
    """Test decoder layer."""
    print("Testing decoder layer...")
    
    d_model = 64
    n_heads = 4
    d_ff = 256
    batch_size = 2
    tgt_seq_len = 6
    src_seq_len = 8
    
    decoder_layer = TransformerDecoderLayer(d_model, n_heads, d_ff)
    tgt = np.random.randn(batch_size, tgt_seq_len, d_model)
    encoder_output = np.random.randn(batch_size, src_seq_len, d_model)
    
    # Create causal mask
    causal_mask = create_causal_mask(tgt_seq_len)
    causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]
    
    output = decoder_layer.forward(tgt, encoder_output, self_attn_mask=causal_mask)
    
    # Check shape
    assert output.shape == tgt.shape, "Decoder layer should preserve target shape"
    
    print("  ✓ Decoder layer test passed")


def test_transformer():
    """Test complete transformer."""
    print("Testing complete transformer...")
    
    vocab_size = 1000
    d_model = 64
    n_heads = 4
    d_ff = 256
    n_layers = 2
    
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers
    )
    
    batch_size = 2
    src_seq_len = 8
    tgt_seq_len = 6
    
    src_ids = np.random.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt_ids = np.random.randint(0, vocab_size, (batch_size, tgt_seq_len))
    
    # Forward pass
    logits = transformer.forward(src_ids, tgt_ids)
    
    # Check shape
    assert logits.shape == (batch_size, tgt_seq_len, vocab_size), \
        f"Logits shape {logits.shape} incorrect"
    
    # Test encoding separately
    encoder_output = transformer.encode(src_ids)
    assert encoder_output.shape == (batch_size, src_seq_len, d_model)
    
    # Test decoding separately
    decoder_output = transformer.decode(tgt_ids, encoder_output)
    assert decoder_output.shape == (batch_size, tgt_seq_len, vocab_size)
    
    print("  ✓ Transformer test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Running Transformer Tests")
    print("=" * 50 + "\n")
    
    tests = [
        test_softmax,
        test_scaled_dot_product_attention,
        test_positional_encoding,
        test_token_embedding,
        test_layer_norm,
        test_causal_mask,
        test_feedforward,
        test_multi_head_attention,
        test_encoder_layer,
        test_decoder_layer,
        test_transformer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
