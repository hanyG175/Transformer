"""
Example: Basic Transformer Usage

This module demonstrates how to use the Transformer implementation
for a simple sequence-to-sequence task.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.transformer import Transformer
from transformer.embeddings import PositionalEncoding, TokenEmbedding
from transformer.attention_impl import MultiHeadAttention, scaled_dot_product_attention


def example_basic_transformer():
    """
    Basic example: Creating and using a Transformer model.
    """
    print("=" * 60)
    print("Example 1: Basic Transformer Creation and Forward Pass")
    print("=" * 60)
    
    # Create a small transformer for demonstration
    vocab_size = 1000
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=256,      # Smaller for demo
        n_heads=4,        # Fewer heads for demo
        d_ff=1024,        # Smaller FFN
        n_layers=2,       # Fewer layers for demo
        max_seq_length=512
    )
    
    # Create dummy data
    batch_size = 2
    src_seq_len = 8
    tgt_seq_len = 6
    
    src_ids = np.random.randint(1, vocab_size, (batch_size, src_seq_len))
    tgt_ids = np.random.randint(1, vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\nInput shapes:")
    print(f"  Source: {src_ids.shape}")
    print(f"  Target: {tgt_ids.shape}")
    
    # Forward pass
    logits = transformer.forward(src_ids, tgt_ids)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"  (batch_size={batch_size}, tgt_seq_len={tgt_seq_len}, vocab_size={vocab_size})")
    
    # Interpret output
    # logits[i, j, :] are the logits for predicting token at position j in batch i
    next_token_logits = logits[0, 0, :]
    predicted_token = np.argmax(next_token_logits)
    
    print(f"\nFor first batch, first position:")
    print(f"  Predicted token ID: {predicted_token}")
    print(f"  Logit at predicted token: {next_token_logits[predicted_token]:.4f}")


def example_positional_encoding():
    """
    Demonstrate positional encoding - how position information is injected.
    """
    print("\n" + "=" * 60)
    print("Example 2: Positional Encoding Visualization")
    print("=" * 60)
    
    d_model = 64
    pos_encoding = PositionalEncoding(d_model=d_model, max_seq_length=512)
    
    # Get encodings for first 4 positions
    seq_len = 4
    encodings = pos_encoding.get_encoding(seq_len)
    
    print(f"\nPositional encodings (d_model={d_model}, seq_len={seq_len}):")
    print(f"Shape: {encodings.shape}")
    
    # Show first encoding
    print(f"\nPosition 0 encoding (first 8 dims):")
    print(f"  {encodings[0, :8]}")
    
    print(f"\nPosition 1 encoding (first 8 dims):")
    print(f"  {encodings[1, :8]}")
    
    # Note: Different positions have different encodings
    # The model learns to interpret these differences as position information
    
    # Analyze frequency content
    print(f"\nAnalyzing frequency content:")
    print(f"  Dimension 0-1 (highest freq): varies rapidly across positions")
    print(f"  Dimension 2-3: varies slowly across positions")
    print(f"  Dimension 62-63 (lowest freq): varies very slowly")
    print(f"\nThis creates a hierarchical position representation.")


def example_token_embedding():
    """
    Demonstrate token embedding - mapping discrete tokens to vectors.
    """
    print("\n" + "=" * 60)
    print("Example 3: Token Embedding")
    print("=" * 60)
    
    vocab_size = 100
    d_model = 64
    
    embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    
    # Embed a sequence
    token_ids = np.array([1, 5, 3, 5, 2])  # Note: token 5 appears twice
    embeddings = embedding.embed(token_ids)
    
    print(f"\nToken embedding:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {d_model}")
    print(f"  Token IDs: {token_ids}")
    print(f"  Output shape: {embeddings.shape}")
    
    # Show that same token gets same embedding
    print(f"\nToken 5 appears at positions 1 and 3:")
    print(f"  Distance between embeddings: {np.linalg.norm(embeddings[1] - embeddings[3]):.6f}")
    print(f"  (Should be ~0 since embeddings are identical)")
    
    # Show that different tokens get different embeddings
    print(f"\nToken 1 vs Token 2:")
    print(f"  Distance between embeddings: {np.linalg.norm(embeddings[0] - embeddings[4]):.6f}")
    print(f"  (Should be > 0 since tokens are different)")


def example_attention():
    """
    Demonstrate attention mechanism - how models focus on relevant parts.
    """
    print("\n" + "=" * 60)
    print("Example 4: Attention Mechanism")
    print("=" * 60)
    
    # Create attention for demonstration
    d_model = 64
    n_heads = 4
    attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    
    # Create dummy queries, keys, values
    batch_size = 1
    seq_len = 3
    
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    # Apply attention
    output, attn_weights = attention.forward(Q, K, V)
    
    print(f"\nAttention computation:")
    print(f"  Query shape: {Q.shape}")
    print(f"  Key shape: {K.shape}")
    print(f"  Value shape: {V.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    
    # Analyze attention weights
    print(f"\nAttention weight analysis:")
    print(f"  Number of heads: {n_heads}")
    print(f"  Weights for each head shape: {attn_weights.shape}")
    
    # Show first head's weights
    print(f"\nFirst head attention weights (showing what each position attends to):")
    head_weights = attn_weights[0, 0, :, :]  # batch 0, head 0
    print(f"  Position 0 attends to positions with weights: {head_weights[0, :]}")
    print(f"  Position 1 attends to positions with weights: {head_weights[1, :]}")
    print(f"  Position 2 attends to positions with weights: {head_weights[2, :]}")
    print(f"  (Each row sums to 1 - these are probability distributions)")


def example_encoder_decoder_separation():
    """
    Show how encoder and decoder can be used separately.
    """
    print("\n" + "=" * 60)
    print("Example 5: Encoder and Decoder Separation")
    print("=" * 60)
    
    vocab_size = 500
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=2
    )
    
    # Example: Machine Translation
    # Encoder processes source language
    src_ids = np.array([[10, 20, 30, 5]])  # "Hello world" in source language
    encoder_output = transformer.encode(src_ids)
    
    print(f"\nEncoder stage:")
    print(f"  Input shape: {src_ids.shape}")
    print(f"  Output shape: {encoder_output.shape}")
    print(f"  The encoder has processed the source sequence into")
    print(f"  a rich representation that captures its meaning.")
    
    # Decoder generates target language from encoder output
    tgt_ids = np.array([[1, 2, 3, 5]])  # "Bonjour le monde" in target language
    decoder_logits = transformer.decode(tgt_ids, encoder_output)
    
    print(f"\nDecoder stage:")
    print(f"  Target input shape: {tgt_ids.shape}")
    print(f"  Encoder output shape: {encoder_output.shape}")
    print(f"  Decoder output shape: {decoder_logits.shape}")
    print(f"  The decoder cross-attends to the encoder output,")
    print(f"  learning to generate the target language.")


def example_hyperparameter_effects():
    """
    Show how different hyperparameters affect the model.
    """
    print("\n" + "=" * 60)
    print("Example 6: Hyperparameter Effects")
    print("=" * 60)
    
    vocab_size = 1000
    
    configs = [
        ("Small", dict(d_model=128, n_heads=2, d_ff=512, n_layers=2)),
        ("Base", dict(d_model=256, n_heads=4, d_ff=1024, n_layers=3)),
        ("Large", dict(d_model=512, n_heads=8, d_ff=2048, n_layers=6)),
    ]
    
    print(f"\nModel configurations and their parameter counts:")
    print(f"{'Configuration':<15} {'d_model':<10} {'n_heads':<10} {'n_layers':<10} {'Est. Params':<15}")
    print("-" * 60)
    
    for name, config in configs:
        d_model = config['d_model']
        n_heads = config['n_heads']
        n_layers = config['n_layers']
        d_ff = config['d_ff']
        
        # Rough parameter count (per layer: 4*d_model^2 for attention + 8*d_model^2 for FFN)
        params_per_layer = 4 * d_model**2 + 8 * d_model**2
        total_params = n_layers * 2 * params_per_layer  # encoder + decoder
        
        print(f"{name:<15} {d_model:<10} {n_heads:<10} {n_layers:<10} {total_params/1e6:>6.1f}M")
    
    print(f"\nTrade-offs:")
    print(f"  Larger d_model: More capacity, slower inference")
    print(f"  More heads: Better specialization, same computation")
    print(f"  More layers: Better performance, slower training and inference")
    print(f"  Larger d_ff: More expressive feed-forward, slower")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRANSFORMER EXAMPLES")
    print("=" * 60)
    
    example_basic_transformer()
    example_positional_encoding()
    example_token_embedding()
    example_attention()
    example_encoder_decoder_separation()
    example_hyperparameter_effects()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
