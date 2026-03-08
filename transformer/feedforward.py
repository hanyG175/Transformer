"""
Feed-Forward Networks in Transformers

This module implements the feed-forward sublayer used in Transformer encoder
and decoder stacks.

Mathematical Foundation:
========================

POSITION-WISE FEED-FORWARD NETWORK
-----------------------------------

Each encoder and decoder layer contains a fully connected feed-forward
network, which is applied to each position separately and identically.
This consists of two linear transformations with a ReLU activation in between.

FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

or equivalently:

FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

Dimensions:
-----------
Input: (batch_size, seq_len, d_model)

First linear layer:
- W_1 ∈ ℝ^(d_model × d_ff)
- b_1 ∈ ℝ^d_ff
- Output: (batch_size, seq_len, d_ff)

ReLU activation: max(0, x)
- Introduces non-linearity
- Prevents gradient vanishing in deeper networks

Second linear layer:
- W_2 ∈ ℝ^(d_ff × d_model)
- b_2 ∈ ℝ^d_model
- Output: (batch_size, seq_len, d_model)

WHY TWO LAYERS WITH EXPANSION?
------------------------------

1. EXPRESSIVE POWER:
   The first layer projects to higher dimension (d_ff >> d_model),
   then projects back down. This creates a bottleneck that allows for
   rich non-linear transformations.

2. PARAMETER EFFICIENCY:
   While seemingly wasteful, using wide intermediate layers is more
   efficient than alternative architectures of comparable capacity.

3. RECTIFIER NETWORKS:
   ReLU allows information to flow selectively - some neurons activate
   while others remain silent. The two-layer structure with ReLU is
   fundamental to modern deep learning.

TYPICAL PARAMETERS:
- d_model = 512 (model dimension)
- d_ff = 2048 (feed-forward hidden dimension)
- Ratio d_ff / d_model = 4

This is sometimes called the "width expansion factor" and is typically
between 2-4 for transformer models.

COMPUTATIONAL COST:
------------------
The feed-forward network typically accounts for ~2/3 of the total
parameters and computation in a Transformer, while attention accounts
for the remaining ~1/3. Despite being parameter-heavy, it's fast since
each position is processed independently (parallelizable).

GATING AND VARIANTS:
--------------------
Modern variants include:
- GELU activation: Gaussian Error Linear Unit
  GELU(x) = x · Φ(x), where Φ is the cumulative distribution function
  of standard normal distribution
  
- SwiGLU: Swish-Gated Linear Unit
  FFN(x) = (x W_1 ⊗ SiLU(x W_3)) W_2
  Uses element-wise multiplication with gating
  
- MLP-Mixer style: Different position-wise and channel-wise mixing

The original Transformer uses ReLU, but GELU often performs better in
modern applications.
"""

import numpy as np


class FeedForwardNetwork:
    """
    Position-wise Feed-Forward Network.
    
    Applies the same feed-forward network to each position separately
    and identically. This consists of two linear transformations with
    ReLU activation in between.
    
    FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
    
    Attributes:
        d_model: Model dimension
        d_ff: Hidden feed-forward dimension
        W_1: First weight matrix
        b_1: First bias vector
        W_2: Second weight matrix
        b_2: Second bias vector
    """
    
    def __init__(self, d_model, d_ff):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension (typically 2-4x d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights using Xavier/Glorot initialization
        # W_1: (d_model, d_ff)
        self.W_1 = self._xavier_init(d_model, d_ff)
        self.b_1 = np.zeros(d_ff)
        
        # W_2: (d_ff, d_model)
        self.W_2 = self._xavier_init(d_ff, d_model)
        self.b_2 = np.zeros(d_model)
    
    def _xavier_init(self, in_dim, out_dim):
        """
        Xavier/Glorot uniform initialization.
        
        For weight matrices, initialize uniformly in:
        [-√(6/(in+out)), √(6/(in+out))]
        
        This helps maintain stable gradient magnitudes across layers.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
        
        Returns:
            weight: Initialized weight matrix
        """
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, (in_dim, out_dim))
    
    def forward(self, x):
        """
        Forward pass.
        
        Applies FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Output of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation: (batch, seq, d_model) -> (batch, seq, d_ff)
        # x W_1 + b_1
        hidden = np.matmul(x, self.W_1) + self.b_1
        
        # ReLU activation
        # hidden = max(0, hidden)
        hidden = np.maximum(0, hidden)  # ReLU: max(0, x)
        
        # Second linear transformation: (batch, seq, d_ff) -> (batch, seq, d_model)
        # hidden W_2 + b_2
        output = np.matmul(hidden, self.W_2) + self.b_2
        
        return output
    
    def forward_with_cache(self, x):
        """
        Forward pass with intermediate values cached for backprop.
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Output of shape (batch_size, seq_len, d_model)
            cache: Tuple of (x, hidden) for use in backward pass
        """
        # First linear transformation
        hidden = np.matmul(x, self.W_1) + self.b_1
        
        # ReLU activation
        hidden_relu = np.maximum(0, hidden)
        
        # Second linear transformation
        output = np.matmul(hidden_relu, self.W_2) + self.b_2
        
        cache = (x, hidden, hidden_relu)
        return output, cache


class GELU:
    """
    Gaussian Error Linear Unit.
    
    GELU(x) = x · Φ(x)
    
    where Φ(x) is the cumulative distribution function of the
    standard normal distribution.
    
    Approximation (used for efficiency):
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    The GELU function is smoother than ReLU and often yields better
    performance in modern deep learning models.
    
    Properties:
    - Non-linear and smooth
    - Allows gradient flow through more of the input range
    - Probabilistic interpretation: GELU(x) is the expected value
      of x when it's filtered by whether it's greater than gaussian noise
    """
    
    @staticmethod
    def forward(x):
        """
        Forward pass through GELU.
        
        Args:
            x: Input array
        
        Returns:
            output: GELU(x)
        """
        # Use the approximation for computational efficiency
        cdf = 0.5 * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
        ))
        return x * cdf


class FeedForwardNetworkWithGELU:
    """
    Feed-Forward Network with GELU activation.
    
    Modern variant that often performs better than ReLU.
    
    Attributes:
        d_model: Model dimension
        d_ff: Hidden feed-forward dimension
        W_1: First weight matrix
        b_1: First bias vector
        W_2: Second weight matrix
        b_2: Second bias vector
    """
    
    def __init__(self, d_model, d_ff):
        """
        Initialize feed-forward network with GELU.
        
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.W_1 = self._xavier_init(d_model, d_ff)
        self.b_1 = np.zeros(d_ff)
        
        self.W_2 = self._xavier_init(d_ff, d_model)
        self.b_2 = np.zeros(d_model)
    
    def _xavier_init(self, in_dim, out_dim):
        """Xavier/Glorot initialization."""
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        return np.random.uniform(-limit, limit, (in_dim, out_dim))
    
    def forward(self, x):
        """
        Forward pass with GELU activation.
        
        FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Output of shape (batch_size, seq_len, d_model)
        """
        hidden = np.matmul(x, self.W_1) + self.b_1
        hidden = GELU.forward(hidden)
        output = np.matmul(hidden, self.W_2) + self.b_2
        return output
