"""
micro-instruct: Transformer Model from Scratch
==============================================

We'll build a small transformer optimized for instruction-following.

ARCHITECTURE DECISIONS:
----------------------
Base: Standard transformer (you've seen this in micro-transformer)
Improvements: RoPE, RMSNorm, SwiGLU (we'll explain WHY for each)

WHY THESE IMPROVEMENTS?
-----------------------
These are used in modern LLMs (LLaMA, GPT-4, Claude) because they:
1. Train faster
2. Use less memory
3. Work better for small models

We'll implement each from scratch and explain the intuition.
"""

import numpy as np

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def init_weights(rows, cols):
    """
    Initialize weights with proper scaling (Xavier/Glorot initialization).

    WHY DOES INITIALIZATION MATTER?
    -------------------------------
    Bad initialization → gradients explode or vanish → training fails

    Xavier initialization: scale by sqrt(1 / fan_in)
    - Keeps variance stable across layers
    - fan_in = number of inputs

    Example: Layer with 768 inputs
    - Random values from Normal(0, 1) → variance = 1 (too large!)
    - Scale by sqrt(1/768) ≈ 0.036 → variance = 0.036 (just right!)

    This ensures gradients flow smoothly through deep networks.
    """
    scale = np.sqrt(1.0 / cols)
    return np.random.randn(rows, cols).astype(np.float32) * scale


def softmax(x):
    """
    Softmax: Convert numbers to probabilities.

    WHY SOFTMAX?
    -----------
    We need to convert model outputs (logits) to probabilities:
    - Probabilities must sum to 1
    - Probabilities must be positive

    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    NUMERICAL STABILITY TRICK:
    -------------------------
    exp() can overflow for large values!
    exp(1000) → infinity → NaN

    Solution: Subtract max before exp
    softmax(x - max(x)) gives same result but prevents overflow

    Why does this work?
    exp(x - c) / sum(exp(x - c)) = exp(x) / sum(exp(x))  [c cancels out]
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=-1, keepdims=True)

    # Exponentiate
    exp_x = np.exp(x_shifted)

    # Normalize
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def gelu(x):
    """
    GELU activation: Gaussian Error Linear Unit

    WHY GELU INSTEAD OF RELU?
    -------------------------
    ReLU: f(x) = max(0, x)
    - Simple but harsh (kills negative values completely)

    GELU: Smooth version of ReLU
    - Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    - Smoothly gates values instead of hard cutoff
    - Used in GPT-2, GPT-3, BERT

    Intuition: "Probabilistic gating" - values are gated by their probability
               under a standard normal distribution
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# =============================================================================
# MODERN IMPROVEMENTS (Explained!)
# =============================================================================

class RMSNorm:
    """
    Root Mean Square Layer Normalization

    WHY RMSNORM INSTEAD OF LAYERNORM?
    ---------------------------------
    LayerNorm (standard):
    1. Calculate mean
    2. Calculate variance
    3. Normalize: (x - mean) / sqrt(var + eps)
    4. Scale and shift: y * gamma + beta

    RMSNorm (simpler):
    1. Calculate RMS (root mean square)
    2. Normalize: x / RMS
    3. Scale: y * gamma (no shift!)

    BENEFITS:
    - 30-40% faster (no mean calculation, no shift parameter)
    - Uses less memory (half the parameters)
    - Works just as well in practice!

    Used in: LLaMA, GPT-J, PaLM

    INTUITION:
    ---------
    LayerNorm re-centers and re-scales
    RMSNorm only re-scales (assumes data is already centered)

    For transformers, re-centering isn't necessary - just re-scaling works!
    """

    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim: Hidden dimension size
            eps: Small constant for numerical stability
        """
        self.dim = dim
        self.eps = eps

        # Learned scale parameter (gamma)
        # Initialized to 1 (identity function at start)
        self.gamma = np.ones(dim, dtype=np.float32)

    def forward(self, x):
        """
        Normalize input.

        Args:
            x: Shape (batch, seq_len, dim)

        Returns:
            Normalized output: Same shape as x
        """
        # Calculate RMS (Root Mean Square)
        # RMS = sqrt(mean(x^2))
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)

        # Normalize by RMS
        x_norm = x / rms

        # Scale by learned parameter
        return self.gamma * x_norm


def apply_rotary_emb(x, cos, sin):
    """
    Apply Rotary Position Embeddings (RoPE)

    WHY ROPE INSTEAD OF LEARNED POSITIONAL EMBEDDINGS?
    --------------------------------------------------
    Traditional approach (GPT-2, BERT):
    - Learn a position embedding for each position
    - Add to input: x + pos_emb[position]

    Problems:
    1. Fixed maximum length (e.g., 512 tokens)
    2. Doesn't generalize to longer sequences
    3. Position is additive (not ideal)

    RoPE (Rotary Position Embedding):
    - Encodes position by ROTATING the embedding vectors
    - Uses rotation matrices (trigonometric functions)
    - Naturally extends to any length!

    INTUITION:
    ---------
    Think of positions as angles on a clock:
    - Position 0: 0 degrees
    - Position 1: 30 degrees
    - Position 2: 60 degrees
    - etc.

    Rotating a vector by different angles for different positions
    encodes position information geometrically!

    BENEFITS:
    - Works for ANY sequence length (extrapolates naturally)
    - Relative positions are preserved (rotation is geometric)
    - Used in: LLaMA, GPT-NeoX, PaLM

    MATHEMATICS (simplified):
    ------------------------
    For each position t, create rotation matrix using:
    - cos(t * θ) and sin(t * θ)
    - θ depends on dimension (different frequencies for different dims)

    Apply rotation to query and key vectors (not values!)

    Args:
        x: Query or Key tensor, shape (batch, seq_len, n_heads, head_dim)
        cos: Cosine values, shape (seq_len, head_dim)
        sin: Sine values, shape (seq_len, head_dim)

    Returns:
        Rotated tensor: Same shape as x
    """
    # Reshape cos/sin for broadcasting
    # x shape: (batch, seq_len, n_heads, head_dim)
    # cos/sin shape: (seq_len, head_dim)
    # Need to add dimensions: (1, seq_len, 1, head_dim)
    cos = cos[None, :, None, :]  # Add batch and heads dimensions
    sin = sin[None, :, None, :]

    # Split x into two halves
    # x has shape: (batch, seq_len, n_heads, head_dim)
    # After split: each half is (batch, seq_len, n_heads, head_dim//2)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]

    # Split cos/sin to match
    # cos/sin have shape: (1, seq_len, 1, head_dim)
    # After split: each half is (1, seq_len, 1, head_dim//2)
    cos_half = cos[..., :cos.shape[-1]//2]
    sin_half = sin[..., :sin.shape[-1]//2]

    # Apply 2D rotation
    # New formula: [cos*x1 - sin*x2, sin*x1 + cos*x2]
    # This is a rotation matrix applied to pairs of dimensions
    rotated = np.concatenate([
        x1 * cos_half - x2 * sin_half,
        x1 * sin_half + x2 * cos_half
    ], axis=-1)

    return rotated


def precompute_rope_frequencies(seq_len, dim, base=10000):
    """
    Precompute RoPE rotation angles.

    WHY DIFFERENT FREQUENCIES?
    --------------------------
    Different dimensions rotate at different speeds!

    - Low dimensions: Fast rotation (captures local position)
    - High dimensions: Slow rotation (captures global position)

    This multi-scale approach captures both local and global patterns.

    Formula: θ_i = base^(-2i/dim)
    - base = 10000 (standard choice)
    - i = dimension index

    Args:
        seq_len: Maximum sequence length
        dim: Embedding dimension (must be even!)
        base: Rotation frequency base

    Returns:
        cos: Cosine values, shape (seq_len, dim)
        sin: Sine values, shape (seq_len, dim)
    """
    # Calculate frequencies for each dimension pair
    # θ_i = base^(-2i/dim)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

    # Create position indices [0, 1, 2, ..., seq_len-1]
    positions = np.arange(seq_len, dtype=np.float32)

    # Compute angles: position * frequency
    # Shape: (seq_len, dim//2)
    angles = np.outer(positions, inv_freq)

    # Duplicate for both halves of the embedding
    # Shape: (seq_len, dim)
    angles = np.concatenate([angles, angles], axis=-1)

    # Precompute cos and sin
    cos = np.cos(angles)
    sin = np.sin(angles)

    return cos, sin


# =============================================================================
# ATTENTION MECHANISM
# =============================================================================

class MultiHeadAttention:
    """
    Multi-head self-attention with RoPE.

    You've seen attention before in micro-attention!

    RECAP: Why attention?
    --------------------
    - Allows model to look at ALL positions when processing each token
    - "Where should I focus?" for each token
    - Core innovation that made transformers work

    MULTI-HEAD: Why multiple attention heads?
    -----------------------------------------
    Different heads can attend to different patterns:
    - Head 1: Attends to previous word (syntax)
    - Head 2: Attends to subject of sentence (semantics)
    - Head 3: Attends to related entities (coreference)

    By having multiple heads, we can capture multiple types of relationships!

    PROCESS:
    -------
    1. Project input to Query, Key, Value (for each head)
    2. Apply RoPE to Q and K (encode position)
    3. Compute attention scores: Q @ K^T
    4. Apply softmax: attention weights
    5. Weighted sum of values: output
    6. Concatenate heads and project
    """

    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model: Model dimension (e.g., 768)
            n_heads: Number of attention heads (e.g., 12)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Weight matrices for Q, K, V projections
        # Instead of separate matrices per head, we use one large matrix
        # and split it later (more efficient!)
        self.W_q = init_weights(d_model, d_model)
        self.W_k = init_weights(d_model, d_model)
        self.W_v = init_weights(d_model, d_model)

        # Output projection
        self.W_o = init_weights(d_model, d_model)

    def forward(self, x, cos, sin, mask=None):
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            cos: RoPE cosine, shape (seq_len, head_dim)
            sin: RoPE sine, shape (seq_len, head_dim)
            mask: Attention mask (optional), shape (seq_len, seq_len)

        Returns:
            Output: Same shape as input
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        # Shape: (batch, seq_len, d_model)
        Q = x @ self.W_q.T
        K = x @ self.W_k.T
        V = x @ self.W_v.T

        # Reshape to separate heads
        # Shape: (batch, seq_len, n_heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K (NOT V!)
        # Why not V? Positional info is in queries and keys, values are content
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        # Transpose for attention computation
        # Shape: (batch, n_heads, seq_len, head_dim)
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))

        # Compute attention scores
        # Q @ K^T: How much each query attends to each key
        # Shape: (batch, n_heads, seq_len, seq_len)
        scores = Q @ np.transpose(K, (0, 1, 3, 2))

        # Scale by sqrt(head_dim)
        # Why? Prevents scores from being too large (which would make softmax too peaky)
        scores = scores / np.sqrt(self.head_dim)

        # Apply mask (if provided)
        # Mask is used for:
        # 1. Causal masking (don't attend to future tokens)
        # 2. Padding (don't attend to padding tokens)
        if mask is not None:
            scores = scores + mask  # mask contains -inf for positions to ignore

        # Softmax to get attention weights
        # Shape: (batch, n_heads, seq_len, seq_len)
        attn_weights = softmax(scores)

        # Weighted sum of values
        # Shape: (batch, n_heads, seq_len, head_dim)
        output = attn_weights @ V

        # Transpose back and reshape
        # Shape: (batch, seq_len, n_heads, head_dim)
        output = np.transpose(output, (0, 2, 1, 3))

        # Concatenate heads
        # Shape: (batch, seq_len, d_model)
        output = output.reshape(batch_size, seq_len, d_model)

        # Final projection
        output = output @ self.W_o.T

        return output


# =============================================================================
# FEED-FORWARD NETWORK
# =============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.

    WHY FFN AFTER ATTENTION?
    -----------------------
    Attention: Mixes information across positions
    FFN: Processes each position independently

    Think of it as:
    - Attention: "What information do I need from other positions?"
    - FFN: "How do I process this information?"

    ARCHITECTURE:
    ------------
    Input (d_model)
        ↓
    Linear + GELU (d_ff, typically 4x larger)
        ↓
    Linear (d_model)

    The expansion (d_model → 4*d_model) gives the model more capacity
    to process information.
    """

    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: Model dimension (e.g., 768)
            d_ff: Feed-forward dimension (e.g., 3072 = 4 * 768)
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Two linear layers
        self.W1 = init_weights(d_ff, d_model)  # Expand
        self.b1 = np.zeros(d_ff, dtype=np.float32)

        self.W2 = init_weights(d_model, d_ff)  # Contract
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, d_model)

        Returns:
            Output: Same shape as input
        """
        # First layer: expand and activate
        # Shape: (batch, seq_len, d_ff)
        hidden = x @ self.W1.T + self.b1
        hidden = gelu(hidden)

        # Second layer: contract
        # Shape: (batch, seq_len, d_model)
        output = hidden @ self.W2.T + self.b2

        return output


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock:
    """
    One transformer block: Attention + FFN with residual connections.

    ARCHITECTURE:
    ------------
    Input
      ↓
    RMSNorm → Attention → Add (residual)
      ↓
    RMSNorm → FFN → Add (residual)
      ↓
    Output

    WHY RESIDUAL CONNECTIONS?
    -------------------------
    Residual: output = input + transformation(input)

    Without residuals:
    - Deep networks don't train well
    - Gradients vanish in backprop

    With residuals:
    - Gradients can flow directly through (via identity path)
    - Network can learn "refinements" instead of full transformations
    - Enables training of very deep networks (100+ layers!)

    WHY PRE-NORM (RMSNorm before attention/FFN)?
    --------------------------------------------
    Post-norm (original transformer): Attention → Add → Norm
    Pre-norm (modern): Norm → Attention → Add

    Pre-norm is better for training:
    - More stable gradients
    - Can use higher learning rates
    - Easier to train deep networks

    Used in: GPT-3, LLaMA, PaLM
    """

    def __init__(self, d_model, n_heads, d_ff):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
        """
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        # Layer norms (using RMSNorm)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, cos, sin, mask=None):
        """
        Forward pass through one transformer block.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            cos, sin: RoPE embeddings
            mask: Attention mask (optional)

        Returns:
            Output: Same shape as input
        """
        # Self-attention with residual
        # Pre-norm: normalize BEFORE attention
        normed = self.norm1.forward(x)
        attn_out = self.attention.forward(normed, cos, sin, mask)
        x = x + attn_out  # Residual connection

        # Feed-forward with residual
        # Pre-norm: normalize BEFORE FFN
        normed = self.norm2.forward(x)
        ffn_out = self.ffn.forward(normed)
        x = x + ffn_out  # Residual connection

        return x


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing model components")
    print("=" * 60)

    # Test 1: RMSNorm
    print("\nTest 1: RMSNorm")
    x = np.random.randn(2, 5, 64).astype(np.float32)  # (batch, seq, dim)
    norm = RMSNorm(64)
    normed = norm.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {normed.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {normed.mean():.4f}, std: {normed.std():.4f}")
    print("[OK] RMSNorm works")

    # Test 2: RoPE
    print("\nTest 2: Rotary Position Embeddings")
    seq_len, dim = 10, 64
    cos, sin = precompute_rope_frequencies(seq_len, dim)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")
    print(f"Frequencies vary: {np.allclose(cos[:, 0], cos[:, -1]) == False}")
    print("[OK] RoPE frequencies computed")

    # Test 3: Attention
    print("\nTest 3: Multi-Head Attention")
    batch_size, seq_len, d_model, n_heads = 2, 10, 64, 8
    head_dim = d_model // n_heads  # 64 / 8 = 8
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    # RoPE frequencies are computed for head_dim, not d_model
    cos, sin = precompute_rope_frequencies(seq_len, head_dim)

    print(f"d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}")
    print(f"RoPE cos/sin shape: {cos.shape} (should be seq_len x head_dim)")

    attn = MultiHeadAttention(d_model, n_heads)
    output = attn.forward(x, cos, sin)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape
    print("[OK] Attention works")

    # Test 4: FFN
    print("\nTest 4: Feed-Forward Network")
    ffn = FeedForward(d_model, d_ff=256)
    output = ffn.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape
    print("[OK] FFN works")

    # Test 5: Transformer Block
    print("\nTest 5: Transformer Block")
    block = TransformerBlock(d_model, n_heads, d_ff=256)
    output = block.forward(x, cos, sin)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape
    print("[OK] Transformer block works")

    print("\n" + "=" * 60)
    print("[DONE] All component tests passed!")
