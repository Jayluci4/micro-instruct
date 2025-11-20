"""
Complete Transformer Model for micro-instruct
============================================

This brings together all components into a full language model:
- Token embeddings
- Positional encoding (RoPE)
- Transformer blocks
- Output head
- Text generation

WHY THIS STRUCTURE?
------------------
Token → Embedding → Transformer Blocks → Output Logits → Probabilities

Each part has a specific job:
1. Embedding: Convert token IDs to vectors
2. Transformer: Process and mix information
3. Output head: Convert vectors back to token probabilities
4. Generation: Sample tokens to create text
"""

import numpy as np
from model import TransformerBlock, RMSNorm, precompute_rope_frequencies, softmax, init_weights

# =============================================================================
# CONFIGURATION
# =============================================================================

class ModelConfig:
    """
    Configuration for micro-instruct model.

    WHY THESE SIZES?
    ---------------
    vocab_size: 91 (from our character tokenizer)
    max_seq_len: 512 (reasonable for conversations)
    n_layers: 12 (12-24 is standard for small models)
    n_heads: 12 (standard choice, divides evenly into d_model)
    d_model: 768 (LLaMA uses 896, we use 768 for simplicity)
    d_ff: 3072 (4x d_model is standard)

    TOTAL PARAMETERS:
    ----------------
    Embeddings: 91 × 768 = 70K
    12 Blocks × ~3.5M = 42M
    Output: 91 × 768 = 70K
    Total: ~50M parameters

    This is TINY compared to GPT-4 (~1.8T params) but perfect for learning!
    """

    def __init__(self):
        # Architecture
        self.vocab_size = 91        # From tokenizer
        self.max_seq_len = 512      # Maximum sequence length
        self.n_layers = 12          # Number of transformer blocks
        self.n_heads = 12           # Attention heads
        self.d_model = 768          # Model dimension
        self.d_ff = 3072            # Feed-forward dimension (4x d_model)
        self.dropout = 0.0          # No dropout for simplicity

        # Derived
        self.head_dim = self.d_model // self.n_heads  # 768 // 12 = 64


# =============================================================================
# COMPLETE TRANSFORMER MODEL
# =============================================================================

class MicroInstructModel:
    """
    Complete transformer model for instruction following.

    ARCHITECTURE OVERVIEW:
    ---------------------
    Input token IDs → Embedding → Add positional info (RoPE) →
    Transformer blocks → Output head → Logits → Softmax → Probabilities

    KEY DESIGN CHOICES:
    ------------------
    1. Learned embeddings (not one-hot)
    2. RoPE for positions (applied in attention, not added to embeddings)
    3. Pre-norm architecture (RMSNorm before attention/FFN)
    4. Shared embedding/output weights (tie weights to save parameters)
    """

    def __init__(self, config):
        """
        Initialize the model.

        Args:
            config: ModelConfig object
        """
        self.config = config

        # ====================================================================
        # EMBEDDING LAYER
        # ====================================================================

        # Token embeddings: Convert token IDs to vectors
        # Why learned embeddings?
        # - One-hot is sparse and wasteful
        # - Learned embeddings are dense and capture semantic similarity
        #
        # Shape: (vocab_size, d_model)
        # Each token gets a learned vector of size d_model
        self.token_embeddings = init_weights(config.vocab_size, config.d_model)

        # ====================================================================
        # TRANSFORMER BLOCKS
        # ====================================================================

        # Stack of transformer blocks
        # Each block: Attention → FFN (with residuals and norms)
        self.blocks = [
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ]

        # ====================================================================
        # OUTPUT LAYER
        # ====================================================================

        # Final layer norm before output
        self.ln_final = RMSNorm(config.d_model)

        # Output head: Convert hidden states to logits
        # Why separate from embeddings?
        # - Could tie weights (use same as token_embeddings.T) to save params
        # - Or keep separate for flexibility
        # We'll keep separate for now
        self.output_head = init_weights(config.vocab_size, config.d_model)

        # ====================================================================
        # POSITIONAL ENCODING (RoPE)
        # ====================================================================

        # Precompute RoPE frequencies
        # These are applied in the attention mechanism, not added to embeddings
        self.rope_cos, self.rope_sin = precompute_rope_frequencies(
            config.max_seq_len,
            config.head_dim
        )

        print(f"[OK] Model initialized: {self.count_parameters() / 1e6:.1f}M parameters")

    def count_parameters(self):
        """
        Count total parameters in the model.

        WHY COUNT PARAMETERS?
        --------------------
        - Tells us model size
        - Helps estimate memory and compute requirements
        - Useful for comparing models

        Returns:
            Total number of parameters
        """
        total = 0

        # Token embeddings
        total += self.token_embeddings.size

        # Transformer blocks
        for block in self.blocks:
            # Attention
            total += block.attention.W_q.size
            total += block.attention.W_k.size
            total += block.attention.W_v.size
            total += block.attention.W_o.size

            # FFN
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size

            # Norms (just gamma, no beta in RMSNorm)
            total += block.norm1.gamma.size
            total += block.norm2.gamma.size

        # Final norm
        total += self.ln_final.gamma.size

        # Output head
        total += self.output_head.size

        return total

    def forward(self, token_ids, targets=None):
        """
        Forward pass through the model.

        WHAT HAPPENS:
        ------------
        1. Embed tokens → vectors
        2. Pass through transformer blocks
        3. Convert to logits
        4. Optionally compute loss (if targets provided)

        Args:
            token_ids: Input token IDs, shape (batch_size, seq_len)
            targets: Target token IDs for training, shape (batch_size, seq_len)

        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided), scalar
        """
        batch_size, seq_len = token_ids.shape

        # ====================================================================
        # EMBEDDING
        # ====================================================================

        # Convert token IDs to embeddings
        # For each token ID, look up its embedding vector
        # Shape: (batch_size, seq_len, d_model)
        x = self.token_embeddings[token_ids]

        # ====================================================================
        # TRANSFORMER BLOCKS
        # ====================================================================

        # Get RoPE embeddings for current sequence length
        # (We precomputed for max_seq_len, now slice to current seq_len)
        cos = self.rope_cos[:seq_len, :]
        sin = self.rope_sin[:seq_len, :]

        # Create causal mask (prevent attending to future tokens)
        # Why causal?
        # - Language modeling: predict next token
        # - Can't look at future tokens!
        #
        # Mask shape: (seq_len, seq_len)
        # mask[i, j] = 0 if i >= j (can attend)
        #            = -inf if i < j (cannot attend)
        mask = self.create_causal_mask(seq_len)

        # Pass through each transformer block
        for block in self.blocks:
            x = block.forward(x, cos, sin, mask)

        # ====================================================================
        # OUTPUT
        # ====================================================================

        # Final layer norm
        x = self.ln_final.forward(x)

        # Project to vocabulary
        # Shape: (batch_size, seq_len, vocab_size)
        logits = x @ self.output_head.T

        # ====================================================================
        # LOSS (if training)
        # ====================================================================

        loss = None
        if targets is not None:
            # Compute cross-entropy loss
            # Why cross-entropy?
            # - Standard for classification (predicting next token)
            # - Penalizes confident wrong predictions more
            loss = self.compute_loss(logits, targets)

        return logits, loss

    def create_causal_mask(self, seq_len):
        """
        Create causal mask for autoregressive generation.

        WHY CAUSAL MASK?
        ---------------
        In language modeling, we predict the next token.
        Token at position i can only see tokens at positions 0..i-1.
        Cannot see position i (that's what we're predicting!)
        Cannot see positions i+1, i+2, ... (those are future!)

        IMPLEMENTATION:
        --------------
        Create upper triangular matrix of -inf values
        When added to attention scores, -inf becomes 0 after softmax

        Example for seq_len=4:
        [[  0, -inf, -inf, -inf],
         [  0,   0, -inf, -inf],
         [  0,   0,   0, -inf],
         [  0,   0,   0,   0]]

        Args:
            seq_len: Sequence length

        Returns:
            Mask of shape (seq_len, seq_len)
        """
        # Create lower triangular matrix of ones
        mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))

        # Convert 0s to -inf, keep 1s as 0
        mask = np.where(mask == 0, -1e10, 0.0)

        return mask

    def compute_loss(self, logits, targets):
        """
        Compute cross-entropy loss.

        WHAT IS CROSS-ENTROPY?
        ---------------------
        Measures how different predicted probabilities are from true labels.

        Formula: -log(p[correct_class])

        Example:
        - If we predict p=0.9 for correct class: loss = -log(0.9) = 0.1
        - If we predict p=0.1 for correct class: loss = -log(0.1) = 2.3

        Lower is better (want high probability on correct class!)

        Args:
            logits: Predicted logits, shape (batch, seq_len, vocab_size)
            targets: True token IDs, shape (batch, seq_len)

        Returns:
            Average loss over all tokens
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for easier processing
        # (batch * seq_len, vocab_size)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute softmax probabilities
        probs = softmax(logits_flat)

        # Get probability of correct token
        # For each position, get prob[correct_token]
        correct_probs = probs[np.arange(len(targets_flat)), targets_flat]

        # Cross-entropy loss: -log(p)
        # Add small epsilon to avoid log(0)
        loss = -np.log(correct_probs + 1e-10)

        # Average over all positions
        return np.mean(loss)

    def generate(self, prompt_ids, max_new_tokens=50, temperature=0.7, top_p=0.9):
        """
        Generate text autoregressively.

        WHAT IS AUTOREGRESSIVE GENERATION?
        ----------------------------------
        Generate one token at a time, feeding each prediction back as input.

        Process:
        1. Start with prompt: [1, 5, 10]
        2. Predict next token: 23
        3. Append: [1, 5, 10, 23]
        4. Predict next: 7
        5. Append: [1, 5, 10, 23, 7]
        6. Repeat until done

        SAMPLING STRATEGIES:
        -------------------
        Temperature: Controls randomness
        - Low (0.1): Deterministic, picks most likely
        - High (1.0+): Random, explores more

        Top-p (nucleus sampling): Only sample from top p% probability mass
        - Filters out very unlikely tokens
        - More coherent than pure random

        Args:
            prompt_ids: Starting tokens, shape (batch, seq_len)
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold

        Returns:
            Generated token IDs, shape (batch, seq_len + max_new_tokens)
        """
        # Ensure prompt is 2D
        if len(prompt_ids.shape) == 1:
            prompt_ids = prompt_ids[np.newaxis, :]  # Add batch dimension

        # Start with prompt
        current_ids = prompt_ids.copy()

        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Forward pass
            # Only use last max_seq_len tokens (context window limit)
            if current_ids.shape[1] > self.config.max_seq_len:
                input_ids = current_ids[:, -self.config.max_seq_len:]
            else:
                input_ids = current_ids

            logits, _ = self.forward(input_ids)

            # Get logits for last position (next token prediction)
            # Shape: (batch, vocab_size)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            # Higher temperature = more uniform distribution
            # Lower temperature = peakier distribution
            next_token_logits = next_token_logits / temperature

            # Convert to probabilities
            probs = softmax(next_token_logits)

            # Top-p filtering (nucleus sampling)
            if top_p < 1.0:
                probs = self.top_p_filtering(probs, top_p)

            # Sample next token
            # Sample from the probability distribution
            next_token = self.sample_from_probs(probs)

            # Append to sequence
            current_ids = np.concatenate([current_ids, next_token[:, np.newaxis]], axis=1)

            # Check for end token (if you want early stopping)
            # if next_token[0] == END_TOKEN_ID:
            #     break

        return current_ids

    def top_p_filtering(self, probs, top_p):
        """
        Nucleus sampling: only sample from top p% of probability mass.

        WHY TOP-P?
        ---------
        Pure sampling from softmax can produce unlikely tokens.
        Top-p keeps only the most likely tokens that sum to p probability.

        Example (top_p=0.9):
        Original: [0.5, 0.3, 0.15, 0.04, 0.01]
        Cumsum:   [0.5, 0.8, 0.95, 0.99, 1.0]
        Keep:     [0.5, 0.3, 0.15] (sum=0.95 > 0.9, stop)
        Filter:   [0.526, 0.316, 0.158, 0, 0] (renormalize)

        This prevents sampling very unlikely tokens.

        Args:
            probs: Probability distribution, shape (batch, vocab_size)
            top_p: Cumulative probability threshold

        Returns:
            Filtered probabilities, same shape
        """
        # Sort probabilities descending
        sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)

        # Cumulative sum
        cumsum_probs = np.cumsum(sorted_probs, axis=-1)

        # Find cutoff: first index where cumsum > top_p
        # Keep everything up to and including this index
        cutoff_mask = cumsum_probs <= top_p

        # Always keep at least one token (the most likely)
        cutoff_mask[:, 0] = True

        # Zero out probabilities beyond cutoff
        filtered_probs = np.where(cutoff_mask, sorted_probs, 0.0)

        # Unsort (put back in original order)
        original_order = np.argsort(sorted_indices, axis=-1)
        filtered_probs = np.take_along_axis(filtered_probs, original_order, axis=-1)

        # Renormalize
        filtered_probs = filtered_probs / (np.sum(filtered_probs, axis=-1, keepdims=True) + 1e-10)

        return filtered_probs

    def sample_from_probs(self, probs):
        """
        Sample token IDs from probability distribution.

        WHY SAMPLE INSTEAD OF ARGMAX?
        ----------------------------
        Argmax (always pick most likely): Deterministic, boring, repetitive
        Sampling: Introduces variety, more human-like

        HOW TO SAMPLE:
        -------------
        1. Convert probabilities to cumulative sum
        2. Generate random number between 0 and 1
        3. Find first position where cumsum > random

        This gives us a sample proportional to probabilities!

        Args:
            probs: Probabilities, shape (batch, vocab_size)

        Returns:
            Sampled token IDs, shape (batch,)
        """
        batch_size = probs.shape[0]
        samples = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            # Sample from categorical distribution
            samples[i] = np.random.choice(len(probs[i]), p=probs[i])

        return samples

    def save(self, filepath):
        """Save model parameters to file."""
        params = {
            'config': self.config.__dict__,
            'token_embeddings': self.token_embeddings,
            'output_head': self.output_head,
            'ln_final_gamma': self.ln_final.gamma,
        }

        # Save transformer blocks
        for i, block in enumerate(self.blocks):
            params[f'block_{i}_attn_Wq'] = block.attention.W_q
            params[f'block_{i}_attn_Wk'] = block.attention.W_k
            params[f'block_{i}_attn_Wv'] = block.attention.W_v
            params[f'block_{i}_attn_Wo'] = block.attention.W_o

            params[f'block_{i}_ffn_W1'] = block.ffn.W1
            params[f'block_{i}_ffn_b1'] = block.ffn.b1
            params[f'block_{i}_ffn_W2'] = block.ffn.W2
            params[f'block_{i}_ffn_b2'] = block.ffn.b2

            params[f'block_{i}_norm1_gamma'] = block.norm1.gamma
            params[f'block_{i}_norm2_gamma'] = block.norm2.gamma

        np.savez(filepath, **params)
        print(f"[OK] Model saved to {filepath}")

    def load(self, filepath):
        """Load model parameters from file."""
        data = np.load(filepath)

        # Load embeddings and output
        self.token_embeddings = data['token_embeddings']
        self.output_head = data['output_head']
        self.ln_final.gamma = data['ln_final_gamma']

        # Load transformer blocks
        for i, block in enumerate(self.blocks):
            block.attention.W_q = data[f'block_{i}_attn_Wq']
            block.attention.W_k = data[f'block_{i}_attn_Wk']
            block.attention.W_v = data[f'block_{i}_attn_Wv']
            block.attention.W_o = data[f'block_{i}_attn_Wo']

            block.ffn.W1 = data[f'block_{i}_ffn_W1']
            block.ffn.b1 = data[f'block_{i}_ffn_b1']
            block.ffn.W2 = data[f'block_{i}_ffn_W2']
            block.ffn.b2 = data[f'block_{i}_ffn_b2']

            block.norm1.gamma = data[f'block_{i}_norm1_gamma']
            block.norm2.gamma = data[f'block_{i}_norm2_gamma']

        print(f"[OK] Model loaded from {filepath}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Complete Transformer Model")
    print("=" * 60)

    # Create config
    config = ModelConfig()
    print(f"\nModel configuration:")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Model dimension: {config.d_model}")
    print(f"  FFN dimension: {config.d_ff}")

    # Create model
    print("\nInitializing model...")
    model = MicroInstructModel(config)

    # Test forward pass
    print("\nTest 1: Forward pass")
    batch_size = 2
    seq_len = 10
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, _ = model.forward(token_ids)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print("[OK] Forward pass works")

    # Test with loss
    print("\nTest 2: Forward pass with loss")
    targets = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model.forward(token_ids, targets)
    print(f"Loss: {loss:.4f}")
    assert loss > 0
    print("[OK] Loss computation works")

    # Test generation
    print("\nTest 3: Text generation")
    prompt = np.array([[1, 5, 10]])  # Small prompt
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0]}")
    assert generated.shape[1] == prompt.shape[1] + 20
    print("[OK] Generation works")

    # Test save/load
    print("\nTest 4: Save and load")
    model.save('test_model.npz')

    # Create new model and load
    model2 = MicroInstructModel(config)
    model2.load('test_model.npz')

    # Test they produce same output
    logits2, _ = model2.forward(token_ids)
    assert np.allclose(logits, logits2, atol=1e-5)
    print("[OK] Save/load works")

    print("\n" + "=" * 60)
    print("[DONE] All model tests passed!")
    print(f"\nModel summary:")
    print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M")
    print(f"  Memory (FP32): ~{model.count_parameters() * 4 / 1e6:.1f}MB")
