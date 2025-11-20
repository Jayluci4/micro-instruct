"""
Training Script for micro-instruct
==================================

This implements the complete training loop for our instruction-following model.

TRAINING PROCESS:
----------------
1. Load data and tokenize
2. Create batches
3. For each batch:
   - Forward pass (compute predictions)
   - Compute loss
   - Backward pass (compute gradients)
   - Update weights
4. Repeat until convergence

THE GRADIENT CHALLENGE:
----------------------
We've built everything in NumPy so far. But backpropagation through a 12-layer
transformer is VERY tedious to implement manually (100s of lines of gradient code).

TWO APPROACHES:
--------------
1. PURE NUMPY: Implement all gradients manually (educational but tedious)
2. PYTORCH: Use autograd for gradients (practical for training)

We'll show BOTH to demonstrate the concepts, but use PyTorch for actual training
since it's 10x faster and we've already taught the concepts.

WHY THIS IS OK:
--------------
- You've learned how gradients work (micro-rnn, micro-lstm)
- The MODEL is still pure NumPy
- We're just using PyTorch for the backward pass
- After training, we can use the model in pure NumPy!
"""

import numpy as np
import json
import time
from pathlib import Path

# Our modules
from tokenizer import SimpleTokenizer
from transformer import MicroInstructModel, ModelConfig

# Optional: PyTorch for faster training
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available. Training will be slower.")


# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    """
    Load and batch training data.

    WHY BATCHING?
    ------------
    Processing one example at a time is slow!
    Batch processing:
    - Uses matrix operations efficiently
    - Averages gradients across examples (more stable)
    - Faster training (10-100x speedup)

    BATCH SIZE TRADE-OFF:
    --------------------
    Small (8-16): More updates, less memory, noisier gradients
    Large (128-256): Fewer updates, more memory, smoother gradients

    We'll use 32 as a good middle ground for our small model.
    """

    def __init__(self, data_file, tokenizer, batch_size=32, max_seq_len=512):
        """
        Args:
            data_file: Path to training_data.json
            tokenizer: SimpleTokenizer instance
            batch_size: Number of examples per batch
            max_seq_len: Maximum sequence length (truncate longer sequences)
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Load data
        print(f"Loading data from {data_file}...")
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        print(f"[OK] Loaded {len(self.data)} examples")

        # Preprocess: tokenize all examples
        print("Tokenizing examples...")
        self.tokenized_data = []

        for i, example in enumerate(self.data):
            if i % 1000 == 0:
                print(f"  {i}/{len(self.data)}...", end='\r')

            # Format as chat
            messages = [
                {'role': 'user', 'content': example['prompt']},
                {'role': 'assistant', 'content': example['response']}
            ]

            # Tokenize
            tokens = self.tokenizer.encode_chat(messages)

            # Skip if too long
            if len(tokens) > max_seq_len:
                continue

            self.tokenized_data.append({
                'tokens': tokens,
                'category': example['category'],
                'difficulty': example.get('difficulty', 'unknown')
            })

        print(f"\n[OK] Tokenized {len(self.tokenized_data)} examples")

    def get_batch(self):
        """
        Get a random batch of training examples.

        PADDING:
        -------
        Examples have different lengths. We need to pad them to same length
        for batch processing.

        Padding strategy:
        - Find max length in batch
        - Pad shorter sequences with <|pad|> token (ID=0)

        Returns:
            input_ids: Shape (batch_size, seq_len)
            target_ids: Shape (batch_size, seq_len)
        """
        # Sample random examples
        batch_data = np.random.choice(self.tokenized_data, self.batch_size, replace=False)

        # Find max length in batch
        max_len = max(len(ex['tokens']) for ex in batch_data)
        max_len = min(max_len, self.max_seq_len)  # Clip to max_seq_len

        # Create padded arrays
        input_ids = np.zeros((self.batch_size, max_len), dtype=np.int32)
        target_ids = np.zeros((self.batch_size, max_len), dtype=np.int32)

        for i, ex in enumerate(batch_data):
            tokens = ex['tokens'][:max_len]  # Truncate if needed
            length = len(tokens)

            # Input: all tokens except last
            # Target: all tokens except first (shifted by 1)
            # This is autoregressive: predict next token
            input_ids[i, :length-1] = tokens[:-1]
            target_ids[i, :length-1] = tokens[1:]

            # Pad rest with 0 (pad token)
            # These will be ignored in loss computation

        return input_ids, target_ids

    def __len__(self):
        """Number of examples"""
        return len(self.tokenized_data)


# =============================================================================
# TRAINING LOOP (NumPy Version - Simplified)
# =============================================================================

def train_numpy_simple(model, dataloader, num_steps=1000, learning_rate=1e-4):
    """
    Simple training loop using NumPy.

    WHY SIMPLIFIED?
    --------------
    Full backprop through transformer is 100s of lines of gradient code.
    We've already taught gradients in micro-rnn and micro-lstm.

    This version:
    - Shows the training structure
    - Uses the forward pass we built
    - Comments where gradients would be computed
    - Good for understanding, not practical for actual training

    For real training, use the PyTorch version below!

    Args:
        model: MicroInstructModel
        dataloader: DataLoader
        num_steps: Number of training steps
        learning_rate: Learning rate
    """
    print("Training with NumPy (simplified)")
    print("=" * 60)
    print("[NOTE] This version skips backprop for brevity.")
    print("       Use PyTorch version for actual training.")
    print("=" * 60)

    for step in range(num_steps):
        # Get batch
        input_ids, target_ids = dataloader.get_batch()

        # Forward pass
        logits, loss = model.forward(input_ids, target_ids)

        # BACKWARD PASS WOULD GO HERE
        # ===========================
        # In full implementation, we would:
        #
        # 1. Compute gradient of loss w.r.t. output logits
        #    d_loss/d_logits = softmax(logits) - one_hot(targets)
        #
        # 2. Backprop through output head
        #    d_loss/d_hidden = d_logits @ W_out
        #    d_loss/d_W_out = hidden.T @ d_logits
        #
        # 3. Backprop through final norm
        #    [RMSNorm gradient computation]
        #
        # 4. Backprop through each transformer block (12 times!)
        #    For each block:
        #    - Backprop through FFN (2 linear layers + GELU)
        #    - Backprop through norm
        #    - Backprop through attention (Q, K, V, O projections + softmax)
        #    - Backprop through norm
        #    - Handle residual connections
        #
        # 5. Backprop through embeddings
        #    [Update embedding vectors]
        #
        # This is 200+ lines of gradient code!
        # We've already taught this in micro-rnn and micro-lstm.
        # For a 12-layer transformer, it's tedious but same principles.

        # UPDATE WEIGHTS WOULD GO HERE
        # ===========================
        # For each parameter p with gradient d_p:
        #   p = p - learning_rate * d_p
        #
        # (Or use Adam optimizer for better convergence)

        # Print progress
        if step % 100 == 0:
            print(f"Step {step}/{num_steps} | Loss: {loss:.4f}")

    print("\n[NOTE] This was a demo. For real training, use PyTorch version!")


# =============================================================================
# PYTORCH WRAPPER (for practical training)
# =============================================================================

if PYTORCH_AVAILABLE:
    class TransformerPyTorch(nn.Module):
        """
        PyTorch wrapper around our NumPy model.

        WHY WRAPPER?
        -----------
        - Our model is in NumPy (educational)
        - PyTorch has autograd (practical for training)
        - This wrapper lets us use PyTorch's backward() for gradients
        - After training, we extract weights back to NumPy model!

        PHILOSOPHY:
        ----------
        We've taught you HOW gradients work (micro-rnn, micro-lstm).
        For a 12-layer transformer, it's the same ideas but tedious.
        Use PyTorch for training, understand the principles.
        """

        def __init__(self, config):
            super().__init__()
            self.config = config

            # Token embeddings
            self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

            # Transformer blocks
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm (RMSNorm is similar)
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)

            # Output head
            self.ln_final = nn.LayerNorm(config.d_model)  # Close to RMSNorm
            self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

            # Tie weights (embedding = output)
            self.output_head.weight = self.token_embeddings.weight

        def forward(self, input_ids, targets=None):
            """Forward pass"""
            # Embed
            x = self.token_embeddings(input_ids)

            # Create causal mask
            seq_len = input_ids.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)

            # Transformer
            x = self.transformer(x, mask=mask, is_causal=True)

            # Output
            x = self.ln_final(x)
            logits = self.output_head(x)

            # Loss
            loss = None
            if targets is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                    ignore_index=0  # Ignore padding
                )

            return logits, loss


    def train_pytorch(num_steps=5000, batch_size=32, learning_rate=3e-4):
        """
        Train using PyTorch for autograd.

        TRAINING PROCESS:
        ----------------
        1. Load data and create batches
        2. Initialize model and optimizer
        3. Training loop:
           - Forward pass (compute loss)
           - Backward pass (compute gradients) ← PyTorch does this!
           - Update weights (optimizer.step())
        4. Save trained weights

        OPTIMIZER: AdamW
        ---------------
        Why AdamW over SGD?
        - Adaptive learning rates (different for each parameter)
        - Momentum (smooths out noisy gradients)
        - Weight decay (regularization)

        Standard for training transformers!
        """
        print("Training with PyTorch")
        print("=" * 60)

        # Initialize
        config = ModelConfig()
        tokenizer = SimpleTokenizer()
        dataloader = DataLoader('training_data.json', tokenizer, batch_size=batch_size)

        # Create PyTorch model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = TransformerPyTorch(config).to(device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params / 1e6:.1f}M")

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Training loop
        print(f"\nStarting training for {num_steps} steps...")
        print("-" * 60)

        model.train()
        start_time = time.time()

        for step in range(num_steps):
            # Get batch
            input_ids, target_ids = dataloader.get_batch()

            # Convert to PyTorch tensors
            input_ids = torch.from_numpy(input_ids).long().to(device)
            target_ids = torch.from_numpy(target_ids).long().to(device)

            # Forward pass
            logits, loss = model(input_ids, target_ids)

            # Backward pass (THIS IS WHERE PYTORCH SHINES!)
            # Computes all gradients automatically
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute new gradients
            optimizer.step()       # Update weights

            # Progress
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | "
                      f"Time: {elapsed:.1f}s")

            # Checkpoint
            if step % 1000 == 0 and step > 0:
                checkpoint_path = f'checkpoint_step_{step}.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"[OK] Saved checkpoint: {checkpoint_path}")

        # Final save
        print("\n" + "=" * 60)
        print("Training complete!")

        final_path = 'model_final.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
        }, final_path)
        print(f"[OK] Saved final model: {final_path}")

        # Test generation
        print("\n" + "=" * 60)
        print("Testing generation...")
        test_generation(model, tokenizer, device)

        return model


    def test_generation(model, tokenizer, device):
        """
        Test the trained model with a few prompts.

        WHY TEST?
        --------
        See if the model learned anything!

        What to look for:
        - Does it follow the format?
        - Are arithmetic answers correct?
        - Does it generate valid JSON for functions?
        """
        model.eval()

        test_prompts = [
            "What is 5 + 3?",
            "What is 15 + 27?",
            "Calculate 12 * 8",
            "Search for Python tutorials"
        ]

        print("\nTest prompts:")
        print("-" * 60)

        for prompt in test_prompts:
            # Format as user message
            messages = [{'role': 'user', 'content': prompt}]
            prompt_tokens = tokenizer.encode_chat(messages)

            # Add assistant token to start generation
            prompt_tokens.append(tokenizer.assistant_id)

            # Convert to tensor
            input_ids = torch.tensor([prompt_tokens]).long().to(device)

            # Generate
            with torch.no_grad():
                for _ in range(100):  # Max 100 tokens
                    logits, _ = model(input_ids)
                    next_token_logits = logits[0, -1, :]

                    # Sample
                    probs = torch.softmax(next_token_logits / 0.7, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Append
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                    # Stop at end token
                    if next_token.item() == tokenizer.end_id:
                        break

            # Decode
            generated_tokens = input_ids[0].cpu().numpy().tolist()
            generated_text = tokenizer.decode(generated_tokens)

            print(f"\nPrompt: {prompt}")
            print(f"Response: {generated_text}")
            print("-" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train micro-instruct model')
    parser.add_argument('--mode', choices=['numpy', 'pytorch'], default='pytorch',
                       help='Training mode (numpy=demo, pytorch=real training)')
    parser.add_argument('--steps', type=int, default=5000,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')

    args = parser.parse_args()

    # Check data exists
    if not Path('training_data.json').exists():
        print("[ERROR] training_data.json not found!")
        print("Run: python generate_data.py first")
        exit(1)

    if args.mode == 'numpy':
        # NumPy demo (simplified, no actual training)
        config = ModelConfig()
        tokenizer = SimpleTokenizer()
        dataloader = DataLoader('training_data.json', tokenizer, batch_size=args.batch_size)
        model = MicroInstructModel(config)

        train_numpy_simple(model, dataloader, num_steps=args.steps, learning_rate=args.lr)

    elif args.mode == 'pytorch':
        if not PYTORCH_AVAILABLE:
            print("[ERROR] PyTorch not installed!")
            print("Install with: pip install torch")
            exit(1)

        train_pytorch(num_steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr)
