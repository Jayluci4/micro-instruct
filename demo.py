"""
Demo Script for micro-instruct
==============================

Interactive demo to test your trained model!

USAGE:
-----
python demo.py

Then type prompts and see what the model generates.
"""

import numpy as np
from tokenizer import SimpleTokenizer
from transformer import ModelConfig

# Import PyTorch components for loading trained models
try:
    import torch
    import torch.nn as nn
    from train import TransformerPyTorch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available. Cannot load trained models.")

def load_model(checkpoint_path='model_final.pt'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to saved model (.pt file)

    Returns:
        model: Loaded PyTorch model
        tokenizer: SimpleTokenizer
        device: torch device
    """
    if not PYTORCH_AVAILABLE:
        print("[ERROR] PyTorch is required to load trained models.")
        print("Install with: pip install torch")
        return None, None, None

    print("Loading model...")

    # Initialize
    config = ModelConfig()
    tokenizer = SimpleTokenizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = TransformerPyTorch(config).to(device)

    # Load weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"[OK] Model loaded from {checkpoint_path}")
        print(f"[OK] Using device: {device}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {checkpoint_path}")
        print("\nTrain a model first:")
        print("  python train.py --mode pytorch --steps 5000")
        return None, None, None

    return model, tokenizer, device


def generate_response(model, tokenizer, device, prompt, max_tokens=200, temperature=0.7):
    """
    Generate response to a prompt.

    Args:
        model: Trained PyTorch model
        tokenizer: Tokenizer
        device: torch device
        prompt: User prompt (string)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response (string)
    """
    # Format as chat
    messages = [
        {'role': 'user', 'content': prompt}
    ]

    # Tokenize
    prompt_tokens = tokenizer.encode_chat(messages)

    # Add assistant token to start generation
    prompt_tokens.append(tokenizer.assistant_id)

    # Convert to tensor
    input_ids = torch.tensor([prompt_tokens]).long().to(device)

    # Generate
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]

            # Sample with temperature
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop at end token
            if next_token.item() == tokenizer.end_id:
                break

    # Decode
    response_tokens = input_ids[0].cpu().numpy().tolist()

    # Extract just the assistant's response
    # Find first assistant token and last end token
    try:
        start_idx = response_tokens.index(tokenizer.assistant_id) + 1
        end_idx = response_tokens.index(tokenizer.end_id, start_idx)
        response_tokens = response_tokens[start_idx:end_idx]
    except ValueError:
        # If no end token, take from assistant token to end
        start_idx = response_tokens.index(tokenizer.assistant_id) + 1
        response_tokens = response_tokens[start_idx:]

    response = tokenizer.decode(response_tokens)

    return response


def interactive_demo(model, tokenizer, device):
    """
    Interactive demo - chat with your model!
    """
    print("\n" + "=" * 60)
    print("micro-instruct Interactive Demo")
    print("=" * 60)
    print("\nType your prompts below. Type 'quit' to exit.")
    print("\nExample prompts:")
    print("  - What is 15 + 27?")
    print("  - Calculate 12 * 8")
    print("  - Search for Python tutorials")
    print("  - Is 5 + 3 = 8? Verify.")
    print("\n" + "-" * 60)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Generate response
        try:
            response = generate_response(model, tokenizer, device, user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            print("Try a different prompt or check the model.")

    print("\n" + "=" * 60)


def test_examples(model, tokenizer, device):
    """
    Test model on predefined examples.
    """
    print("\n" + "=" * 60)
    print("Testing on Example Prompts")
    print("=" * 60)

    test_cases = [
        # Arithmetic
        ("What is 5 + 3?", "arithmetic"),
        ("What is 15 + 27?", "arithmetic"),
        ("What is 12 * 8?", "arithmetic"),
        ("What is (5 + 3) * 2?", "arithmetic"),

        # Function calling
        ("Calculate 25 + 17", "function"),
        ("Search for machine learning", "function"),
        ("What's the weather in Paris?", "function"),

        # Verification
        ("Is 5 + 3 = 8? Verify.", "verification"),
        ("Is 10 + 5 = 20? Verify.", "verification"),
    ]

    for prompt, category in test_cases:
        print(f"\n[{category.upper()}]")
        print(f"Prompt: {prompt}")

        response = generate_response(model, tokenizer, device, prompt, max_tokens=150)
        print(f"Response:\n{response}")
        print("-" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Demo micro-instruct model')
    parser.add_argument('--model', type=str, default='model_final.pt',
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--mode', choices=['interactive', 'test'], default='interactive',
                       help='Demo mode')

    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_model(args.model)

    if model is None:
        exit(1)

    # Run demo
    if args.mode == 'interactive':
        interactive_demo(model, tokenizer, device)
    elif args.mode == 'test':
        test_examples(model, tokenizer, device)
