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
from transformer import MicroInstructModel, ModelConfig

def load_model(checkpoint_path='model_final.npz'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to saved model

    Returns:
        model: Loaded MicroInstructModel
        tokenizer: SimpleTokenizer
    """
    print("Loading model...")

    # Initialize
    config = ModelConfig()
    tokenizer = SimpleTokenizer()
    model = MicroInstructModel(config)

    # Load weights
    try:
        model.load(checkpoint_path)
        print(f"[OK] Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {checkpoint_path}")
        print("\nTrain a model first:")
        print("  python train.py --mode pytorch --steps 5000")
        return None, None

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_tokens=200, temperature=0.7):
    """
    Generate response to a prompt.

    Args:
        model: Trained model
        tokenizer: Tokenizer
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

    # Convert to numpy array
    input_ids = np.array([prompt_tokens], dtype=np.int32)

    # Generate
    generated = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature
    )

    # Decode
    response_tokens = generated[0].tolist()

    # Extract just the assistant's response
    # Find first assistant token and last end token
    try:
        start_idx = response_tokens.index(tokenizer.assistant_id) + 1
        end_idx = response_tokens.index(tokenizer.end_id, start_idx)
        response_tokens = response_tokens[start_idx:end_idx]
    except ValueError:
        # If no end token, take all
        response_tokens = response_tokens[start_idx:]

    response = tokenizer.decode(response_tokens)

    return response


def interactive_demo(model, tokenizer):
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
            response = generate_response(model, tokenizer, user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            print("Try a different prompt or check the model.")

    print("\n" + "=" * 60)


def test_examples(model, tokenizer):
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

        response = generate_response(model, tokenizer, prompt, max_tokens=150)
        print(f"Response:\n{response}")
        print("-" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Demo micro-instruct model')
    parser.add_argument('--model', type=str, default='model_final.npz',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', choices=['interactive', 'test'], default='interactive',
                       help='Demo mode')

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model)

    if model is None:
        exit(1)

    # Run demo
    if args.mode == 'interactive':
        interactive_demo(model, tokenizer)
    elif args.mode == 'test':
        test_examples(model, tokenizer)
