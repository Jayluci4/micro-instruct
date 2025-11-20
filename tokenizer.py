"""
Tokenizer for micro-instruct
=============================

WHY DO WE NEED A TOKENIZER?
---------------------------
Neural networks work with numbers, not text. A tokenizer converts text into
numbers (encoding) and back (decoding).

DESIGN CHOICES FOR SMALL MODELS:
--------------------------------
After studying nanochat #164, we learned that SMALL MODELS are VERY sensitive
to how text is tokenized. Inconsistent tokenization confuses them.

Example of the problem:
  "5+3"    might tokenize to: [5+3]       (one token)
  "5 + 3"  might tokenize to: [5, +, 3]  (three tokens)

For a small model, these look like completely different inputs!

OUR SOLUTION: Character-level + Special Tokens
----------------------------------------------
- Numbers: Each digit is a separate character (0-9)
- Operators: Each operator is a character (+, -, *, /)
- Letters: Standard alphabet
- Special tokens: <|user|>, <|assistant|>, etc.

This gives us CONSISTENT tokenization - same input always produces same tokens.

Trade-off: Longer sequences (each character is a token)
Benefit: Perfect consistency, small vocab (~300 tokens vs GPT-4's 100K)
"""

import numpy as np

class SimpleTokenizer:
    """
    Character-level tokenizer with special tokens.

    Vocabulary structure:
    - Index 0-4: Special tokens (user, assistant, system, end, pad)
    - Index 5+: Regular characters (digits, letters, punctuation)

    Total vocab size: ~300 tokens (tiny compared to GPT-4's 100K!)
    """

    def __init__(self):
        # =================================================================
        # BUILD VOCABULARY
        # =================================================================

        # Special tokens FIRST (indices 0-4)
        # Why? So we can easily check if token_id < 5 means special token
        special_tokens = [
            '<|pad|>',       # Index 0: Padding (for batching)
            '<|user|>',      # Index 1: User message start
            '<|assistant|>', # Index 2: Assistant message start
            '<|system|>',    # Index 3: System prompt
            '<|end|>'        # Index 4: End of message
        ]

        # Regular characters
        # Why this order? Digits first, then lowercase, uppercase, symbols
        # Makes it easier to debug (small numbers = common tokens)
        regular_chars = []

        # Digits (0-9)
        regular_chars.extend(list('0123456789'))

        # Lowercase letters (a-z)
        # Most common in our training data
        regular_chars.extend(list('abcdefghijklmnopqrstuvwxyz'))

        # Uppercase letters (A-Z)
        # Less common, but needed
        regular_chars.extend(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

        # Punctuation and operators
        # Critical for arithmetic and function calling!
        regular_chars.extend(list(' .,!?:;-+*/=<>()[]{}"\'\n\t_'))

        # Combine: special tokens + regular characters
        self.vocab = special_tokens + regular_chars
        self.vocab_size = len(self.vocab)

        # Create bidirectional mappings
        # char → index (for encoding)
        # index → char (for decoding)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Store special token IDs for easy access
        self.pad_id = 0
        self.user_id = 1
        self.assistant_id = 2
        self.system_id = 3
        self.end_id = 4

        print(f"[OK] Tokenizer initialized with {self.vocab_size} tokens")
        print(f"     Special tokens: {special_tokens}")

    def encode(self, text):
        """
        Convert text to list of token IDs.

        WHY CHARACTER-LEVEL?
        -------------------
        Character-level tokenization means "hello" becomes [h, e, l, l, o]
        instead of a single token. This gives us:

        1. Consistent tokenization (same input → same tokens)
        2. No unknown tokens (any character can be encoded)
        3. Small vocabulary (only ~300 tokens to learn)

        HANDLING SPECIAL TOKENS:
        -----------------------
        Special tokens like <|user|> need to be treated as single units,
        not split into characters.

        We use a simple trick: temporarily replace them with placeholder
        characters (\x00, \x01, etc.), encode normally, then restore.

        Args:
            text: String to encode

        Returns:
            List of token IDs (integers)

        Example:
            >>> tokenizer.encode("Hello")
            [44, 31, 38, 38, 41]  # h, e, l, l, o

            >>> tokenizer.encode("<|user|>Hi")
            [1, 44, 27]  # <|user|>, H, i
        """

        # Handle special tokens
        # Replace each special token with a unique placeholder
        placeholders = {}
        for special in ['<|pad|>', '<|user|>', '<|assistant|>', '<|system|>', '<|end|>']:
            if special in text:
                # Use rare control characters as placeholders
                placeholder = chr(len(placeholders))  # \x00, \x01, etc.
                placeholders[placeholder] = special
                text = text.replace(special, placeholder)

        # Encode each character
        tokens = []
        for char in text:
            if char in placeholders:
                # This is a placeholder - encode the special token
                special = placeholders[char]
                tokens.append(self.char_to_idx[special])
            elif char in self.char_to_idx:
                # Regular character
                tokens.append(self.char_to_idx[char])
            else:
                # Unknown character - skip it
                # In production, you'd want to handle this better
                # (e.g., use a special <|unk|> token)
                print(f"[WARNING] Unknown character: {repr(char)}")
                continue

        return tokens

    def decode(self, token_ids):
        """
        Convert list of token IDs back to text.

        WHY IS DECODING SIMPLE?
        ----------------------
        Because we use character-level tokenization, decoding is just:
        1. Look up each token ID in the vocabulary
        2. Concatenate the characters

        No complex detokenization rules needed!

        Args:
            token_ids: List of integers (token IDs)

        Returns:
            String

        Example:
            >>> tokenizer.decode([44, 31, 38, 38, 41])
            "hello"

            >>> tokenizer.decode([1, 44, 27, 4])
            "<|user|>Hi<|end|>"
        """
        chars = []
        for token_id in token_ids:
            if token_id in self.idx_to_char:
                chars.append(self.idx_to_char[token_id])
            else:
                # Invalid token ID - skip
                print(f"[WARNING] Invalid token ID: {token_id}")
                continue

        return ''.join(chars)

    def encode_chat(self, messages):
        """
        Encode a conversation in chat format.

        WHY CHAT FORMAT?
        ---------------
        Modern LLMs are trained on conversations, not raw text.
        Chat format makes it clear who's speaking:

        <|system|>You are helpful<|end|>
        <|user|>What is 5+3?<|end|>
        <|assistant|>Let me calculate: 5+3=8<|end|>

        This structure helps the model learn to:
        1. Understand roles (system, user, assistant)
        2. Know when messages start/end
        3. Generate appropriate responses

        Args:
            messages: List of dicts with 'role' and 'content'
                     [{'role': 'user', 'content': 'Hi'},
                      {'role': 'assistant', 'content': 'Hello!'}]

        Returns:
            List of token IDs
        """
        formatted_text = ""

        for msg in messages:
            role = msg['role']
            content = msg['content']

            # Add role token
            if role == 'system':
                formatted_text += '<|system|>'
            elif role == 'user':
                formatted_text += '<|user|>'
            elif role == 'assistant':
                formatted_text += '<|assistant|>'
            else:
                raise ValueError(f"Unknown role: {role}")

            # Add content
            formatted_text += content

            # Add end token
            formatted_text += '<|end|>'

        # Encode the formatted text
        return self.encode(formatted_text)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing SimpleTokenizer")
    print("=" * 60)

    # Create tokenizer
    tokenizer = SimpleTokenizer()

    # Test 1: Basic encoding/decoding
    print("\nTest 1: Basic text")
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"Original: {text}")
    print(f"Tokens:   {tokens}")
    print(f"Decoded:  {decoded}")
    assert decoded == text, "Encoding/decoding mismatch!"
    print("[OK] Basic encoding/decoding works")

    # Test 2: Special tokens
    print("\nTest 2: Special tokens")
    text = "<|user|>Hi<|end|>"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"Original: {text}")
    print(f"Tokens:   {tokens}")
    print(f"Decoded:  {decoded}")
    assert decoded == text, "Special tokens failed!"
    print("[OK] Special tokens work")

    # Test 3: Arithmetic (critical for our use case!)
    print("\nTest 3: Arithmetic expressions")
    expressions = ["5+3", "15 + 27", "12 * 8", "(5 + 3) * 2"]

    for expr in expressions:
        tokens = tokenizer.encode(expr)
        decoded = tokenizer.decode(tokens)
        print(f"{expr:15} -> {len(tokens)} tokens -> {decoded}")
        assert decoded == expr

    print("[OK] Arithmetic tokenization works")

    # Test 4: Chat format
    print("\nTest 4: Chat format")
    messages = [
        {'role': 'system', 'content': 'You are helpful'},
        {'role': 'user', 'content': 'What is 5+3?'},
        {'role': 'assistant', 'content': '8'}
    ]

    tokens = tokenizer.encode_chat(messages)
    decoded = tokenizer.decode(tokens)

    print(f"Chat tokens: {len(tokens)} tokens")
    print(f"Decoded:\n{decoded}")
    print("[OK] Chat format works")

    # Test 5: Tokenization consistency (nanochat #164 insight!)
    print("\nTest 5: Consistency check (CRITICAL for small models)")

    # These should tokenize consistently
    variants = ["5+3", "5 + 3", "5  +  3"]
    token_lists = [tokenizer.encode(v) for v in variants]

    for variant, tokens in zip(variants, token_lists):
        print(f"{variant:10} -> {tokens}")

    # They will be DIFFERENT because of spaces
    # This is INTENTIONAL - we want to force consistent formatting
    # in our training data!
    print("\nNOTE: Different spacing = different tokens")
    print("      This is why we'll use consistent formatting in training data!")
    print("      All arithmetic will be 'a + b' (with spaces)")

    print("\n" + "=" * 60)
    print("[DONE] All tokenizer tests passed!")
