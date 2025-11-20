"""
Synthetic Data Generation for micro-instruct
============================================

WHY SYNTHETIC DATA?
------------------
We don't have a large corpus of instruction-following examples.
Solution: Generate them programmatically!

KEY INSIGHT FROM NANOCHAT #164:
-------------------------------
Small models are VERY sensitive to formatting.
We must be consistent with:
1. Spacing (always "5 + 3", not "5+3" or "5  +  3")
2. Token boundaries (use delimiters if needed)
3. Step-by-step format (train the reasoning pattern)

WHAT WE'LL GENERATE:
-------------------
1. Arithmetic (5000 examples)
   - Addition, subtraction, multiplication
   - Single digit, double digit, multi-step
   - WITH step-by-step reasoning

2. Function calling (2000 examples)
   - Calculator, search, weather, etc.
   - JSON output format
   - Argument extraction

3. Self-verification (1000 examples)
   - Check if answer is correct
   - Explain why/why not
"""

import numpy as np
import random
import json

# =============================================================================
# ARITHMETIC DATA GENERATION
# =============================================================================

def generate_single_digit_addition(n=1000):
    """
    Generate simple single-digit addition.

    WHY START SIMPLE?
    ----------------
    Small models need easy examples to bootstrap learning.
    Curriculum: Easy → Medium → Hard

    FORMATTING (CRITICAL!):
    ----------------------
    ALWAYS use consistent spacing: "5 + 3" (not "5+3" or "5  +  3")
    This ensures same tokenization every time!

    Example:
    Q: What is 5 + 3?
    A: Let me calculate step by step:
       5 + 3 = 8

       Answer: 8
    """
    data = []

    for _ in range(n):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        result = a + b

        prompt = f"What is {a} + {b}?"

        # STEP-BY-STEP REASONING
        # Why? Small models learn patterns better with explicit steps
        response = f"""Let me calculate step by step:
{a} + {b} = {result}

Answer: {result}"""

        data.append({
            'prompt': prompt,
            'response': response,
            'category': 'arithmetic',
            'difficulty': 'easy'
        })

    return data


def generate_double_digit_addition(n=1500):
    """
    Generate double-digit addition with detailed steps.

    WHY MORE STEPS?
    --------------
    Double-digit is harder. Show the model HOW to break it down:
    15 + 27 = (10 + 5) + (20 + 7)
            = (10 + 20) + (5 + 7)
            = 30 + 12
            = 42

    This teaches the model to decompose complex problems!
    """
    data = []

    for _ in range(n):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        result = a + b

        prompt = f"What is {a} + {b}?"

        # Break down into tens and ones
        a_tens, a_ones = a // 10, a % 10
        b_tens, b_ones = b // 10, b % 10

        tens_sum = (a_tens + b_tens) * 10
        ones_sum = a_ones + b_ones

        response = f"""Let me calculate step by step:
{a} + {b}

Breaking down:
{a} = {a_tens}0 + {a_ones}
{b} = {b_tens}0 + {b_ones}

Add tens: {a_tens}0 + {b_tens}0 = {tens_sum}
Add ones: {a_ones} + {b_ones} = {ones_sum}

Total: {tens_sum} + {ones_sum} = {result}

Answer: {result}"""

        data.append({
            'prompt': prompt,
            'response': response,
            'category': 'arithmetic',
            'difficulty': 'medium'
        })

    return data


def generate_subtraction(n=800):
    """Generate subtraction examples."""
    data = []

    for _ in range(n):
        # Ensure positive result
        a = random.randint(10, 99)
        b = random.randint(1, a)
        result = a - b

        prompt = f"What is {a} - {b}?"

        response = f"""Let me calculate step by step:
{a} - {b}

Working it out:
{a} - {b} = {result}

Answer: {result}"""

        data.append({
            'prompt': prompt,
            'response': response,
            'category': 'arithmetic',
            'difficulty': 'medium'
        })

    return data


def generate_multiplication(n=800):
    """Generate multiplication examples."""
    data = []

    for _ in range(n):
        a = random.randint(2, 12)
        b = random.randint(2, 12)
        result = a * b

        prompt = f"What is {a} * {b}?"

        # Show as repeated addition for small numbers
        if a <= 5:
            steps = " + ".join([str(b)] * a)
            response = f"""Let me calculate step by step:
{a} * {b} means {a} groups of {b}

{steps} = {result}

Answer: {result}"""
        else:
            response = f"""Let me calculate step by step:
{a} * {b} = {result}

Answer: {result}"""

        data.append({
            'prompt': prompt,
            'response': response,
            'category': 'arithmetic',
            'difficulty': 'medium'
        })

    return data


def generate_multi_step_arithmetic(n=400):
    """
    Generate multi-step problems: (a + b) * c

    WHY MULTI-STEP?
    --------------
    Tests if model can:
    1. Parse parentheses
    2. Follow order of operations
    3. Chain steps together
    """
    data = []

    for _ in range(n):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        c = random.randint(2, 5)

        prompt = f"What is ({a} + {b}) * {c}?"

        step1 = a + b
        result = step1 * c

        response = f"""Let me calculate step by step:
({a} + {b}) * {c}

Step 1: Parentheses first
{a} + {b} = {step1}

Step 2: Multiply
{step1} * {c} = {result}

Answer: {result}"""

        data.append({
            'prompt': prompt,
            'response': response,
            'category': 'arithmetic',
            'difficulty': 'hard'
        })

    return data


# =============================================================================
# FUNCTION CALLING DATA GENERATION
# =============================================================================

def generate_function_calling(n=2000):
    """
    Generate function calling examples.

    WHY FUNCTION CALLING?
    --------------------
    Real agents need to use tools (calculator, search, etc.)
    Model must learn to:
    1. Recognize when to use a tool
    2. Extract the right function name
    3. Extract arguments in correct format (JSON)

    CONSISTENT FORMAT (CRITICAL!):
    -----------------------------
    Always output JSON with same structure:
    {
      "function": "function_name",
      "arguments": {...}
    }
    """
    data = []

    templates = [
        # Calculator
        ("Calculate {expr}", "calculator", lambda e: {"expression": e}),
        ("Compute {expr}", "calculator", lambda e: {"expression": e}),
        ("What is {expr}?", "calculator", lambda e: {"expression": e}),

        # Search
        ("Search for {query}", "search", lambda q: {"query": q}),
        ("Find information about {query}", "search", lambda q: {"query": q}),
        ("Look up {query}", "search", lambda q: {"query": q}),

        # Weather
        ("What's the weather in {city}?", "get_weather", lambda c: {"city": c}),
        ("Weather forecast for {city}", "get_weather", lambda c: {"city": c}),
        ("How's the weather in {city}?", "get_weather", lambda c: {"city": c}),

        # Email
        ("Send email to {to} with subject {subj}", "send_email",
         lambda t, s: {"to": t, "subject": s}),
    ]

    # Generate examples
    for _ in range(n):
        if random.random() < 0.4:  # 40% calculator
            expr = f"{random.randint(1, 50)} + {random.randint(1, 50)}"
            prompt = random.choice([
                f"Calculate {expr}",
                f"Compute {expr}",
                f"What is {expr}?"
            ])
            func = "calculator"
            args = {"expression": expr}

        elif random.random() < 0.6:  # 30% search
            topics = ["Python", "machine learning", "transformers", "React",
                     "Django", "JavaScript", "neural networks", "databases"]
            query = random.choice(topics) + " tutorial"
            prompt = random.choice([
                f"Search for {query}",
                f"Find information about {query}",
                f"Look up {query}"
            ])
            func = "search"
            args = {"query": query}

        else:  # 30% weather
            cities = ["Paris", "London", "Tokyo", "New York", "Berlin",
                     "Sydney", "Mumbai", "Toronto"]
            city = random.choice(cities)
            prompt = random.choice([
                f"What's the weather in {city}?",
                f"Weather forecast for {city}",
                f"How's the weather in {city}?"
            ])
            func = "get_weather"
            args = {"city": city}

        # Format response as JSON
        # CONSISTENT SPACING IN JSON (important for tokenization!)
        function_call = {
            "function": func,
            "arguments": args
        }

        response = f"""I'll use the {func} function.

Function call:
{json.dumps(function_call, indent=2)}"""

        data.append({
            'prompt': prompt,
            'response': response,
            'category': 'function_calling',
            'difficulty': 'medium'
        })

    return data


# =============================================================================
# SELF-VERIFICATION DATA GENERATION
# =============================================================================

def generate_self_verification(n=1000):
    """
    Generate self-verification examples.

    WHY SELF-VERIFICATION?
    ---------------------
    Good models can check their own work!
    This teaches the model to:
    1. Evaluate a claim
    2. Show reasoning
    3. Correct mistakes

    PATTERN:
    -------
    Q: Is X correct?
    A: Let me verify...
       [Show work]
       [Conclusion]
    """
    data = []

    # Half correct, half incorrect
    for _ in range(n // 2):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        correct_sum = a + b

        prompt = f"Is {a} + {b} = {correct_sum}? Verify."

        response = f"""Let me verify:
{a} + {b}

Calculating: {a} + {b} = {correct_sum}

The claim says {correct_sum}.
My calculation also gives {correct_sum}.

Yes, this is correct."""

        data.append({
            'prompt': prompt,
            'response': response,
            'category': 'verification',
            'difficulty': 'medium'
        })

    # Incorrect examples
    for _ in range(n // 2):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        correct_sum = a + b
        wrong_sum = correct_sum + random.randint(1, 10)

        prompt = f"Is {a} + {b} = {wrong_sum}? Verify."

        response = f"""Let me verify:
{a} + {b}

Calculating: {a} + {b} = {correct_sum}

The claim says {wrong_sum}.
But my calculation gives {correct_sum}.

{correct_sum} is not equal to {wrong_sum}.

No, this is incorrect. The correct answer is {correct_sum}."""

        data.append({
            'prompt': prompt,
            'response': response,
            'category': 'verification',
            'difficulty': 'medium'
        })

    return data


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_all_data(output_file='training_data.json'):
    """
    Generate complete training dataset.

    CURRICULUM:
    ----------
    Mix of easy and hard examples:
    - 40% easy (single digit)
    - 40% medium (double digit, functions)
    - 20% hard (multi-step)

    This helps the model learn progressively!
    """
    print("Generating synthetic training data...")
    print("=" * 60)

    # Generate each category
    data = []

    print("\n1. Generating arithmetic examples...")
    data.extend(generate_single_digit_addition(1000))      # Easy
    data.extend(generate_double_digit_addition(1500))      # Medium
    data.extend(generate_subtraction(800))                  # Medium
    data.extend(generate_multiplication(800))               # Medium
    data.extend(generate_multi_step_arithmetic(400))        # Hard
    print(f"   [OK] {len([d for d in data if d['category'] == 'arithmetic'])} arithmetic examples")

    print("\n2. Generating function calling examples...")
    data.extend(generate_function_calling(2000))
    print(f"   [OK] {len([d for d in data if d['category'] == 'function_calling'])} function examples")

    print("\n3. Generating self-verification examples...")
    data.extend(generate_self_verification(1000))
    print(f"   [OK] {len([d for d in data if d['category'] == 'verification'])} verification examples")

    # Shuffle
    print("\n4. Shuffling data...")
    random.shuffle(data)

    # Statistics
    total = len(data)
    by_difficulty = {}
    for d in data:
        diff = d.get('difficulty', 'unknown')
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

    print("\n" + "=" * 60)
    print(f"Total examples: {total}")
    print(f"\nBy difficulty:")
    for diff, count in sorted(by_difficulty.items()):
        pct = 100 * count / total
        print(f"  {diff:10s}: {count:5d} ({pct:.1f}%)")

    print(f"\nBy category:")
    by_category = {}
    for d in data:
        cat = d['category']
        by_category[cat] = by_category.get(cat, 0) + 1

    for cat, count in sorted(by_category.items()):
        pct = 100 * count / total
        print(f"  {cat:20s}: {count:5d} ({pct:.1f}%)")

    # Save
    print(f"\n5. Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"   [OK] Saved {total} examples")

    # Save a few examples for inspection
    print("\n6. Sample examples:")
    print("-" * 60)
    for i in range(3):
        ex = data[i]
        print(f"\nExample {i+1} ({ex['category']}, {ex.get('difficulty', 'N/A')}):")
        print(f"Q: {ex['prompt']}")
        print(f"A: {ex['response'][:150]}...")

    print("\n" + "=" * 60)
    print("[DONE] Data generation complete!")
    print(f"\nNext step: Train the model using this data")

    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    generate_all_data()
