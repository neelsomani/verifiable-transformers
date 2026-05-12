"""
Token vocabulary for the small verifiable Transformer.

This module defines a custom vocabulary for three symbolic tasks:
- quote_close: Match opening quotes (' or ")
- bracket_type: Match opening brackets ([ or {)
- add_mod_5: Addition modulo 5

The vocabulary is kept minimal (32 tokens) to make SMT verification tractable.
"""

from typing import Dict, List, Set

# Special tokens
PAD = 0
BOS = 1

# Task identifier tokens
TASK_QUOTE = 2
TASK_BRACKET = 3
TASK_ADD = 4

# Content tokens (used as filler in syntax tasks)
A = 5
B = 6
C = 7
D = 8

# Quote tokens
SINGLE_QUOTE = 9      # '
DOUBLE_QUOTE = 10     # "

# Bracket tokens
LEFT_BRACKET = 11     # [
LEFT_BRACE = 12       # {
RIGHT_BRACKET = 13    # ]
RIGHT_BRACE = 14      # }

# Arithmetic tokens
PLUS = 15             # +
EQ = 16               # =

# Digit tokens (0-4 for mod 5 addition)
DIGIT_0 = 17
DIGIT_1 = 18
DIGIT_2 = 19
DIGIT_3 = 20
DIGIT_4 = 21

# Vocabulary size (padded to power of 2)
VOCAB_SIZE = 32

# Human-readable token names
TOKEN_NAMES: Dict[int, str] = {
    PAD: "<PAD>",
    BOS: "<BOS>",
    TASK_QUOTE: "<TASK_QUOTE>",
    TASK_BRACKET: "<TASK_BRACKET>",
    TASK_ADD: "<TASK_ADD>",
    A: "A",
    B: "B",
    C: "C",
    D: "D",
    SINGLE_QUOTE: "'",
    DOUBLE_QUOTE: '"',
    LEFT_BRACKET: "[",
    LEFT_BRACE: "{",
    RIGHT_BRACKET: "]",
    RIGHT_BRACE: "}",
    PLUS: "+",
    EQ: "=",
    DIGIT_0: "0",
    DIGIT_1: "1",
    DIGIT_2: "2",
    DIGIT_3: "3",
    DIGIT_4: "4",
}

# Reverse mapping from name to token ID
NAME_TO_TOKEN: Dict[str, int] = {v: k for k, v in TOKEN_NAMES.items()}

# Content tokens for syntax tasks
CONTENT_TOKENS: List[int] = [A, B, C, D]

# Digit tokens for arithmetic
DIGIT_TOKENS: List[int] = [DIGIT_0, DIGIT_1, DIGIT_2, DIGIT_3, DIGIT_4]

# Candidate output sets per task
QUOTE_CANDIDATES: Set[int] = {SINGLE_QUOTE, DOUBLE_QUOTE}
BRACKET_CANDIDATES: Set[int] = {RIGHT_BRACKET, RIGHT_BRACE}
ADD_CANDIDATES: Set[int] = set(DIGIT_TOKENS)

# Map task token to candidate set
TASK_TO_CANDIDATES: Dict[int, Set[int]] = {
    TASK_QUOTE: QUOTE_CANDIDATES,
    TASK_BRACKET: BRACKET_CANDIDATES,
    TASK_ADD: ADD_CANDIDATES,
}

# Map task token to task name
TASK_NAMES: Dict[int, str] = {
    TASK_QUOTE: "quote_close",
    TASK_BRACKET: "bracket_type",
    TASK_ADD: "add_mod_5",
}

# Reverse mapping
TASK_NAME_TO_TOKEN: Dict[str, int] = {v: k for k, v in TASK_NAMES.items()}


def token_to_str(token_id: int) -> str:
    """Convert token ID to human-readable string."""
    return TOKEN_NAMES.get(token_id, f"<UNK:{token_id}>")


def tokens_to_str(token_ids: List[int]) -> str:
    """Convert list of token IDs to human-readable string."""
    return " ".join(token_to_str(tid) for tid in token_ids)


def digit_value(digit_token: int) -> int:
    """Convert digit token to its numeric value (0-4)."""
    if digit_token < DIGIT_0 or digit_token > DIGIT_4:
        raise ValueError(f"Invalid digit token: {digit_token}")
    return digit_token - DIGIT_0


def value_to_digit(value: int) -> int:
    """Convert numeric value (0-4) to digit token."""
    if value < 0 or value > 4:
        raise ValueError(f"Invalid digit value: {value}")
    return DIGIT_0 + value


def get_task_name(task_token: int) -> str:
    """Get task name from task token."""
    return TASK_NAMES.get(task_token, "unknown")


def get_candidates(task_token: int) -> Set[int]:
    """Get candidate output tokens for a task."""
    return TASK_TO_CANDIDATES.get(task_token, set())


def save_vocab(filepath: str) -> None:
    """Save vocabulary to JSON file."""
    import json
    vocab_dict = {
        "vocab_size": VOCAB_SIZE,
        "token_names": {str(k): v for k, v in TOKEN_NAMES.items()},
        "task_names": {str(k): v for k, v in TASK_NAMES.items()},
        "task_to_candidates": {str(k): list(v) for k, v in TASK_TO_CANDIDATES.items()},
    }
    with open(filepath, "w") as f:
        json.dump(vocab_dict, f, indent=2)


if __name__ == "__main__":
    # Print vocabulary summary
    print("Small Verifiable Transformer Vocabulary")
    print("=" * 50)
    print(f"Total vocab size: {VOCAB_SIZE}")
    print(f"Used tokens: {len(TOKEN_NAMES)}")
    print("\nTokens by category:")
    print("\nSpecial:", [token_to_str(t) for t in [PAD, BOS]])
    print("Tasks:", [token_to_str(t) for t in [TASK_QUOTE, TASK_BRACKET, TASK_ADD]])
    print("Content:", [token_to_str(t) for t in CONTENT_TOKENS])
    print("Quotes:", [token_to_str(t) for t in [SINGLE_QUOTE, DOUBLE_QUOTE]])
    print("Brackets:", [token_to_str(t) for t in [LEFT_BRACKET, LEFT_BRACE, RIGHT_BRACKET, RIGHT_BRACE]])
    print("Arithmetic:", [token_to_str(t) for t in [PLUS, EQ]])
    print("Digits:", [token_to_str(t) for t in DIGIT_TOKENS])
    print("\nTask candidate sets:")
    for task_token, task_name in TASK_NAMES.items():
        candidates = get_candidates(task_token)
        print(f"  {task_name}: {[token_to_str(c) for c in candidates]}")
