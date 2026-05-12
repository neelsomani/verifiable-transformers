"""
Token vocabulary for the small verifiable Transformer.

This module defines a custom vocabulary for two symbolic tasks:
- quote_close: Match opening quotes (' or ")
- bracket_type: Match opening brackets ([ or {)

The vocabulary is kept minimal (32 tokens) to make SMT verification tractable.
"""

from typing import Dict, List, Set

# Special tokens
PAD = 0
BOS = 1

# Task identifier tokens
TASK_QUOTE = 2
TASK_BRACKET = 3

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

# Vocabulary size (padded to power of 2)
VOCAB_SIZE = 32

# Human-readable token names
TOKEN_NAMES: Dict[int, str] = {
    PAD: "<PAD>",
    BOS: "<BOS>",
    TASK_QUOTE: "<TASK_QUOTE>",
    TASK_BRACKET: "<TASK_BRACKET>",
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
}

# Reverse mapping from name to token ID
NAME_TO_TOKEN: Dict[str, int] = {v: k for k, v in TOKEN_NAMES.items()}

# Content tokens for syntax tasks
CONTENT_TOKENS: List[int] = [A, B, C, D]

# Candidate output sets per task
QUOTE_CANDIDATES: Set[int] = {SINGLE_QUOTE, DOUBLE_QUOTE}
BRACKET_CANDIDATES: Set[int] = {RIGHT_BRACKET, RIGHT_BRACE}

# Map task token to candidate set
TASK_TO_CANDIDATES: Dict[int, Set[int]] = {
    TASK_QUOTE: QUOTE_CANDIDATES,
    TASK_BRACKET: BRACKET_CANDIDATES,
}

# Map task token to task name
TASK_NAMES: Dict[int, str] = {
    TASK_QUOTE: "quote_close",
    TASK_BRACKET: "bracket_type",
}

# Reverse mapping
TASK_NAME_TO_TOKEN: Dict[str, int] = {v: k for k, v in TASK_NAMES.items()}


def token_to_str(token_id: int) -> str:
    """Convert token ID to human-readable string."""
    return TOKEN_NAMES.get(token_id, f"<UNK:{token_id}>")


def tokens_to_str(token_ids: List[int]) -> str:
    """Convert list of token IDs to human-readable string."""
    return " ".join(token_to_str(tid) for tid in token_ids)


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
    print("Tasks:", [token_to_str(t) for t in [TASK_QUOTE, TASK_BRACKET]])
    print("Content:", [token_to_str(t) for t in CONTENT_TOKENS])
    print("Quotes:", [token_to_str(t) for t in [SINGLE_QUOTE, DOUBLE_QUOTE]])
    print("Brackets:", [token_to_str(t) for t in [LEFT_BRACKET, LEFT_BRACE, RIGHT_BRACKET, RIGHT_BRACE]])
    print("\nTask candidate sets:")
    for task_token, task_name in TASK_NAMES.items():
        candidates = get_candidates(task_token)
        print(f"  {task_name}: {[token_to_str(c) for c in candidates]}")
