"""Bounded domain generation for formal verification."""

from typing import List, Set, Tuple
import itertools


def generate_bounded_sequences(
    task: str,
    max_length: int,
    vocab: Set[int],
    special_tokens: dict,
) -> List[Tuple[List[int], str]]:
    """Generate all valid sequences for bounded verification.

    Args:
        task: Task name (quote_close, bracket_type, induction_ABCAB)
        max_length: Maximum sequence length
        vocab: Set of vocabulary token IDs
        special_tokens: Dict mapping special token names to IDs

    Returns:
        List of (token_sequence, label) pairs
    """
    if task == "quote_close":
        return generate_quote_close_sequences(max_length, special_tokens)
    elif task == "bracket_type":
        return generate_bracket_type_sequences(max_length, special_tokens)
    elif task == "induction_ABCAB":
        return generate_induction_sequences(max_length, vocab)
    else:
        raise ValueError(f"Unknown task: {task}")


def generate_quote_close_sequences(
    max_length: int,
    special_tokens: dict,
    max_sequences: int = 10000,
) -> List[Tuple[List[int], str]]:
    """Generate quote closing sequences.

    Pattern: <content> <quote> <more_content>
    where the quote is either ' or "

    For exhaustive formal verification, use a small content vocabulary (e.g., 2 tokens).
    With content_vocab size m, approximate count at length L is: 2 * (L-2) * m^(L-1).

    Args:
        max_length: Maximum sequence length
        special_tokens: Dict with 'single_quote' and 'double_quote' token IDs
        max_sequences: Safety cap to prevent accidental explosion

    Returns:
        List of (sequence, correct_closing_quote) pairs
    """
    single_id = special_tokens["single_quote"]
    double_id = special_tokens["double_quote"]

    # For simplicity, we use a fixed vocabulary for content
    # In practice, this would be domain-specific tokens
    content_vocab = special_tokens.get("content_tokens", list(range(10, 20)))

    sequences = []

    # Generate sequences of varying lengths
    for seq_len in range(3, max_length + 1):
        # Position of opening quote
        for quote_pos in range(1, seq_len - 1):
            content_before_len = quote_pos
            content_after_len = seq_len - quote_pos - 1

            for content_before in itertools.product(content_vocab, repeat=content_before_len):
                for content_after in itertools.product(content_vocab, repeat=content_after_len):
                    # Single quote version
                    seq_single = list(content_before) + [single_id] + list(content_after)
                    sequences.append((seq_single, "single"))

                    # Double quote version
                    seq_double = list(content_before) + [double_id] + list(content_after)
                    sequences.append((seq_double, "double"))

                    # Safety cap to prevent accidental explosion
                    if len(sequences) >= max_sequences:
                        raise RuntimeError(
                            f"Domain size exceeded max_sequences={max_sequences}. "
                            f"Generated {len(sequences)} sequences at length {seq_len}. "
                            "Use a smaller content vocabulary or lower max_length for exhaustive verification. "
                            f"Example: content_tokens=[10, 11] gives ~136 sequences at max_length=5."
                        )

    return sequences


def generate_bracket_type_sequences(
    max_length: int,
    special_tokens: dict,
    max_sequences: int = 10000,
) -> List[Tuple[List[int], str]]:
    """Generate bracket type sequences.

    Pattern: <content> <bracket> <more_content>
    where bracket is [ or {

    For exhaustive formal verification, use a small content vocabulary (e.g., 2 tokens).
    With content_vocab size m, approximate count at length L is: 2 * (L-2) * m^(L-1).

    Args:
        max_length: Maximum sequence length
        special_tokens: Dict with 'left_bracket' and 'left_brace' token IDs
        max_sequences: Safety cap to prevent accidental explosion

    Returns:
        List of (sequence, correct_closing_bracket) pairs
    """
    bracket_id = special_tokens["left_bracket"]  # [
    brace_id = special_tokens["left_brace"]  # {

    content_vocab = special_tokens.get("content_tokens", list(range(10, 20)))

    sequences = []

    for seq_len in range(3, max_length + 1):
        for bracket_pos in range(1, seq_len - 1):
            content_before_len = bracket_pos
            content_after_len = seq_len - bracket_pos - 1

            for content_before in itertools.product(content_vocab, repeat=content_before_len):
                for content_after in itertools.product(content_vocab, repeat=content_after_len):
                    # Bracket version
                    seq_bracket = list(content_before) + [bracket_id] + list(content_after)
                    sequences.append((seq_bracket, "bracket"))

                    # Brace version
                    seq_brace = list(content_before) + [brace_id] + list(content_after)
                    sequences.append((seq_brace, "brace"))

                    # Safety cap to prevent accidental explosion
                    if len(sequences) >= max_sequences:
                        raise RuntimeError(
                            f"Domain size exceeded max_sequences={max_sequences}. "
                            f"Generated {len(sequences)} sequences at length {seq_len}. "
                            "Use a smaller content vocabulary or lower max_length for exhaustive verification. "
                            f"Example: content_tokens=[10, 11] gives ~136 sequences at max_length=5."
                        )

    return sequences


def generate_induction_sequences(
    max_length: int,
    vocab: Set[int],
) -> List[Tuple[List[int], int]]:
    """Generate induction sequences.

    Pattern: A B C ... A B
    Expected output: C

    Args:
        max_length: Maximum sequence length
        vocab: Vocabulary to sample A, B, C from

    Returns:
        List of (sequence, expected_token) pairs
    """
    # Use a small subset for tractability
    token_subset = list(vocab)[:10] if len(vocab) > 10 else list(vocab)

    sequences = []

    # Minimum length: A B C A B (5 tokens)
    for seq_len in range(5, max_length + 1):
        # Try different positions for the pattern
        for pattern_len in [3, 4]:  # ABC or ABCD patterns
            if seq_len < 2 * pattern_len - 1:
                continue

            # Choose tokens for the pattern
            for tokens in itertools.permutations(token_subset, pattern_len):
                A, B, C = tokens[0], tokens[1], tokens[2]

                # Build sequence: A B C ... A B
                seq = list(tokens)  # A B C ...

                # Add filler if needed
                filler_len = seq_len - 2 * pattern_len + 1
                if filler_len > 0:
                    # Add filler tokens (can be anything except pattern tokens)
                    filler_vocab = [t for t in token_subset if t not in tokens]
                    if filler_vocab:
                        for filler in itertools.product(filler_vocab, repeat=min(filler_len, 2)):
                            seq_with_filler = list(tokens) + list(filler) + [A, B]
                            sequences.append((seq_with_filler, C))

                            if len(sequences) >= 1000:
                                return sequences
                else:
                    # No filler, just A B C A B
                    seq.extend([A, B])
                    sequences.append((seq, C))

                if len(sequences) >= 1000:
                    return sequences

    return sequences


def enumerate_small_domain(
    vocab_size: int,
    max_length: int,
    max_sequences: int = 10000,
) -> List[List[int]]:
    """Enumerate all sequences up to max_length (for very small domains).

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum sequence length
        max_sequences: Maximum number of sequences to generate

    Returns:
        List of all token sequences
    """
    sequences = []

    for length in range(1, max_length + 1):
        for seq in itertools.product(range(vocab_size), repeat=length):
            sequences.append(list(seq))
            if len(sequences) >= max_sequences:
                return sequences

    return sequences
