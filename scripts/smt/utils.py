"""Helper functions for SMT verification."""

from typing import Set, Tuple, Dict, Any, List


def parse_circuit_edges(circuit: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """Parse edges from circuit JSON, handling multiple formats.

    Extracted circuits write edges as:
        ["emb", "attn_0"]  (list format)
        {"from": "emb", "to": "attn_0"}  (GPT-2 dict format)
        {"source": "emb", "target": "attn_0"}  (small model dict format)

    Args:
        circuit: Circuit dict with "edges" field

    Returns:
        Set of (from_node, to_node) tuples
    """
    edges = circuit["edges"]
    parsed = set()

    for e in edges:
        if isinstance(e, dict):
            if "from" in e and "to" in e:
                # GPT-2 format: {"from": "emb", "to": "attn_0"}
                parsed.add((e["from"], e["to"]))
            elif "source" in e and "target" in e:
                # Small model format: {"source": "emb", "target": "attn_0"}
                parsed.add((e["source"], e["target"]))
            else:
                raise ValueError(f"Invalid edge dict format: {e}")
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            # List format: ["emb", "attn_0"]
            parsed.add((e[0], e[1]))
        else:
            raise ValueError(f"Invalid edge format: {e}")

    return parsed


def get_norm_params(norm_module):
    """Extract gamma/beta or weight/bias from normalization module.

    Args:
        norm_module: Normalization module (LayerNorm or SignedL1BandNorm)

    Returns:
        Tuple of (gamma, beta) tensors
    """
    # SignedL1BandNorm uses gamma/beta
    if hasattr(norm_module, "gamma") and hasattr(norm_module, "beta"):
        return norm_module.gamma, norm_module.beta

    # LayerNorm uses weight/bias
    if hasattr(norm_module, "weight") and hasattr(norm_module, "bias"):
        return norm_module.weight, norm_module.bias

    raise TypeError(f"Unsupported norm type: {type(norm_module)}")


def get_candidate_tokens(task: str, tokenizer=None) -> Dict[str, List[int]]:
    """Get candidate output tokens for each task (GPT-2 tasks).

    For tractability, we only encode logits for candidate tokens,
    not the full vocabulary (50K+ tokens).

    Args:
        task: Task name (quote_close, bracket_type, induction_ABCAB)
        tokenizer: GPT-2 tokenizer (optional, uses default token IDs if None)

    Returns:
        Dict with "candidates" and "names" lists
    """
    if task == "quote_close":
        if tokenizer:
            single_id = tokenizer.encode("'", add_special_tokens=False)[0]
            double_id = tokenizer.encode('"', add_special_tokens=False)[0]
        else:
            # Default GPT-2 token IDs
            single_id = 6  # '
            double_id = 1  # "

        return {
            "candidates": [single_id, double_id],
            "names": ["single_quote", "double_quote"],
        }

    elif task == "bracket_type":
        if tokenizer:
            right_bracket = tokenizer.encode("]", add_special_tokens=False)[0]
            right_brace = tokenizer.encode("}", add_special_tokens=False)[0]
        else:
            right_bracket = 60  # ]
            right_brace = 92  # }

        return {
            "candidates": [right_bracket, right_brace],
            "names": ["right_bracket", "right_brace"],
        }

    elif task == "induction_ABCAB":
        # For induction, candidates are the pattern tokens (A, B, C, ...)
        # Use a small synthetic vocabulary
        return {
            "candidates": list(range(20, 30)),  # 10 pattern tokens
            "names": [f"tok_{i}" for i in range(20, 30)],
        }

    else:
        raise ValueError(f"Unknown task: {task}")


def get_small_candidate_tokens(task: str) -> Dict[str, List[int]]:
    """Get candidate output tokens for small verifiable model tasks.

    Small model uses a custom 32-token vocabulary (see scripts/small/vocab.py):
    - quote_close: tokens 9 (') and 10 (")
    - bracket_type: tokens 13 (]) and 14 (})

    Args:
        task: Task name (quote_close, bracket_type)

    Returns:
        Dict with "candidates" and "names" lists
    """
    if task == "quote_close":
        return {
            "candidates": [9, 10],  # SINGLE_QUOTE, DOUBLE_QUOTE
            "names": ["single_quote", "double_quote"],
        }
    elif task == "bracket_type":
        return {
            "candidates": [13, 14],  # RIGHT_BRACKET, RIGHT_BRACE
            "names": ["right_bracket", "right_brace"],
        }
    else:
        raise ValueError(f"Unknown small model task: {task}")


def get_bandnorm_params(hidden_size: int) -> Dict[str, float]:
    """Get SignedL1BandNorm hyperparameters.

    These must match the values used in training.

    Args:
        hidden_size: Model dimension

    Returns:
        Dict with l1_low, l1_high, half_low, half_high
    """
    l1_low_per_dim = 0.55
    l1_high_per_dim = 1.05

    l1_low = l1_low_per_dim * hidden_size
    l1_high = l1_high_per_dim * hidden_size

    return {
        "l1_low": l1_low,
        "l1_high": l1_high,
        "half_low": l1_low / 2.0,
        "half_high": l1_high / 2.0,
    }
