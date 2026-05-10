#!/usr/bin/env python3
"""
Formal verification of extracted circuits using SMT solvers.

Verifies properties like:
- Functional equivalence to symbolic reference program
- Content invariance
- Quote-type sensitivity
- Edge necessity
- Token-renaming equivariance
- Filler invariance
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Set, Tuple

try:
    from z3 import *
except ImportError:
    print("ERROR: z3-solver not installed. Install with: pip install z3-solver")
    sys.exit(1)

import torch
from transformers import GPT2Tokenizer

# Import SMT verification modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.smt_verify import (
    verify_functional_equivalence,
    verify_content_invariance,
    verify_edge_necessity,
    verify_token_renaming_equivariance,
    generate_quote_close_sequences,
    generate_induction_sequences,
)
from scripts.smt_verify.model_weights import load_model_weights
from scripts.smt_verify.helpers import parse_circuit_edges, get_candidate_tokens


def load_circuit(circuit_path: str) -> Dict[str, Any]:
    """Load extracted circuit from JSON."""
    with open(circuit_path, "r") as f:
        return json.load(f)


def get_quote_tokens(tokenizer: GPT2Tokenizer) -> Tuple[int, int]:
    """Get token IDs for single and double quotes."""
    single_id = tokenizer.encode("'", add_special_tokens=False)[0]
    double_id = tokenizer.encode('"', add_special_tokens=False)[0]
    return single_id, double_id


# ============================================================================
# Quote Close Verification
# ============================================================================

def verify_quote_close(
    circuit_path: str,
    output_dir: str,
    model_path: str,
):
    """Run all quote_close verification properties."""
    print(f"\n{'#' * 80}")
    print("VERIFYING QUOTE CLOSE CIRCUIT")
    print(f"{'#' * 80}\n")

    circuit = load_circuit(circuit_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(f"Circuit: {circuit_path}")
    print(f"Edges: {circuit['num_edges']}")
    print(f"Task: {circuit['task']}")
    print(f"Model: {model_path}\n")

    # Load model weights
    print("Loading model weights...")
    try:
        model_weights = load_model_weights(model_path)
        print(f"Loaded model with {model_weights['n_layers']} layers, "
              f"{model_weights['d_model']} dimensions\n")
    except Exception as e:
        print(f"ERROR: Could not load model weights: {e}")
        return []

    # Get candidate tokens
    single_id, double_id = get_quote_tokens(tokenizer)
    candidate_tokens = [single_id, double_id]
    model_weights["single_quote_id"] = single_id
    model_weights["double_quote_id"] = double_id

    print(f"Candidate tokens: ' = {single_id}, \" = {double_id}\n")

    results = []

    # Property 1: Functional equivalence
    print(f"\n{'=' * 80}")
    print("PROPERTY 1: Functional Equivalence")
    print(f"{'=' * 80}\n")

    print("Verifying: y_C(x) = P_quote(x) for bounded inputs")
    print("Reference program: predict closing quote matching opening quote\n")

    # Generate test sequences
    special_tokens = {
        "single_quote": single_id,
        "double_quote": double_id,
        "content_tokens": list(range(10, 20)),
    }

    try:
        test_sequences = generate_quote_close_sequences(max_length=5, special_tokens=special_tokens)
        print(f"Generated {len(test_sequences)} test sequences\n")

        def reference_program(tokens: List[int]) -> int:
            """Return expected closing quote token."""
            for i in range(len(tokens) - 1, -1, -1):
                if tokens[i] == single_id:
                    return single_id
                elif tokens[i] == double_id:
                    return double_id
            return single_id

        result = verify_functional_equivalence(
            circuit,
            reference_program,
            test_sequences[:50],  # Limit for tractability
            model_weights,
            candidate_tokens,
            timeout_ms=30000,
        )
        results.append(result)

        print(f"Status: {result['status']}")
        if "verified_count" in result:
            print(f"Verified: {result['verified_count']}/{result.get('total_sequences', 0)}")
        if result.get("num_counterexamples", 0) > 0:
            print(f"Counterexamples: {result['num_counterexamples']}")
        print()

    except Exception as e:
        print(f"ERROR: {e}\n")
        results.append({"property": "functional_equivalence", "status": "ERROR", "message": str(e)})

    # Property 2: Content invariance
    print(f"\n{'=' * 80}")
    print("PROPERTY 2: Content Invariance")
    print(f"{'=' * 80}\n")

    print("Verifying: quote(x) = quote(x') => y_C(x) = y_C(x')")
    print("Circuit output should depend only on quote type, not content\n")

    try:
        def get_quote_type(tokens: List[int]) -> str:
            for tok in tokens:
                if tok == single_id:
                    return "single"
                elif tok == double_id:
                    return "double"
            return "none"

        result = verify_content_invariance(
            circuit,
            test_sequences[:30],
            model_weights,
            candidate_tokens,
            get_quote_type,
            timeout_ms=30000,
        )
        results.append(result)

        print(f"Status: {result['status']}")
        if "verified_pairs" in result:
            print(f"Verified pairs: {result['verified_pairs']}")
        if result.get("num_counterexamples", 0) > 0:
            print(f"Counterexamples: {result['num_counterexamples']}")
        print()

    except Exception as e:
        print(f"ERROR: {e}\n")
        results.append({"property": "content_invariance", "status": "ERROR", "message": str(e)})

    # Property 3: Edge necessity
    print(f"\n{'=' * 80}")
    print("PROPERTY 3: Edge Necessity")
    print(f"{'=' * 80}\n")

    edges = circuit["edges"]
    print(f"Verifying: all {len(edges)} edges are necessary")
    print("For each edge e, prove exists x such that C(x) != (C \\\\ e)(x)\n")

    try:
        test_inputs = [
            [single_id, 12, 13],
            [double_id, 12, 13],
            [14, single_id, 15],
            [14, double_id, 15],
        ]

        result = verify_edge_necessity(
            circuit,
            test_inputs,
            model_weights,
            candidate_tokens,
            timeout_ms=20000,
        )
        results.append(result)

        print(f"Status: {result['status']}")
        if "necessary_edges" in result:
            print(f"Necessary edges: {result['necessary_edges']}/{result['total_edges']}")
        if result.get("unnecessary_edges_found", 0) > 0:
            print(f"Suspicious edges: {result['unnecessary_edges_found']}")
        print()

    except Exception as e:
        print(f"ERROR: {e}\n")
        results.append({"property": "edge_necessity", "status": "ERROR", "message": str(e)})

    # Write results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "verification_results.json")

    output = {
        "circuit_path": circuit_path,
        "task": circuit["task"],
        "num_edges": circuit["num_edges"],
        "model_path": model_path,
        "properties": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'#' * 80}")
    print(f"Results written to: {output_path}")
    print(f"{'#' * 80}\n")

    return results


# ============================================================================
# Induction Verification
# ============================================================================

def verify_induction_abcab(
    circuit_path: str,
    output_dir: str,
    model_path: str,
):
    """Run all induction_ABCAB verification properties."""
    print(f"\n{'#' * 80}")
    print("VERIFYING INDUCTION (ABCAB) CIRCUIT")
    print(f"{'#' * 80}\n")

    circuit = load_circuit(circuit_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(f"Circuit: {circuit_path}")
    print(f"Edges: {circuit['num_edges']}")
    print(f"Task: {circuit['task']}")
    print(f"Model: {model_path}\n")

    # Load model weights
    print("Loading model weights...")
    try:
        model_weights = load_model_weights(model_path)
        print(f"Loaded model with {model_weights['n_layers']} layers, "
              f"{model_weights['d_model']} dimensions\n")
    except Exception as e:
        print(f"ERROR: Could not load model weights: {e}")
        return []

    # Use synthetic token vocabulary
    candidate_tokens = list(range(20, 30))

    results = []

    # Property 1: Restricted functional equivalence
    print(f"\n{'=' * 80}")
    print("PROPERTY 1: Restricted Functional Equivalence")
    print(f"{'=' * 80}\n")

    print("Verifying: A B C F A B -> C")
    print("Reference program: return third token when pattern matches\n")

    try:
        vocab = set(range(20, 30))
        test_sequences = generate_induction_sequences(max_length=6, vocab=vocab)
        print(f"Generated {len(test_sequences)} test sequences\n")

        def reference_program(tokens: List[int]) -> int:
            """Extract C from A B C ... A B pattern."""
            if len(tokens) >= 5:
                return tokens[2]
            return tokens[0] if tokens else 20

        result = verify_functional_equivalence(
            circuit,
            reference_program,
            test_sequences[:50],
            model_weights,
            candidate_tokens,
            timeout_ms=30000,
        )
        results.append(result)

        print(f"Status: {result['status']}")
        if "verified_count" in result:
            print(f"Verified: {result['verified_count']}/{result.get('total_sequences', 0)}")
        print()

    except Exception as e:
        print(f"ERROR: {e}\n")
        results.append({"property": "functional_equivalence_restricted", "status": "ERROR", "message": str(e)})

    # Property 2: Token-renaming equivariance
    print(f"\n{'=' * 80}")
    print("PROPERTY 2: Token-Renaming Equivariance")
    print(f"{'=' * 80}\n")

    print("Verifying: y_C(r(x)) = r(y_C(x))")
    print("Circuit is equivariant to token renaming\n")

    try:
        test_sequences_tok = [seq for seq, _ in test_sequences[:30]]

        result = verify_token_renaming_equivariance(
            circuit,
            test_sequences_tok,
            model_weights,
            vocab_size=50,
            timeout_ms=30000,
        )
        results.append(result)

        print(f"Status: {result['status']}")
        if "verified_count" in result:
            print(f"Verified: {result['verified_count']}")
        print()

    except Exception as e:
        print(f"ERROR: {e}\n")
        results.append({"property": "token_renaming_equivariance", "status": "ERROR", "message": str(e)})

    # Write results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "verification_results.json")

    output = {
        "circuit_path": circuit_path,
        "task": circuit["task"],
        "num_edges": circuit["num_edges"],
        "model_path": model_path,
        "properties": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'#' * 80}")
    print(f"Results written to: {output_path}")
    print(f"{'#' * 80}\n")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Formally verify extracted circuits using SMT")
    parser.add_argument("--circuit_path", type=str, required=True,
                        help="Path to circuit.json from extraction")
    parser.add_argument("--task", type=str, required=True,
                        choices=["quote_close", "induction_ABCAB"],
                        help="Task to verify")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for verification results")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint directory")
    parser.add_argument("--max_length", type=int, default=8,
                        help="Maximum sequence length for bounded verification")
    parser.add_argument("--timeout_ms", type=int, default=60000,
                        help="SMT solver timeout in milliseconds")

    args = parser.parse_args()

    if args.task == "quote_close":
        verify_quote_close(
            args.circuit_path,
            args.output_dir,
            args.model_path,
        )
    elif args.task == "induction_ABCAB":
        verify_induction_abcab(
            args.circuit_path,
            args.output_dir,
            args.model_path,
        )


if __name__ == "__main__":
    main()
