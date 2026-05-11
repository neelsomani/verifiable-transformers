#!/usr/bin/env python3
"""
Formal verification of extracted circuits using SMT solvers.

Verifies properties like:
- Functional equivalence to symbolic reference program
- Content invariance
- Edge necessity
- Continuous robustness
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
    verify_continuous_robustness,
    generate_quote_close_sequences,
    generate_bracket_type_sequences,
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


def get_bracket_tokens(tokenizer: GPT2Tokenizer) -> Tuple[int, int, int, int]:
    """Get token IDs for [, {, ], }."""
    left_bracket = tokenizer.encode("[", add_special_tokens=False)[0]
    left_brace = tokenizer.encode("{", add_special_tokens=False)[0]
    right_bracket = tokenizer.encode("]", add_special_tokens=False)[0]
    right_brace = tokenizer.encode("}", add_special_tokens=False)[0]
    return left_bracket, left_brace, right_bracket, right_brace


# ============================================================================
# Quote Close Verification
# ============================================================================

def verify_quote_close(
    circuit_path: str,
    output_dir: str,
    model_path: str,
    max_length: int,
    timeout_ms: int,
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
        "content_tokens": list(range(10, 12)),  # Small vocab for exhaustive verification
    }

    try:
        test_sequences = generate_quote_close_sequences(max_length=max_length, special_tokens=special_tokens)
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
            test_sequences,
            model_weights,
            candidate_tokens,
            timeout_ms=timeout_ms,
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
            test_sequences,
            model_weights,
            candidate_tokens,
            get_quote_type,
            timeout_ms=timeout_ms,
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
        # Use bounded domain sequences for formal verification
        test_inputs = [seq for seq, _ in test_sequences]

        result = verify_edge_necessity(
            circuit,
            test_inputs,
            model_weights,
            candidate_tokens,
            timeout_ms=timeout_ms,
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

    # Property 4: Continuous robustness
    print(f"\n{'=' * 80}")
    print("PROPERTY 4: Continuous Robustness")
    print(f"{'=' * 80}\n")

    print("Verifying: decision is stable under perturbations to final residual")
    print("For all x, all η with ||η||_∞ ≤ ε: g_T(r_E(x)+η) = g_T(r_E(x))\n")

    try:
        # Use bounded domain sequences for formal verification
        test_inputs = [seq for seq, _ in test_sequences]

        result = verify_continuous_robustness(
            circuit,
            test_inputs,
            model_weights,
            candidate_tokens,
            epsilon=0.01,
            timeout_ms=timeout_ms,
        )
        results.append(result)

        print(f"Status: {result['status']}")
        print(f"Epsilon: {result.get('epsilon', 0.01)}")
        if "verified_count" in result:
            print(f"Verified: {result['verified_count']}/{len(test_inputs)}")
        if result.get("num_violations", 0) > 0:
            print(f"Violations found: {result['num_violations']}")
        print()

    except Exception as e:
        print(f"ERROR: {e}\n")
        results.append({"property": "continuous_robustness", "status": "ERROR", "message": str(e)})

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
# Bracket Type Verification
# ============================================================================

def verify_bracket_type(
    circuit_path: str,
    output_dir: str,
    model_path: str,
    max_length: int,
    timeout_ms: int,
):
    """Run all bracket_type verification properties."""
    print(f"\n{'#' * 80}")
    print("VERIFYING BRACKET TYPE CIRCUIT")
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
    left_bracket, left_brace, right_bracket, right_brace = get_bracket_tokens(tokenizer)
    candidate_tokens = [right_bracket, right_brace]
    model_weights["left_bracket_id"] = left_bracket
    model_weights["left_brace_id"] = left_brace
    model_weights["right_bracket_id"] = right_bracket
    model_weights["right_brace_id"] = right_brace

    print(f"Candidate tokens: [ = {left_bracket}, {{ = {left_brace}, "
          f"] = {right_bracket}, }} = {right_brace}\n")

    results = []

    # Property 1: Functional equivalence
    print(f"\n{'=' * 80}")
    print("PROPERTY 1: Functional Equivalence")
    print(f"{'=' * 80}\n")

    print("Verifying: y_C(x) = P_bracket(x) for bounded inputs")
    print("Reference program: predict closing bracket matching opening bracket\n")

    # Generate test sequences
    special_tokens = {
        "left_bracket": left_bracket,
        "left_brace": left_brace,
        "right_bracket": right_bracket,
        "right_brace": right_brace,
        "content_tokens": list(range(10, 12)),  # Small vocab for exhaustive verification
    }

    try:
        test_sequences = generate_bracket_type_sequences(max_length=max_length, special_tokens=special_tokens)
        print(f"Generated {len(test_sequences)} test sequences\n")

        def reference_program(tokens: List[int]) -> int:
            """Return expected closing bracket token."""
            for i in range(len(tokens) - 1, -1, -1):
                if tokens[i] == left_bracket:
                    return right_bracket
                elif tokens[i] == left_brace:
                    return right_brace
            return right_bracket

        result = verify_functional_equivalence(
            circuit,
            reference_program,
            test_sequences,
            model_weights,
            candidate_tokens,
            timeout_ms=timeout_ms,
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

    print("Verifying: bracket(x) = bracket(x') => y_C(x) = y_C(x')")
    print("Circuit output should depend only on bracket type, not content\n")

    try:
        def get_bracket_type(tokens: List[int]) -> str:
            for tok in tokens:
                if tok == left_bracket:
                    return "bracket"
                elif tok == left_brace:
                    return "brace"
            return "none"

        result = verify_content_invariance(
            circuit,
            test_sequences,
            model_weights,
            candidate_tokens,
            get_bracket_type,
            timeout_ms=timeout_ms,
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
        # Use bounded domain sequences for formal verification
        test_inputs = [seq for seq, _ in test_sequences]

        result = verify_edge_necessity(
            circuit,
            test_inputs,
            model_weights,
            candidate_tokens,
            timeout_ms=timeout_ms,
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

    # Property 4: Continuous robustness
    print(f"\n{'=' * 80}")
    print("PROPERTY 4: Continuous Robustness")
    print(f"{'=' * 80}\n")

    print("Verifying: decision is stable under perturbations to final residual")
    print("For all x, all η with ||η||_∞ ≤ ε: g_T(r_E(x)+η) = g_T(r_E(x))\n")

    try:
        # Use bounded domain sequences for formal verification
        test_inputs = [seq for seq, _ in test_sequences]

        result = verify_continuous_robustness(
            circuit,
            test_inputs,
            model_weights,
            candidate_tokens,
            epsilon=0.01,
            timeout_ms=timeout_ms,
        )
        results.append(result)

        print(f"Status: {result['status']}")
        print(f"Epsilon: {result.get('epsilon', 0.01)}")
        if "verified_count" in result:
            print(f"Verified: {result['verified_count']}/{len(test_inputs)}")
        if result.get("num_violations", 0) > 0:
            print(f"Violations found: {result['num_violations']}")
        print()

    except Exception as e:
        print(f"ERROR: {e}\n")
        results.append({"property": "continuous_robustness", "status": "ERROR", "message": str(e)})

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
                        choices=["quote_close", "bracket_type"],
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
            args.max_length,
            args.timeout_ms,
        )
    elif args.task == "bracket_type":
        verify_bracket_type(
            args.circuit_path,
            args.output_dir,
            args.model_path,
            args.max_length,
            args.timeout_ms,
        )


if __name__ == "__main__":
    main()
