#!/usr/bin/env python3
"""
Sanity test: verify SMT encoder matches PyTorch forward pass.

Tests the SMT encoder on short sequences and compares candidate logits
against PyTorch outputs to ensure correctness before trusting verification results.
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import torch
import numpy as np
from z3 import Solver, sat
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.smt import encode_circuit_forward
from scripts.gpt2.model_weights import load_model_weights
from scripts.smt.utils import parse_circuit_edges
from scripts.gpt2.extract import (
    controlled_forward,
    build_circuit_graph,
)


def load_pytorch_model(model_path: str, model_info_path: str = None):
    """Load PyTorch model for reference."""
    from scripts.gpt2.train import apply_model_variants
    from transformers import GPT2Config

    # Load model_info.json
    if model_info_path is None:
        model_info_path = os.path.join(model_path, "model_info.json")
        if not os.path.exists(model_info_path):
            parent_dir = os.path.dirname(model_path)
            model_info_path = os.path.join(parent_dir, "model_info.json")

    with open(model_info_path, "r") as f:
        model_info = json.load(f)

    norm_variant = model_info.get("norm_variant", "layernorm")
    attn_variant = model_info.get("attn_variant", "softmax")
    activation_variant = model_info.get("activation_variant", "gelu")

    # Load config
    config = GPT2Config.from_pretrained(model_path)

    # Apply activation variant
    if activation_variant == "leaky_relu":
        config.activation_function = "leaky_relu"
    elif activation_variant == "relu":
        config.activation_function = "relu"

    # Create model
    model = GPT2LMHeadModel(config)

    # Apply variants
    apply_model_variants(
        model,
        norm_variant=norm_variant,
        attn_variant=attn_variant,
        activation_variant=activation_variant,
    )

    # Load weights
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_path, "model.safetensors")

    if os.path.exists(weights_path):
        if weights_path.endswith(".bin"):
            state_dict = torch.load(weights_path, map_location="cpu")
        else:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)

        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Model weights not found in {model_path}")

    model.eval()
    return model


def get_pytorch_logits(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    circuit_edges: set,
) -> torch.Tensor:
    """Get logits from PyTorch circuit using controlled_forward."""
    graph = build_circuit_graph(model.config.n_layer)

    with torch.no_grad():
        logits = controlled_forward(
            model=model,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            edges_to_keep=circuit_edges,
            graph=graph,
            ablation_cache=None,
            ablation_mode="zero",
            return_node_outputs=False,
        )
        return logits[0, -1, :]  # Last position logits


def get_smt_logits(
    input_tokens: List[int],
    circuit_edges: set,
    model_weights: Dict,
    candidate_tokens: List[int],
) -> Dict[int, float]:
    """Get logits from SMT encoder."""
    import time

    solver = Solver()
    solver.set("timeout", 30000)

    # Encode circuit forward pass
    print(f"     Encoding circuit...", end="", flush=True)
    start = time.time()
    logits_z3 = encode_circuit_forward(
        input_tokens,
        circuit_edges,
        model_weights,
        candidate_tokens,
        solver,
        "test",
    )
    encode_time = time.time() - start
    num_constraints = len(solver.assertions())
    print(f" done ({encode_time:.1f}s, {num_constraints} constraints)", flush=True)

    # Check satisfiability
    print(f"     Solving constraints...", end="", flush=True)
    start = time.time()
    result = solver.check()
    solve_time = time.time() - start
    print(f" done ({solve_time:.1f}s)", flush=True)

    if result != sat:
        raise RuntimeError(f"SMT solver returned {result}, expected sat")

    # Extract logit values
    model = solver.model()
    logits = {}
    for tok in candidate_tokens:
        if tok in logits_z3:
            val = model.eval(logits_z3[tok], model_completion=True)
            # Convert Z3 rational to float
            if hasattr(val, "as_fraction"):
                logits[tok] = float(val.as_fraction())
            else:
                logits[tok] = float(val.as_decimal(20).rstrip("?"))

    return logits


def test_encoder_sanity(
    model_path: str,
    circuit_path: str,
    candidate_tokens: List[int],
    test_sequences: List[List[int]],
    tolerance: float = 1e-3,
):
    """Test SMT encoder against PyTorch on test sequences.

    Args:
        model_path: Path to model checkpoint
        circuit_path: Path to circuit.json
        candidate_tokens: Token IDs to compare
        test_sequences: List of input token sequences
        tolerance: Maximum allowed difference in logits
    """
    print(f"\n{'='*80}")
    print("SMT ENCODER SANITY TEST")
    print(f"{'='*80}\n")

    # Load circuit
    with open(circuit_path, "r") as f:
        circuit = json.load(f)
    circuit_edges = parse_circuit_edges(circuit)

    print(f"Circuit: {circuit_path}")
    print(f"Edges: {circuit['num_edges']}")
    print(f"Candidate tokens: {candidate_tokens}")
    print(f"Test sequences: {len(test_sequences)}")
    print(f"Tolerance: {tolerance}\n")

    # Load models
    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model(model_path)

    print("Loading SMT model weights...")
    model_weights = load_model_weights(model_path)
    print()

    # Test each sequence
    all_pass = True
    for i, input_tokens in enumerate(test_sequences):
        print(f"Test {i+1}/{len(test_sequences)}: {input_tokens}")

        # Get PyTorch logits
        input_ids = torch.tensor([input_tokens])
        pytorch_logits_full = get_pytorch_logits(pytorch_model, input_ids, circuit_edges)
        pytorch_logits = {tok: pytorch_logits_full[tok].item() for tok in candidate_tokens}

        # Get SMT logits
        try:
            smt_logits = get_smt_logits(
                input_tokens,
                circuit_edges,
                model_weights,
                candidate_tokens,
            )
        except Exception as e:
            print(f"  ❌ FAILED: SMT encoding error: {e}\n")
            all_pass = False
            continue

        # Compare logits
        max_diff = 0.0
        mismatches = []

        for tok in candidate_tokens:
            pytorch_val = pytorch_logits.get(tok, 0.0)
            smt_val = smt_logits.get(tok, 0.0)
            diff = abs(pytorch_val - smt_val)
            max_diff = max(max_diff, diff)

            if diff > tolerance:
                mismatches.append((tok, pytorch_val, smt_val, diff))

        # Check projected decision (argmax) agreement
        pt_argmax = max(candidate_tokens, key=lambda t: pytorch_logits.get(t, float('-inf')))
        smt_argmax = max(candidate_tokens, key=lambda t: smt_logits.get(t, float('-inf')))

        decision_mismatch = (pt_argmax != smt_argmax)

        if decision_mismatch:
            print(f"  ❌ DECISION MISMATCH: PyTorch={pt_argmax}, SMT={smt_argmax}")
            print(f"     Max logit diff = {max_diff:.6f}")
            if mismatches:
                for tok, pt_val, smt_val, diff in mismatches:
                    print(f"     Token {tok}: PyTorch={pt_val:.6f}, SMT={smt_val:.6f}, diff={diff:.6f}")
            print()
            all_pass = False
        elif mismatches:
            print(f"  ⚠️  Logit differences found but decision agrees: {pt_argmax}")
            print(f"     Max diff = {max_diff:.6f}")
            for tok, pt_val, smt_val, diff in mismatches:
                print(f"     Token {tok}: PyTorch={pt_val:.6f}, SMT={smt_val:.6f}, diff={diff:.6f}")
            print()
            # Don't fail if decision agrees
        else:
            print(f"  ✓ Decision match: {pt_argmax}, max diff = {max_diff:.6f}")
            print()

    # Summary
    print(f"{'='*80}")
    if all_pass:
        print("✓ ALL TESTS PASSED")
        print("SMT encoder projected decisions match PyTorch on tested sequences")
    else:
        print("❌ SOME TESTS FAILED")
        print("SMT encoder decisions do not match PyTorch!")
        print("DO NOT TRUST VERIFICATION RESULTS UNTIL THIS IS FIXED")
    print(f"{'='*80}\n")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Test SMT encoder against PyTorch")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--circuit_path", type=str, required=True,
                        help="Path to circuit.json")
    parser.add_argument("--task", type=str, required=True,
                        choices=["quote_close", "bracket_type"],
                        help="Task to test")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Maximum allowed logit difference")

    args = parser.parse_args()

    # Setup tokenizer and test sequences
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if args.task == "quote_close":
        single_id = tokenizer.encode("'", add_special_tokens=False)[0]
        double_id = tokenizer.encode('"', add_special_tokens=False)[0]
        candidate_tokens = [single_id, double_id]

        # Test sequences
        test_sequences = [
            [single_id, 12, 13],
            [double_id, 12, 13],
            [14, single_id],
            [14, double_id],
            [single_id],
        ]

    elif args.task == "bracket_type":
        left_bracket = tokenizer.encode("[", add_special_tokens=False)[0]
        left_brace = tokenizer.encode("{", add_special_tokens=False)[0]
        right_bracket = tokenizer.encode("]", add_special_tokens=False)[0]
        right_brace = tokenizer.encode("}", add_special_tokens=False)[0]
        candidate_tokens = [right_bracket, right_brace]

        # Test sequences
        test_sequences = [
            [left_bracket, 12, 13],
            [left_brace, 12, 13],
            [14, left_bracket],
            [14, left_brace],
            [left_bracket],
        ]

    # Run tests
    success = test_encoder_sanity(
        args.model_path,
        args.circuit_path,
        candidate_tokens,
        test_sequences,
        args.tolerance,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
