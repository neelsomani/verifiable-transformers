#!/usr/bin/env python3
"""
Formal verification entrypoint for small verifiable Transformer circuits.

Uses the exhaustive small-task evaluation dataset instead of the GPT-style SMT
domain generators.
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List

try:
    from z3 import Solver, sat
except ImportError:
    print("ERROR: z3-solver not installed. Install with: pip install z3-solver")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.small import get_eval_dataset
from scripts.smt import (
    encode_circuit_forward,
    get_small_candidate_tokens,
    verify_content_invariance,
    verify_continuous_robustness,
    verify_edge_necessity,
    verify_functional_equivalence,
)
from scripts.smt.utils import parse_circuit_edges
from scripts.smt.trace import trace_circuit_forward


DEFAULT_WEIGHTS_PATH = "artifacts/small/smt_weights.json"
DEFAULT_CIRCUIT_ROOT = "artifacts/small_circuits"
DEFAULT_CHECKPOINT_PATH = "artifacts/small/checkpoint-final"


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def assert_verifiable_weights(model_weights: Dict[str, Any]) -> None:
    """Reject weights for model variants the SMT encoder does not implement."""
    expected = {
        "norm_variant": "signed_l1_band_norm",
        "attn_variant": "sparsemax",
        "activation_variant": "leaky_relu",
    }

    for key, value in expected.items():
        actual = model_weights.get(key)
        if actual != value:
            raise ValueError(
                f"SMT verification only supports {key}={value!r}, got {actual!r}"
            )


def default_circuit_path(task: str) -> str:
    """Return the conventional small-circuit path for a task."""
    return os.path.join(DEFAULT_CIRCUIT_ROOT, task, "circuit.json")


def load_small_test_sequences(task: str) -> List[tuple[List[int], int]]:
    """Load exhaustive small-task inputs and targets."""
    examples = get_eval_dataset(task)
    return [(example["input_ids"], example["target"]) for example in examples]


def z3_value_to_float(value: Any) -> float:
    """Convert a Z3 numeric value to float."""
    if hasattr(value, "as_fraction"):
        return float(value.as_fraction())
    return float(value.as_decimal(30).rstrip("?"))


def get_smt_candidate_logits(
    input_tokens: List[int],
    circuit_edges: set,
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    timeout_ms: int,
    ctx_prefix: str,
) -> Dict[int, float]:
    """Evaluate SMT circuit logits for one concrete input."""
    solver = Solver()
    solver.set("timeout", timeout_ms)

    logits_z3 = encode_circuit_forward(
        input_tokens,
        circuit_edges,
        model_weights,
        candidate_tokens,
        solver,
        ctx_prefix,
        trace=trace_circuit_forward(input_tokens, circuit_edges, model_weights, ctx_prefix),
    )

    result = solver.check()
    if result != sat:
        raise RuntimeError(f"SMT solver returned {result}, expected sat")

    model = solver.model()
    return {
        tok: z3_value_to_float(model.eval(logits_z3[tok], model_completion=True))
        for tok in candidate_tokens
        if tok in logits_z3
    }


def run_pytorch_circuit_validation(
    task: str,
    checkpoint_path: str,
    circuit: Dict[str, Any],
    candidate_tokens: List[int],
    test_sequences: List[tuple[List[int], int]],
) -> Dict[str, Any]:
    """Validate extracted circuit behavior against the exhaustive task domain."""
    import torch

    from scripts.small.extract import CircuitGraph, controlled_forward, load_model
    from scripts.small.extract_weights import load_small_config

    circuit_edges = parse_circuit_edges(circuit)
    config = load_small_config(checkpoint_path)
    graph = CircuitGraph(config.n_layers)
    device = torch.device("cpu")
    pytorch_model = load_model(checkpoint_path, config, device)

    failures = []

    print("\nPyTorch circuit validation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Examples: {len(test_sequences)} exhaustive eval examples")

    for idx, (input_tokens, expected_token) in enumerate(test_sequences):
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = controlled_forward(
                pytorch_model,
                input_ids,
                circuit_edges,
                graph,
            )[0, -1, :]

        if not torch.isfinite(logits).all():
            failures.append({
                "input": input_tokens,
                "expected": expected_token,
                "message": "non-finite logits",
            })
            continue

        candidate_logits = {tok: float(logits[tok].item()) for tok in candidate_tokens}
        predicted_token = max(candidate_tokens, key=lambda tok: candidate_logits[tok])

        if predicted_token != expected_token:
            failures.append({
                "input": input_tokens,
                "expected": expected_token,
                "predicted": predicted_token,
                "candidate_logits": candidate_logits,
            })

        if (idx + 1) % 50 == 0:
            print(f"  Checked {idx + 1}/{len(test_sequences)}")

    status = "PASSED" if not failures else "FAILED"
    print(f"Validation status: {status}")
    if failures:
        print(f"Failures: {len(failures)}")

    return {
        "property": "pytorch_circuit_validation",
        "status": status,
        "examples_checked": len(test_sequences),
        "num_failures": len(failures),
        "failures": failures[:10],
    }


def run_smt_sanity_check(
    task: str,
    checkpoint_path: str,
    circuit: Dict[str, Any],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    test_sequences: List[tuple[List[int], int]],
    num_examples: int,
    tolerance: float,
    timeout_ms: int,
) -> Dict[str, Any]:
    """Compare SMT encode_circuit_forward against PyTorch controlled_forward."""
    import torch

    from scripts.small.extract import CircuitGraph, controlled_forward, load_model
    from scripts.small.extract_weights import load_small_config

    selected_sequences = test_sequences if num_examples <= 0 else test_sequences[:num_examples]
    circuit_edges = parse_circuit_edges(circuit)

    config = load_small_config(checkpoint_path)
    graph = CircuitGraph(config.n_layers)
    device = torch.device("cpu")
    pytorch_model = load_model(checkpoint_path, config, device)

    max_logit_diff = 0.0
    failures = []
    logit_failures = []
    examples_checked = 0

    print("\nSMT-vs-PyTorch sanity check")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Examples: {len(selected_sequences)}")
    print(f"Tolerance: {tolerance}")

    for idx, (input_tokens, expected_token) in enumerate(selected_sequences):
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            pytorch_logits_full = controlled_forward(
                pytorch_model,
                input_ids,
                circuit_edges,
                graph,
            )[0, -1, :]

        pytorch_logits = {
            tok: float(pytorch_logits_full[tok].item())
            for tok in candidate_tokens
        }
        smt_logits = get_smt_candidate_logits(
            input_tokens,
            circuit_edges,
            model_weights,
            candidate_tokens,
            timeout_ms,
            f"sanity_{idx}",
        )

        diffs = {
            tok: abs(pytorch_logits[tok] - smt_logits[tok])
            for tok in candidate_tokens
        }
        example_max_diff = max(diffs.values()) if diffs else 0.0
        max_logit_diff = max(max_logit_diff, example_max_diff)

        pytorch_argmax = max(candidate_tokens, key=lambda tok: pytorch_logits[tok])
        smt_argmax = max(candidate_tokens, key=lambda tok: smt_logits[tok])

        examples_checked += 1
        print(
            f"  {idx + 1:02d}: expected={expected_token}, "
            f"torch={pytorch_argmax}, smt={smt_argmax}, "
            f"max_diff={example_max_diff:.6g}"
        )

        if pytorch_argmax != smt_argmax:
            failures.append({
                "input": input_tokens,
                "expected": expected_token,
                "pytorch_argmax": pytorch_argmax,
                "smt_argmax": smt_argmax,
                "max_logit_diff": example_max_diff,
                "pytorch_logits": pytorch_logits,
                "smt_logits": smt_logits,
            })
        elif example_max_diff > tolerance:
            logit_failures.append({
                "input": input_tokens,
                "expected": expected_token,
                "argmax": pytorch_argmax,
                "max_logit_diff": example_max_diff,
                "pytorch_logits": pytorch_logits,
                "smt_logits": smt_logits,
            })

    status = "PASSED" if not failures and not logit_failures else "FAILED"
    print(f"Sanity status: {status}")
    print(f"Max candidate logit diff: {max_logit_diff:.6g}")

    return {
        "property": "smt_pytorch_sanity",
        "status": status,
        "examples_checked": examples_checked,
        "max_logit_diff": max_logit_diff,
        "tolerance": tolerance,
        "num_failures": len(failures),
        "num_logit_failures": len(logit_failures),
        "failures": failures[:10],
        "logit_failures": logit_failures[:10],
    }


def make_feature_fn(task: str) -> Callable[[List[int]], Any]:
    """Group examples by the task-relevant output feature."""
    expected_by_input = {
        tuple(input_ids): target
        for input_ids, target in load_small_test_sequences(task)
    }

    def feature_fn(tokens: List[int]) -> Any:
        return expected_by_input[tuple(tokens)]

    return feature_fn


def run_property(
    property_name: str,
    circuit: Dict[str, Any],
    test_sequences: List[tuple[List[int], int]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    timeout_ms: int,
    epsilon: float,
) -> Dict[str, Any]:
    """Run one verification property."""
    test_inputs = [seq for seq, _ in test_sequences]

    if property_name == "functional_equivalence":
        expected_by_input = {
            tuple(input_ids): target
            for input_ids, target in test_sequences
        }

        def reference_program(tokens: List[int]) -> int:
            return expected_by_input[tuple(tokens)]

        return verify_functional_equivalence(
            circuit,
            reference_program,
            test_sequences,
            model_weights,
            candidate_tokens,
            timeout_ms=timeout_ms,
        )

    if property_name == "content_invariance":
        return verify_content_invariance(
            circuit,
            test_sequences,
            model_weights,
            candidate_tokens,
            make_feature_fn(circuit["task"]),
            timeout_ms=timeout_ms,
        )

    if property_name == "edge_necessity":
        return verify_edge_necessity(
            circuit,
            test_inputs,
            model_weights,
            candidate_tokens,
            timeout_ms=timeout_ms,
        )

    if property_name == "continuous_robustness":
        return verify_continuous_robustness(
            circuit,
            test_inputs,
            model_weights,
            candidate_tokens,
            epsilon=epsilon,
            timeout_ms=timeout_ms,
        )

    raise ValueError(f"Unknown property: {property_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formally verify small verifiable Transformer circuits using SMT"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["quote_close", "bracket_type"],
        help="Small-model task to verify",
    )
    parser.add_argument(
        "--circuit_path",
        type=str,
        default=None,
        help="Path to circuit.json. Defaults to artifacts/small_circuits/<task>/circuit.json",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Path to SMT weights JSON (default: {DEFAULT_WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to small model checkpoint for sanity checks (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for verification results",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["functional_equivalence", "edge_necessity"],
        choices=[
            "functional_equivalence",
            "content_invariance",
            "edge_necessity",
            "continuous_robustness",
        ],
        help="Verification properties to run",
    )
    parser.add_argument(
        "--timeout_ms",
        type=int,
        default=60000,
        help="SMT solver timeout per query in milliseconds",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Perturbation radius for continuous robustness",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run exhaustive PyTorch circuit validation before formal properties",
    )
    parser.add_argument(
        "--smt_sanity_check",
        action="store_true",
        help="Run expensive SMT-vs-PyTorch forward comparison before formal properties",
    )
    parser.add_argument(
        "--sanity_examples",
        type=int,
        default=8,
        help="Number of eval examples for sanity check; use <=0 for all",
    )
    parser.add_argument(
        "--sanity_tolerance",
        type=float,
        default=1e-3,
        help="Reported tolerance for sanity-check candidate logit differences",
    )
    parser.add_argument(
        "--max_inputs",
        type=int,
        default=None,
        help="Limit verification to the first N eval examples for debugging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    circuit_path = args.circuit_path or default_circuit_path(args.task)

    circuit = load_json(circuit_path)
    model_weights = load_json(args.weights_path)
    assert_verifiable_weights(model_weights)
    candidate_info = get_small_candidate_tokens(args.task)
    candidate_tokens = candidate_info["candidates"]
    test_sequences = load_small_test_sequences(args.task)
    if args.max_inputs is not None:
        test_sequences = test_sequences[:args.max_inputs]

    print("Small SMT Circuit Verification")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Circuit: {circuit_path}")
    print(f"Weights: {args.weights_path}")
    print(f"Inputs: {len(test_sequences)} exhaustive eval examples")
    print(f"Candidates: {candidate_tokens} ({candidate_info['names']})")
    print(f"Properties: {', '.join(args.properties)}")
    print("=" * 60)

    results = []
    if args.sanity_check:
        try:
            sanity_result = run_pytorch_circuit_validation(
                args.task,
                args.checkpoint,
                circuit,
                candidate_tokens,
                test_sequences,
            )
        except Exception as e:
            sanity_result = {
                "property": "pytorch_circuit_validation",
                "status": "ERROR",
                "message": str(e),
            }

        results.append(sanity_result)
        if sanity_result.get("status") != "PASSED":
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, "verification_results.json")
            with open(output_path, "w") as f:
                json.dump({
                    "task": args.task,
                    "circuit_path": circuit_path,
                    "weights_path": args.weights_path,
                    "checkpoint": args.checkpoint,
                    "num_inputs": len(test_sequences),
                    "candidate_tokens": candidate_tokens,
                    "candidate_names": candidate_info["names"],
                    "properties": results,
                }, f, indent=2)
            print(f"\nPyTorch circuit validation failed; results written to: {output_path}")
            sys.exit(1)

    if args.smt_sanity_check:
        try:
            sanity_result = run_smt_sanity_check(
                args.task,
                args.checkpoint,
                circuit,
                model_weights,
                candidate_tokens,
                test_sequences,
                args.sanity_examples,
                args.sanity_tolerance,
                args.timeout_ms,
            )
        except Exception as e:
            sanity_result = {
                "property": "smt_pytorch_sanity",
                "status": "ERROR",
                "message": str(e),
            }

        results.append(sanity_result)
        if sanity_result.get("status") != "PASSED":
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, "verification_results.json")
            with open(output_path, "w") as f:
                json.dump({
                    "task": args.task,
                    "circuit_path": circuit_path,
                    "weights_path": args.weights_path,
                    "checkpoint": args.checkpoint,
                    "num_inputs": len(test_sequences),
                    "candidate_tokens": candidate_tokens,
                    "candidate_names": candidate_info["names"],
                    "properties": results,
                }, f, indent=2)
            print(f"\nSanity check failed; results written to: {output_path}")
            sys.exit(1)

    for property_name in args.properties:
        print(f"\n{'=' * 80}")
        print(f"PROPERTY: {property_name}")
        print(f"{'=' * 80}\n")

        try:
            result = run_property(
                property_name,
                circuit,
                test_sequences,
                model_weights,
                candidate_tokens,
                args.timeout_ms,
                args.epsilon,
            )
        except Exception as e:
            result = {
                "property": property_name,
                "status": "ERROR",
                "message": str(e),
            }

        results.append(result)
        print(f"Status: {result.get('status', 'UNKNOWN')}")
        if result.get("message"):
            print(f"Message: {result['message']}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "verification_results.json")
    output = {
        "task": args.task,
        "circuit_path": circuit_path,
        "weights_path": args.weights_path,
        "checkpoint": args.checkpoint,
        "num_inputs": len(test_sequences),
        "candidate_tokens": candidate_tokens,
        "candidate_names": candidate_info["names"],
        "properties": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
