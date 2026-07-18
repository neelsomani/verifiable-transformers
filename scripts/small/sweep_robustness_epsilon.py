#!/usr/bin/env python3
"""Compare matched BandNorm and norm-free robustness over epsilon."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.small.verify import load_small_test_sequences
from scripts.smt import get_small_candidate_tokens, verify_continuous_robustness


RUNS = {
    "bandnorm": {
        "circuit_root": Path("artifacts/small_band_norm_matched_circuits"),
        "weights": Path("artifacts/small_band_norm_matched/smt_weights.json"),
    },
    "norm_free": {
        "circuit_root": Path("artifacts/small_norm_free_circuits"),
        "weights": Path("artifacts/small_norm_free/smt_weights.json"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/robustness_eps_sweep.json")
    parser.add_argument("--timeout_ms", type=int, default=60000)
    return parser.parse_args()


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def robustness_property(path: Path) -> dict:
    payload = load(path)
    return next(
        value
        for value in payload["properties"]
        if value["property"] == "projected_continuous_robustness"
    )


def classify_bandnorm(results: list[dict]) -> str:
    """Distinguish a finite margin from branch adjacency at the smallest radius."""
    if results[0]["status"] == "UNKNOWN_BRANCH_UNSTABLE":
        return "branch-adjacent"
    if any(value["status"] == "VERIFIED" for value in results[:-1]) and results[-1][
        "status"
    ] == "UNKNOWN_BRANCH_UNSTABLE":
        return "margin-thin"
    if all(value["status"] == "VERIFIED" for value in results):
        return "certified-through-epsilon0"
    return "other"


def compact(result: dict, epsilon: float, wall_seconds: float) -> dict:
    return {
        "epsilon": epsilon,
        "status": result["status"],
        "verified_count": result["verified_count"],
        "timeout_count": result["timeout_count"],
        "error_count": result["error_count"],
        "num_decision_violations": result["num_decision_violations"],
        "num_branch_unstable": result["num_branch_unstable"],
        "branch_certificates_required": result["branch_certificates_required"],
        "assertion_count": result["assertion_count"],
        "assertion_attribution": result["assertion_attribution"],
        "solve_seconds": result["solve_seconds"],
        "wall_time_seconds": wall_seconds,
    }


def main() -> None:
    args = parse_args()
    original_sources = {
        task: Path("artifacts/small_circuits")
        / task
        / "verification"
        / "verification_results.json"
        for task in ("quote_close", "bracket_type")
    }
    matched_sources = {
        task: Path("artifacts/small_band_norm_matched_circuits")
        / task
        / "verification"
        / "verification_results.json"
        for task in ("quote_close", "bracket_type")
    }
    original_eps = {
        task: robustness_property(path)["certified_epsilon"]
        for task, path in original_sources.items()
    }
    matched_eps = {
        task: robustness_property(path)["certified_epsilon"]
        for task, path in matched_sources.items()
    }
    eps0_values = set(original_eps.values())
    if len(eps0_values) != 1:
        raise RuntimeError(f"Original Table 2 epsilons differ: {original_eps}")
    epsilon0 = eps0_values.pop()
    if any(value != epsilon0 for value in matched_eps.values()):
        raise RuntimeError(
            f"Matched and original epsilon differ: {matched_eps} vs {epsilon0}"
        )
    epsilons = [epsilon0 / 10, epsilon0 / 4, epsilon0 / 2, epsilon0]

    payload = {
        "epsilon_confirmation": {
            "original_table2": original_eps,
            "matched": matched_eps,
            "equal": True,
            "epsilon0": epsilon0,
            "original_sources": {
                task: str(path) for task, path in original_sources.items()
            },
            "matched_sources": {
                task: str(path) for task, path in matched_sources.items()
            },
        },
        "sweep_definition": ["epsilon0/10", "epsilon0/4", "epsilon0/2", "epsilon0"],
        "classification_definition": {
            "margin-thin": (
                "At least one smaller radius is certified, but epsilon0 reaches a "
                "BandNorm branch boundary."
            ),
            "branch-adjacent": (
                "The smallest tested positive radius already reaches a BandNorm "
                "branch boundary."
            ),
        },
        "runs": {},
    }
    for run_name, paths in RUNS.items():
        weights = load(paths["weights"])
        payload["runs"][run_name] = {}
        for task in ("quote_close", "bracket_type"):
            circuit = load(paths["circuit_root"] / task / "circuit.json")
            examples = load_small_test_sequences(task)
            candidates = get_small_candidate_tokens(task)["candidates"]
            sweep = []
            for epsilon in epsilons:
                started = time.perf_counter()
                result = verify_continuous_robustness(
                    circuit,
                    [tokens for tokens, _ in examples],
                    weights,
                    candidates,
                    epsilon=epsilon,
                    timeout_ms=args.timeout_ms,
                )
                sweep.append(compact(result, epsilon, time.perf_counter() - started))
            task_result = {"results": sweep}
            if run_name == "bandnorm":
                task_result["classification"] = classify_bandnorm(sweep)
            payload["runs"][run_name][task] = task_result

    payload["cause_classification"] = {
        task: payload["runs"]["bandnorm"][task]["classification"]
        for task in ("quote_close", "bracket_type")
    }
    payload["norm_free_contrast"] = (
        "No branch classification applies: norm-free robustness has no norm "
        "branch certificate."
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
