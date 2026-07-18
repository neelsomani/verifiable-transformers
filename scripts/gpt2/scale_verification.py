#!/usr/bin/env python3
"""Increase GPT-2 circuit proof length until the first non-verification result."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--circuit_path", required=True)
    parser.add_argument("--task", required=True, choices=["quote_close", "bracket_type"])
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--start_length", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=8)
    parser.add_argument("--timeout_ms", type=int, default=60000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    attempts = []
    largest_verified = None
    for length in range(args.start_length, args.max_length + 1):
        output_dir = os.path.join(args.output_root, f"length_{length}")
        command = [
            sys.executable,
            "scripts/gpt2/verify.py",
            "--model_path",
            args.model_path,
            "--circuit_path",
            args.circuit_path,
            "--task",
            args.task,
            "--max_length",
            str(length),
            "--timeout_ms",
            str(args.timeout_ms),
            "--output_dir",
            output_dir,
        ]
        completed = subprocess.run(command, check=False)
        result_path = os.path.join(output_dir, "verification_results.json")
        if os.path.exists(result_path):
            with open(result_path) as handle:
                result = json.load(handle)
            statuses = [value.get("status") for value in result["properties"]]
        else:
            statuses = ["MISSING_RESULT"]
        passed = completed.returncode == 0 and statuses and all(
            status == "VERIFIED" for status in statuses
        )
        attempts.append(
            {
                "length": length,
                "returncode": completed.returncode,
                "statuses": statuses,
                "passed": passed,
                "result_path": result_path,
            }
        )
        if not passed:
            break
        largest_verified = length

    summary = {
        "task": args.task,
        "model_path": args.model_path,
        "circuit_path": args.circuit_path,
        "largest_fully_verified_length": largest_verified,
        "stopped_at_first_nonverification": True,
        "attempts": attempts,
    }
    with open(os.path.join(args.output_root, "scaling_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
