#!/usr/bin/env python3
"""Materialize the smallest exact-agreement circuit from a threshold sweep."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.compare_sweeps import load_sweep_results, recommend_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--synthesis_results",
        default=None,
        help=(
            "For healed-model selection, require every globally installed "
            "program synthesized for this task to occur in the circuit."
        ),
    )
    parser.add_argument(
        "--installed_programs",
        default=None,
        help=(
            "Optional programs_selected.json. When present, only this jointly "
            "accepted subset is treated as installed."
        ),
    )
    return parser.parse_args()


def circuit_nodes(path: str) -> set[str]:
    with open(os.path.join(path, "circuit.json")) as handle:
        circuit = json.load(handle)
    nodes = set()
    for edge in circuit["edges"]:
        if isinstance(edge, dict):
            nodes.update((edge["source"], edge["target"]))
        else:
            nodes.update(edge)
    return nodes


def main() -> None:
    args = parse_args()
    results = load_sweep_results(args.sweep_dir, args.task)
    if not results:
        raise RuntimeError(f"No {args.task} sweep results in {args.sweep_dir}")
    required_heads = set()
    if args.synthesis_results is not None:
        with open(args.synthesis_results) as handle:
            synthesis = json.load(handle)
        if args.installed_programs is not None:
            with open(args.installed_programs) as handle:
                installed = set(json.load(handle))
        else:
            installed = set(synthesis.get("programs", {}))
        required_heads = {
            f"attn_{key.replace('.', '_h_')}"
            for key, report in synthesis.get("tasks", {}).get(args.task, {}).items()
            if report.get("accepted") and key in installed
        }
        results = [
            result
            for result in results
            if required_heads <= circuit_nodes(result["path"])
        ]
        if not results:
            raise RuntimeError(
                f"No {args.task} sweep circuit contains installed task programs "
                f"{sorted(required_heads)}"
            )
    best = recommend_threshold(results, args.task)
    if best["projected_agreement"] < 1.0:
        raise RuntimeError("No threshold achieved exact projected agreement")
    output_dir = os.path.join(args.output_root, args.task)
    os.makedirs(output_dir, exist_ok=True)
    copied = []
    for name in ("circuit.json", "edge_log.json", "circuit.dot", "summary.txt"):
        source = os.path.join(best["path"], name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(output_dir, name))
            copied.append(name)
    selection = {
        "task": args.task,
        "source": best["path"],
        "threshold": best["threshold"],
        "projected_agreement": best["projected_agreement"],
        "num_edges": best["num_edges"],
        "copied": copied,
        "required_program_heads": sorted(required_heads),
        "selection_rule": (
            "exact projected agreement; contain all task program heads when "
            "provided; then minimum edge count"
        ),
    }
    with open(os.path.join(output_dir, "selection.json"), "w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
