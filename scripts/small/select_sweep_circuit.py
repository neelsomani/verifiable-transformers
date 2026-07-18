#!/usr/bin/env python3
"""Select the smallest exact small-model circuit containing intended programs."""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--required_head",
        action="append",
        default=[],
        help="Required node as <layer>.<head> or attn_<layer>_h_<head>.",
    )
    return parser.parse_args()


def node_name(value: str) -> str:
    if value.startswith("attn_"):
        return value
    layer, head = (int(part) for part in value.split("."))
    return f"attn_{layer}_h_{head}"


def circuit_nodes(circuit: dict) -> set[str]:
    nodes = set()
    for edge in circuit["edges"]:
        if isinstance(edge, dict):
            nodes.update((edge["source"], edge["target"]))
        else:
            nodes.update(edge)
    return nodes


def main() -> None:
    args = parse_args()
    with open(args.sweep) as handle:
        entries = json.load(handle)
    required = {node_name(value) for value in args.required_head}
    eligible = [
        entry
        for entry in entries
        if entry["metrics"]["agreement"] == 1.0
        and required <= circuit_nodes(entry["circuit"])
    ]
    if not eligible:
        raise RuntimeError(
            f"No exact-agreement circuit contains required heads {sorted(required)}"
        )
    selected = min(
        eligible,
        key=lambda entry: (
            entry["n_edges"],
            entry["metrics"]["mean_candidate_kl"],
            entry["threshold"],
        ),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "circuit.json"), "w") as handle:
        json.dump(selected["circuit"], handle, indent=2)
    with open(os.path.join(args.output_dir, "edge_log.json"), "w") as handle:
        json.dump(selected["edge_log"], handle, indent=2)
    with open(os.path.join(args.output_dir, "circuit.dot"), "w") as handle:
        handle.write("digraph circuit {\n  rankdir=LR;\n")
        for edge in selected["circuit"]["edges"]:
            source = edge["source"] if isinstance(edge, dict) else edge[0]
            target = edge["target"] if isinstance(edge, dict) else edge[1]
            handle.write(f'  "{source}" -> "{target}";\n')
        handle.write("}\n")
    selection = {
        "sweep": args.sweep,
        "required_heads": sorted(required),
        "selection_rule": (
            "exact projected agreement, contains all preregistered intended "
            "program heads, then minimum edges/KL/threshold"
        ),
        "threshold": selected["threshold"],
        "n_edges": selected["n_edges"],
        "metrics": selected["metrics"],
    }
    with open(os.path.join(args.output_dir, "selection.json"), "w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
