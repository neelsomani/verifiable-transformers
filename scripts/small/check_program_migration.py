#!/usr/bin/env python3
"""Check that healed program heads are necessary and have no behavioral bypass."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.circuits import CircuitGraph
from scripts.programs import load_programs
from scripts.small.config import SmallVerifiableConfig
from scripts.small.extract import evaluate_circuit, load_model


HEAD_PATTERN = re.compile(r"^attn_(\d+)_h_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--circuit_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def edge_set(circuit: dict) -> set[tuple[str, str]]:
    return {
        (edge["source"], edge["target"])
        if isinstance(edge, dict)
        else tuple(edge)
        for edge in circuit["edges"]
    }


def main() -> None:
    args = parse_args()
    config = SmallVerifiableConfig.load(
        os.path.join(os.path.dirname(args.checkpoint), "config.json")
    )
    programs = load_programs(os.path.join(args.checkpoint, "programs.json"))
    program_nodes = {
        f"attn_{layer}_h_{head}" for layer, head in programs
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, config, device)
    graph = CircuitGraph(config.n_layers, config.n_heads)
    full_edges = graph.get_edges()
    task_reports = {}

    for task in ("quote_close", "bracket_type"):
        with open(os.path.join(args.circuit_root, task, "circuit.json")) as handle:
            circuit = json.load(handle)
        circuit_edges = edge_set(circuit)
        intended = sorted(
            node
            for node in program_nodes
            if any(node in edge for edge in circuit_edges)
        )
        if not intended:
            raise RuntimeError(f"{task} circuit contains no installed program head")

        ablations = {}
        for node in intended:
            without_node = {edge for edge in full_edges if edge[0] != node}
            metrics = evaluate_circuit(model, task, without_node, graph, device)
            ablations[node] = {
                "metrics": metrics,
                "necessary": metrics["agreement"] < 1.0,
            }

        without_intended = {
            edge for edge in full_edges if edge[0] not in intended
        }
        without_all_programs = {
            edge for edge in full_edges if edge[0] not in program_nodes
        }
        intended_metrics = evaluate_circuit(
            model, task, without_intended, graph, device
        )
        all_metrics = evaluate_circuit(
            model, task, without_all_programs, graph, device
        )
        task_reports[task] = {
            "intended_program_heads": intended,
            "circuit_edges": len(circuit_edges),
            "full_metrics": evaluate_circuit(model, task, full_edges, graph, device),
            "individual_ablations": ablations,
            "without_intended_program_heads": intended_metrics,
            "without_all_program_heads": all_metrics,
            "necessary": all(value["necessary"] for value in ablations.values()),
            "non_bypassed": intended_metrics["agreement"] < 1.0,
        }

    output = {
        "checkpoint": args.checkpoint,
        "circuit_root": args.circuit_root,
        "program_nodes": sorted(program_nodes),
        "criterion": (
            "Every intended program head must be individually necessary and the "
            "remaining full graph outside those intended heads must have "
            "projected agreement < 1.0."
        ),
        "tasks": task_reports,
    }
    output["migration_pass"] = all(
        value["necessary"] and value["non_bypassed"]
        for value in task_reports.values()
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))
    if not output["migration_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
