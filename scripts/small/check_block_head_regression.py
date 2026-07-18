#!/usr/bin/env python3
"""Re-run block-level extraction and prove its retained-head union is identical."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.circuits import (
    CircuitGraph,
    controlled_forward,
    controlled_forward_block,
    expand_block_edges,
)
from scripts.small import get_eval_dataset, vocab
from scripts.small.config import SmallVerifiableConfig
from scripts.small.extract import (
    cleanup_graph,
    compute_candidate_kl,
    compute_projected_agreement,
    load_model,
)


TASKS = ("quote_close", "bracket_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--min_agreement", type=float, default=1.0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def extract_block_circuit(
    model,
    input_ids: torch.Tensor,
    candidates: list[int],
    graph: CircuitGraph,
    threshold: float,
    min_agreement: float,
) -> tuple[set[tuple[str, str]], list[dict], torch.Tensor]:
    """Legacy ACDC pass followed by a projected-decision fixed-point trim."""
    with torch.no_grad():
        full = controlled_forward_block(model, input_ids, graph.get_edges(), graph)
    retained = graph.get_edges()
    current_kl = 0.0
    decisions = []
    for child in reversed(graph.nodes):
        if child == "emb":
            continue
        for edge in graph.incoming_edges[child]:
            if edge not in retained:
                continue
            candidate = retained - {edge}
            with torch.no_grad():
                logits = controlled_forward_block(model, input_ids, candidate, graph)
            agreement = compute_projected_agreement(full, logits, candidates)
            candidate_kl = compute_candidate_kl(full, logits, candidates)
            delta = candidate_kl - current_kl
            removed = agreement >= min_agreement and delta < threshold
            decisions.append(
                {
                    "edge": list(edge),
                    "stage": "legacy_block_acdc",
                    "agreement": agreement,
                    "delta": delta,
                    "decision": "removed" if removed else "kept",
                }
            )
            if removed:
                retained = candidate
                current_kl = candidate_kl

    retained = cleanup_graph(retained, graph)
    changed = True
    while changed:
        changed = False
        for edge in sorted(retained):
            candidate = cleanup_graph(retained - {edge}, graph)
            with torch.no_grad():
                logits = controlled_forward_block(model, input_ids, candidate, graph)
            agreement = compute_projected_agreement(full, logits, candidates)
            removed = agreement >= min_agreement
            decisions.append(
                {
                    "edge": list(edge),
                    "stage": "projected_trim",
                    "agreement": agreement,
                    "decision": "removed" if removed else "kept",
                }
            )
            if removed:
                retained = candidate
                changed = True
                break
    return retained, decisions, full


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = SmallVerifiableConfig.load(
        os.path.join(os.path.dirname(args.checkpoint), "config.json")
    )
    model = load_model(args.checkpoint, config, device).eval()
    block_graph = CircuitGraph(config.n_layers, per_head=False)
    head_graph = CircuitGraph(config.n_layers, config.n_heads, per_head=True)
    reports = {}

    for task in TASKS:
        examples = get_eval_dataset(task)
        input_ids = torch.tensor(
            [example["input_ids"] for example in examples],
            dtype=torch.long,
            device=device,
        )
        candidates = sorted(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task]))
        block_edges, edge_log, full_block = extract_block_circuit(
            model,
            input_ids,
            candidates,
            block_graph,
            args.threshold,
            args.min_agreement,
        )
        head_union = expand_block_edges(block_edges, config.n_heads)
        with torch.no_grad():
            block_logits = controlled_forward_block(
                model, input_ids, block_edges, block_graph
            )
            head_logits = controlled_forward(
                model, input_ids, head_union, head_graph
            )
            full_heads = controlled_forward(
                model, input_ids, head_graph.get_edges(), head_graph
            )

        max_circuit_diff = float((block_logits - head_logits).abs().max().item())
        max_full_diff = float((full_block - full_heads).abs().max().item())
        block_agreement = compute_projected_agreement(
            full_block, block_logits, candidates
        )
        union_agreement = compute_projected_agreement(
            block_logits, head_logits, candidates
        )
        passed = (
            block_agreement >= args.min_agreement
            and union_agreement == 1.0
            and max_circuit_diff <= args.atol
            and max_full_diff <= args.atol
        )
        reports[task] = {
            "domain_size": len(examples),
            "legacy_block_edges": [list(edge) for edge in sorted(block_edges)],
            "legacy_block_edge_count": len(block_edges),
            "expanded_head_union_edges": [list(edge) for edge in sorted(head_union)],
            "expanded_head_union_edge_count": len(head_union),
            "block_circuit_projected_agreement_with_full": block_agreement,
            "head_union_projected_agreement_with_block_circuit": union_agreement,
            "full_graph_max_abs_diff": max_full_diff,
            "circuit_max_abs_diff": max_circuit_diff,
            "atol": args.atol,
            "passed": passed,
            "edge_log": edge_log,
        }

    output = {
        "requirement": (
            "Re-run block-level extractions; the union of retained heads must "
            "reproduce each block-level circuit."
        ),
        "checkpoint": args.checkpoint,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "threshold": args.threshold,
        "min_agreement": args.min_agreement,
        "tasks": reports,
        "regression_pass": all(report["passed"] for report in reports.values()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))
    if not output["regression_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
