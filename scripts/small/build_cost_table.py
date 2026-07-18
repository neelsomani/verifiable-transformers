#!/usr/bin/env python3
"""Build the unified small-model verification and bilinear-cost table."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.small.extract import load_model
from scripts.small.extract_weights import load_small_config


HEAD_PATTERN = re.compile(r"^attn_(\d+)_h_(\d+)$")
FORMAL_PROPERTIES = {
    "projected_functional_equivalence",
    "projected_content_invariance",
    "projected_edge_necessity",
    "projected_continuous_robustness",
}


RUNS = {
    "bandnorm": {
        "checkpoint": "artifacts/small_band_norm_matched/checkpoint-final",
        "circuit_root": "artifacts/small_band_norm_matched_circuits",
        "weights": "artifacts/small_band_norm_matched/smt_weights.json",
    },
    "norm_free": {
        "checkpoint": "artifacts/small_norm_free/checkpoint-final",
        "circuit_root": "artifacts/small_norm_free_circuits",
        "weights": "artifacts/small_norm_free/smt_weights.json",
    },
    "program_healed": {
        "checkpoint": (
            "artifacts/small_program_healed_chase_round1_core_aware/checkpoint-final"
        ),
        "circuit_root": (
            "artifacts/small_program_healed_chase_round1_core_aware_circuits"
        ),
        "weights": (
            "artifacts/small_program_healed_chase_round1_core_aware/smt_weights.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--sequence_length", type=int, default=6)
    return parser.parse_args()


def load(path: str):
    with open(path) as handle:
        return json.load(handle)


def active_heads(circuit: dict) -> set[str]:
    result = set()
    for edge in circuit["edges"]:
        source = edge["source"] if isinstance(edge, dict) else edge[0]
        target = edge["target"] if isinstance(edge, dict) else edge[1]
        for node in (source, target):
            if HEAD_PATTERN.match(node):
                result.add(node)
    return result


def formal_properties(verification: dict) -> list[dict]:
    return [
        value
        for value in verification["properties"]
        if value.get("property") in FORMAL_PROPERTIES
    ]


def assertion_metrics(properties: list[dict], edge_count: int) -> dict:
    categories = {
        "norm": 0,
        "attention": 0,
        "mlp": 0,
        "embedding/residual": 0,
        "decision": 0,
    }
    norm_instances = 0
    per_property_attribution = {}
    per_property_solve_seconds = {}
    for value in properties:
        name = value["property"]
        profile = value.get("assertion_attribution")
        if profile is None:
            raise RuntimeError(f"Missing assertion attribution for {name}")
        per_property_attribution[name] = profile
        per_property_solve_seconds[name] = value.get("solve_seconds", 0.0)
        norm_instances += profile.get("norm_instances", 0)
        for category in categories:
            categories[category] += profile["assertions"].get(category, 0)
    assertion_count = sum(categories.values())
    solve_seconds = sum(per_property_solve_seconds.values())
    return {
        "assertion_categories": categories,
        "assertion_attribution": categories,
        "per_property_assertion_attribution": per_property_attribution,
        "profiled_assertions": assertion_count,
        "profiled_solve_seconds": solve_seconds,
        "per_property_solve_seconds": per_property_solve_seconds,
        "assertions_per_edge": assertion_count / edge_count,
        "solve_seconds_per_edge": solve_seconds / edge_count,
        "norm_instances": norm_instances,
        "norm_attributable_assertions": categories["norm"],
        "norm_attributable_assertions_per_norm_instance": (
            categories["norm"] / norm_instances if norm_instances else None
        ),
    }


def make_row(
    *,
    row_type: str,
    comparison_group: str | None,
    run_name: str,
    task: str,
    weights: dict,
    program_nodes: set[str],
    circuit: dict,
    verification: dict,
    sequence_length: int,
) -> dict:
    properties = {value["property"]: value for value in verification["properties"]}
    formal = formal_properties(verification)
    heads = active_heads(circuit)
    program_heads = heads & program_nodes
    neural_heads = heads - program_nodes
    causal_pairs = sequence_length * (sequence_length + 1) // 2
    bilinear_terms = len(neural_heads) * weights["head_dim"] * causal_pairs
    robustness = properties.get("projected_continuous_robustness", {})
    metrics = assertion_metrics(formal, circuit["num_edges"])
    qualitative_claim = (
        "Final decision is affine plus candidate argmax; no norm branch "
        "certificate is required and robustness cannot be branch-unknown."
        if weights["norm_variant"] == "none"
        else "BandNorm verification requires branch certificates; whether the "
        "epsilon box remains within one branch is checkpoint-dependent."
    )
    return {
        "row_type": row_type,
        "comparison_group": comparison_group,
        "run": run_name,
        "task": task,
        "norm_variant": weights["norm_variant"],
        "model_n_heads": weights["n_heads"],
        "head_dim": weights["head_dim"],
        "circuit_edges": circuit["num_edges"],
        "active_attention_heads": len(heads),
        "active_program_heads": len(program_heads),
        "active_neural_heads": len(neural_heads),
        "qk_bilinear_terms_length_6": bilinear_terms,
        "zero_attention_bilinear_terms": bilinear_terms == 0,
        **metrics,
        "descriptive_wall_time_seconds_not_comparable": sum(
            value.get("wall_time_seconds", 0.0) for value in formal
        ),
        "branch_certificates_required": robustness.get(
            "branch_certificates_required"
        ),
        "robustness_branch_unknown_possible": weights["norm_variant"] != "none",
        "robustness_provable_without_branch_certificates": (
            weights["norm_variant"] == "none"
            and robustness.get("status") == "VERIFIED"
        ),
        "all_four_properties_verified": len(formal) == 4
        and all(value.get("status") == "VERIFIED" for value in formal),
        "formal_statuses": {
            value["property"]: value.get("status") for value in formal
        },
        "timeouts": sum(value.get("timeout_count", 0) for value in formal),
        "errors": sum(value.get("error_count", 0) for value in formal),
        "unknowns": sum(
            str(value.get("status", "")).startswith("UNKNOWN") for value in formal
        ),
        "qualitative_claim": qualitative_claim,
        "quantitative_claim_scope": (
            "raw_and_normalized_within_matched_topology_group"
            if comparison_group is not None
            else "normalized_metrics_only_across_different_topologies"
        ),
    }


def main() -> None:
    args = parse_args()
    rows = []
    run_metadata = {}
    for run_name, paths in RUNS.items():
        if not all(os.path.exists(path) for path in paths.values()):
            continue
        weights = load(paths["weights"])
        program_nodes = {
            "attn_{}_h_{}".format(*key.split("."))
            for key in weights.get("program_heads", {})
        }
        config = load_small_config(paths["checkpoint"])
        model = load_model(paths["checkpoint"], config, torch.device("cpu"))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        physical_qk_parameters = 0
        for block in model.transformer.h:
            attention = block.attn
            if hasattr(attention, "query_proj"):
                for projection in (attention.query_proj, attention.key_proj):
                    if projection is not None:
                        physical_qk_parameters += sum(
                            parameter.numel() for parameter in projection.parameters()
                        )
            else:
                # GPT-2's c_attn is a single Q/K/V Conv1D with equal thirds.
                physical_qk_parameters += 2 * attention.c_attn.weight.numel() // 3
                physical_qk_parameters += 2 * attention.c_attn.bias.numel() // 3
        run_metadata[run_name] = {
            "norm_variant": weights["norm_variant"],
            "parameter_count": parameter_count,
            "n_heads": weights["n_heads"],
            "head_dim": weights["head_dim"],
            "physical_qk_parameters": physical_qk_parameters,
            "installed_program_heads": sorted(program_nodes),
        }

        for task in ("quote_close", "bracket_type"):
            circuit_path = os.path.join(paths["circuit_root"], task, "circuit.json")
            verification_path = os.path.join(
                paths["circuit_root"], task, "verification", "verification_results.json"
            )
            if not os.path.exists(circuit_path) or not os.path.exists(verification_path):
                continue
            circuit = load(circuit_path)
            verification = load(verification_path)
            rows.append(
                make_row(
                    row_type="selected_circuit",
                    comparison_group=None,
                    run_name=run_name,
                    task=task,
                    weights=weights,
                    program_nodes=program_nodes,
                    circuit=circuit,
                    verification=verification,
                    sequence_length=args.sequence_length,
                )
            )

    matched_selection_path = "artifacts/small_matched_topology/selection.json"
    matched_selection = load(matched_selection_path)
    for task, task_selection in matched_selection["tasks"].items():
        if task_selection["status"] != "SELECTED":
            continue
        comparison_group = f"{task}_{task_selection['matched_edge_count']}_edges"
        for run_name in ("bandnorm", "norm_free"):
            paths = RUNS[run_name]
            weights = load(paths["weights"])
            base = os.path.join("artifacts/small_matched_topology", task, run_name)
            verification_path = os.path.join(
                base, "verification", "verification_results.json"
            )
            if not os.path.exists(verification_path):
                raise RuntimeError(f"Missing matched verification: {verification_path}")
            rows.append(
                make_row(
                    row_type="matched_topology",
                    comparison_group=comparison_group,
                    run_name=run_name,
                    task=task,
                    weights=weights,
                    program_nodes=set(),
                    circuit=load(os.path.join(base, "circuit.json")),
                    verification=load(verification_path),
                    sequence_length=args.sequence_length,
                )
            )

    matched_topology_comparisons = {}
    for row in rows:
        if row["row_type"] == "matched_topology":
            matched_topology_comparisons.setdefault(row["comparison_group"], []).append(
                row
            )

    payload = {
        "sequence_length": args.sequence_length,
        "bilinear_term_definition": (
            "active neural heads * head_dim * causal query-key pairs; program "
            "weights are rational constants and contribute zero Q/K products"
        ),
        "claim_policy": {
            "qualitative": "Reported per run from its own verification result.",
            "quantitative": (
                "Across different circuit topologies, compare only instrumented "
                "normalized columns (assertions per edge, solve seconds per edge, "
                "and norm-attributable assertions per norm instance). Raw totals "
                "are eligible only inside an explicitly matched-topology group."
            ),
            "raw_wall_time": (
                "Descriptive execution metadata only; it is not a cross-topology "
                "quantitative claim."
            ),
        },
        "assertion_category_schema": [
            "norm",
            "attention",
            "mlp",
            "embedding/residual",
            "decision",
        ],
        "matched_topology_selection": matched_selection,
        "matched_topology_comparisons": matched_topology_comparisons,
        "runs": run_metadata,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(payload, handle, indent=2)
    if rows:
        with open(args.output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
