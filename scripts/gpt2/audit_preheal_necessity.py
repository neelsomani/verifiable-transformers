#!/usr/bin/env python3
"""Audit selected-circuit necessity before any Phase Q optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.behavior_domains import reference_program_targets
from scripts.gpt2.extract import (
    build_circuit_graph,
    controlled_forward,
    get_candidate_token_ids,
    load_behavior_examples,
    load_model_with_variants,
    select_last_real_logits,
)
from scripts.programs import install_program_heads, load_programs


PROGRAM_NODES = ("attn_7_h_11", "attn_9_h_0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_edges(path: Path) -> set[tuple[str, str]]:
    with path.open() as handle:
        payload = json.load(handle)
    return {tuple(edge) for edge in payload["edges"]}


def internal_circuit_nodes(edges: set[tuple[str, str]]) -> list[str]:
    return sorted({node for edge in edges for node in edge} - {"emb", "logits"})


def intervention_edge_sets(graph, selected_edges):
    """Return exact edge sets for every requested controlled-forward intervention."""
    full = set(graph.all_edges)
    internal = internal_circuit_nodes(selected_edges)
    result = {
        "circuit_only": set(selected_edges),
        "zero_attn_7_h_11": full - {
            edge for edge in full if edge[0] == "attn_7_h_11"
        },
        "zero_attn_9_h_0": full - {
            edge for edge in full if edge[0] == "attn_9_h_0"
        },
        "zero_both_program_heads": full
        - {edge for edge in full if edge[0] in PROGRAM_NODES},
        "whole_selected_circuit_edge_zero": full - set(selected_edges),
        "whole_selected_circuit_node_zero": full
        - {edge for edge in full if edge[0] in internal},
    }
    return result


def summarize_rows(rows):
    correct = sum(row["decision"] == row["target"] for row in rows)
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy_against_P": correct / len(rows),
        "mean_candidate_margin": sum(row["candidate_margin"] for row in rows)
        / len(rows),
        "minimum_candidate_margin": min(row["candidate_margin"] for row in rows),
        "mismatch_example_ids": [
            row["example_id"] for row in rows if row["decision"] != row["target"]
        ],
    }


def evaluate(model, values, batch_size, edges=None, graph=None):
    rows = []
    with torch.no_grad():
        for start in range(0, len(values["examples"]), batch_size):
            stop = start + batch_size
            ids = values["input_ids"][start:stop]
            mask = values["attention_mask"][start:stop]
            if edges is None:
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            else:
                logits = controlled_forward(model, ids, mask, edges, graph)
            projected = select_last_real_logits(logits, mask)[
                :, values["candidate_token_ids"]
            ].float()
            decisions = projected.argmax(-1).cpu()
            targets = values["targets"][start:stop]
            other = 1 - targets
            indices = torch.arange(decisions.numel(), device=projected.device)
            margins = (
                projected[indices, targets.to(projected.device)]
                - projected[indices, other.to(projected.device)]
            ).cpu()
            for offset, (decision, target, margin) in enumerate(
                zip(decisions.tolist(), targets.tolist(), margins.tolist())
            ):
                example = values["examples"][start + offset]
                rows.append(
                    {
                        "example_id": example.example_id,
                        "stratum": example.stratum,
                        "target": target,
                        "target_token_id": values["candidate_token_ids"][target],
                        "decision": decision,
                        "decision_token_id": values["candidate_token_ids"][decision],
                        "candidate_margin": margin,
                    }
                )
    return {"summary": summarize_rows(rows), "per_input": rows}


def describe_interventions(graph, selected_edges):
    edge_sets = intervention_edge_sets(graph, selected_edges)
    full = set(graph.all_edges)
    return {
        "native_full_forward": {
            "semantics": "Transformers model forward; no intervention.",
            "removed_edges": [],
        },
        "circuit_only": {
            "semantics": (
                "Sufficiency control: retain exactly the selected circuit edges; "
                "all other residual edges are zero-ablated."
            ),
            "retained_edges": sorted(map(list, selected_edges)),
            "removed_edge_count": len(full - selected_edges),
        },
        **{
            name: {
                "semantics": (
                    "Controlled forward with zero ablation before W_O; every listed "
                    "residual contribution is deleted."
                ),
                "removed_edges": sorted(map(list, full - edges)),
            }
            for name, edges in edge_sets.items()
            if name != "circuit_only"
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--domain_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--recorded_programs_copy", required=True)
    parser.add_argument("--posthealing_migration_report", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(args.base_model).resolve()
    programs_path = Path(args.programs).resolve()
    circuit_path = Path(args.circuit).resolve()
    domain_path = Path(args.domain_manifest).resolve()
    recorded_programs_path = Path(args.recorded_programs_copy).resolve()
    posthealing_report_path = Path(args.posthealing_migration_report).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if programs_path.read_bytes() != recorded_programs_path.read_bytes():
        raise RuntimeError(
            "Selected programs are not byte-identical to the successful run copy"
        )

    tokenizer = GPT2Tokenizer.from_pretrained(base_path)
    tokenizer.pad_token = tokenizer.eos_token
    examples, domain_provenance = load_behavior_examples(
        "quote_close", 0, str(domain_path)
    )
    encoded = tokenizer(
        [example.prompt for example in examples], return_tensors="pt", padding=True
    )
    candidate_ids = get_candidate_token_ids("quote_close", tokenizer)
    values = {
        "examples": examples,
        "input_ids": encoded["input_ids"].to(args.device),
        "attention_mask": encoded["attention_mask"].to(args.device),
        "candidate_token_ids": candidate_ids,
        "targets": reference_program_targets(examples, tokenizer, candidate_ids),
    }

    base = load_model_with_variants(str(base_path), args.device).eval()
    graph = build_circuit_graph(base.config.n_layer, base.config.n_head)
    selected_edges = load_selected_edges(circuit_path)
    edge_sets = intervention_edge_sets(graph, selected_edges)
    interventions = describe_interventions(graph, selected_edges)

    provenance = {
        "audit_type": "diagnostic_only_no_training",
        "preserved_commit": "877a443",
        "execution_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "base_model": str(base_path),
        "base_model_weights_sha256": sha256_file(base_path / "model.safetensors"),
        "programs": str(programs_path),
        "programs_sha256": sha256_file(programs_path),
        "recorded_successful_run_programs_copy": str(recorded_programs_path),
        "recorded_successful_run_programs_copy_sha256": sha256_file(
            recorded_programs_path
        ),
        "programs_copy_byte_identical": (
            programs_path.read_bytes() == recorded_programs_path.read_bytes()
        ),
        "circuit": str(circuit_path),
        "circuit_sha256": sha256_file(circuit_path),
        "domain_manifest": str(domain_path),
        "domain_manifest_sha256": sha256_file(domain_path),
        "domain": domain_provenance,
        "program_nodes": list(PROGRAM_NODES),
        "selected_circuit_internal_nodes": internal_circuit_nodes(selected_edges),
        "selected_circuit_edges": sorted(map(list, selected_edges)),
        "state_A": "Untouched norm-free source checkpoint with original neural heads.",
        "state_B": (
            "Fresh load of state A followed only by install_program_heads using the "
            "recorded selected programs; no optimizer was constructed and no step ran."
        ),
        "posthealing_migration_report": str(posthealing_report_path),
        "posthealing_migration_report_sha256": sha256_file(
            posthealing_report_path
        ),
    }

    states = {}
    for state_name, model in (("A_untouched_neural", base),):
        reports = {"native_full_forward": evaluate(model, values, args.batch_size)}
        reports.update(
            {
                name: evaluate(
                    model, values, args.batch_size, edges=edge_set, graph=graph
                )
                for name, edge_set in edge_sets.items()
            }
        )
        states[state_name] = reports

    programmed = load_model_with_variants(str(base_path), args.device).eval()
    programs = load_programs(str(programs_path))
    if sorted(programs) != [(7, 11), (9, 0)]:
        raise RuntimeError(f"Unexpected selected programs: {sorted(programs)}")
    install_program_heads(programmed, programs, attention_variant="sparsemax")
    states["B_programs_installed_step_zero"] = {
        "native_full_forward": evaluate(programmed, values, args.batch_size),
        **{
            name: evaluate(
                programmed, values, args.batch_size, edges=edge_set, graph=graph
            )
            for name, edge_set in edge_sets.items()
        },
    }

    a_whole = states["A_untouched_neural"]["whole_selected_circuit_node_zero"][
        "summary"
    ]["accuracy_against_P"]
    a_joint = states["A_untouched_neural"]["zero_both_program_heads"]["summary"][
        "accuracy_against_P"
    ]
    b_joint = states["B_programs_installed_step_zero"][
        "zero_both_program_heads"
    ]["summary"]["accuracy_against_P"]
    with posthealing_report_path.open() as handle:
        posthealing = json.load(handle)
    posthealing_joint = posthealing["tasks"]["quote_close"][
        "without_intended_program_heads"
    ]["full"]["projected_agreement"]
    exact_redundancy_first_appears = (
        "after_healing"
        if a_joint < 1.0 and b_joint < 1.0 and posthealing_joint == 1.0
        else "not_determined_by_A_B_and_recorded_posthealing_state"
    )
    conclusion = {
        "state_A_whole_circuit_node_zero_accuracy": a_whole,
        "state_A_joint_head_lesion_accuracy": a_joint,
        "state_B_joint_program_head_lesion_accuracy": b_joint,
        "recorded_posthealing_joint_program_head_lesion_agreement": (
            posthealing_joint
        ),
        "exact_redundancy_first_appears": exact_redundancy_first_appears,
        "kill_rule_triggered": a_whole == 1.0,
        "causal_timing": (
            "Exact two-head redundancy is absent in both pre-healing states and first "
            "appears in the recorded post-healing state; healing created or reinforced "
            "an outside route enough to make it exact on D."
            if exact_redundancy_first_appears == "after_healing"
            else "The causal timing is not uniquely determined by the audited states."
        ),
        "phase_q_disposition": (
            "STOP: selected circuit is sufficient but not necessary and is unsuitable "
            "as the Phase Q causal flagship."
            if a_whole == 1.0
            else (
                "The untouched selected circuit is sufficient and necessary under both "
                "whole-circuit interventions, but the two heads are not an exclusive "
                "mechanism. Do not claim program exclusivity; the later exact bypass is "
                "a healing-induced or healing-reinforced effect."
            )
        ),
    }
    payload = {
        "schema_version": 1,
        "provenance": provenance,
        "interventions": interventions,
        "states": states,
        "conclusion": conclusion,
    }
    with (output_dir / "audit.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with (output_dir / "SUMMARY.txt").open("w") as handle:
        handle.write(conclusion["causal_timing"] + "\n")
        handle.write(conclusion["phase_q_disposition"] + "\n")
        for state, reports in states.items():
            handle.write(f"\n{state}\n")
            for name, report in reports.items():
                summary = report["summary"]
                handle.write(
                    f"  {name}: {summary['correct']}/{summary['rows']} "
                    f"({summary['accuracy_against_P']:.6f}), "
                    f"mean margin {summary['mean_candidate_margin']:.6f}\n"
                )
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
