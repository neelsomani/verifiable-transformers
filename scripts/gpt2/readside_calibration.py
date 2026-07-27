#!/usr/bin/env python3
"""Run the preregistered Phase Q read-side calibration ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.audit_preheal_necessity import (
    intervention_edge_sets,
    load_selected_edges,
)
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
LOCKED_IDENTITY_INTERVENTIONS = (
    "native_full_forward",
    "zero_both_program_heads",
    "whole_selected_circuit_node_zero",
    "whole_selected_circuit_edge_zero",
)
BASELINE_BATTERY = (
    "native_full_forward",
    "circuit_only",
    "zero_attn_7_h_11",
    "zero_attn_9_h_0",
    "zero_both_program_heads",
    "whole_selected_circuit_node_zero",
    "whole_selected_circuit_edge_zero",
)


def configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_hash(values) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_parameter_hashes(model) -> dict[str, str]:
    result = {}
    for name, value in model.state_dict().items():
        raw = value.detach().cpu().contiguous().numpy().tobytes()
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def evaluate_rows(model, values, batch_size, edges=None, graph=None):
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
            indices = torch.arange(decisions.numel(), device=projected.device)
            margins = (
                projected[indices, targets.to(projected.device)]
                - projected[indices, (1 - targets).to(projected.device)]
            ).cpu()
            for offset in range(decisions.numel()):
                example = values["examples"][start + offset]
                rows.append(
                    {
                        "example_id": example.example_id,
                        "target": int(targets[offset]),
                        "decision": int(decisions[offset]),
                        "candidate_logits": [
                            float(v) for v in projected[offset].detach().cpu()
                        ],
                        "candidate_margin": float(margins[offset]),
                    }
                )
    return rows


def compare_semantics_to_audit(rows, audit_rows):
    if [r["example_id"] for r in rows] != [r["example_id"] for r in audit_rows]:
        raise RuntimeError("Domain order does not reproduce the audit")
    decisions_equal = all(
        row["decision"] == audit["decision"] for row, audit in zip(rows, audit_rows)
    )
    mismatches = [
        row["example_id"] for row in rows if row["decision"] != row["target"]
    ]
    audit_mismatches = [
        row["example_id"]
        for row in audit_rows
        if row["decision"] != row["target"]
    ]
    return {
        "rows": len(rows),
        "correct": sum(row["decision"] == row["target"] for row in rows),
        "decisions_exact": decisions_equal,
        "mismatch_ids_exact": mismatches == audit_mismatches,
        "mismatch_ids_sha256": canonical_hash(mismatches),
        "candidate_logits_sha256": canonical_hash(
            [row["candidate_logits"] for row in rows]
        ),
        "candidate_margins_sha256": canonical_hash(
            [row["candidate_margin"] for row in rows]
        ),
        "pass": decisions_equal and mismatches == audit_mismatches,
        "numerical_comparison_to_old_audit": "not_performed_by_amendment",
    }


def compare_paired_rows(first, second, epsilon):
    if [r["example_id"] for r in first] != [r["example_id"] for r in second]:
        raise RuntimeError("Paired deterministic runs used different domain order")
    decisions_exact = all(
        left["decision"] == right["decision"] for left, right in zip(first, second)
    )
    first_mismatches = [
        row["example_id"] for row in first if row["decision"] != row["target"]
    ]
    second_mismatches = [
        row["example_id"] for row in second if row["decision"] != row["target"]
    ]
    logit_max_abs = max(
        abs(left_logit - right_logit)
        for left, right in zip(first, second)
        for left_logit, right_logit in zip(
            left["candidate_logits"], right["candidate_logits"]
        )
    )
    margin_max_abs = max(
        abs(left["candidate_margin"] - right["candidate_margin"])
        for left, right in zip(first, second)
    )
    return {
        "rows": len(first),
        "correct": sum(row["decision"] == row["target"] for row in first),
        "decisions_exact": decisions_exact,
        "mismatch_ids_exact": first_mismatches == second_mismatches,
        "mismatch_ids_sha256": canonical_hash(first_mismatches),
        "candidate_logit_max_abs_difference": logit_max_abs,
        "candidate_margin_max_abs_difference": margin_max_abs,
        "epsilon": epsilon,
        "pass": (
            decisions_exact
            and first_mismatches == second_mismatches
            and logit_max_abs <= epsilon
            and margin_max_abs <= epsilon
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    registration_path = Path(args.registration)
    registration = json.loads(registration_path.read_text())
    root = Path(__file__).resolve().parents[2]
    frozen = registration["frozen_inputs"]
    policy = registration["execution_policy"]
    configure_determinism(policy["seed"])

    audit = json.loads((root / frozen["audit"]).read_text())
    base_path = root / frozen["source_model"]
    tokenizer = GPT2Tokenizer.from_pretrained(base_path)
    tokenizer.pad_token = tokenizer.eos_token
    examples, _ = load_behavior_examples(
        "quote_close", 0, str(root / frozen["domain_manifest"])
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

    model = load_model_with_variants(str(base_path), args.device).float().eval()
    programs = load_programs(str(root / frozen["programs"]))
    install_program_heads(model, programs, attention_variant="sparsemax")
    graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
    selected_edges = load_selected_edges(root / frozen["circuit"])
    edge_sets = intervention_edge_sets(graph, selected_edges)

    paired_runs = []
    for _ in range(2):
        run = {}
        for name in BASELINE_BATTERY:
            edges = None if name == "native_full_forward" else edge_sets[name]
            run[name] = evaluate_rows(
                model,
                values,
                policy["batch_size_for_identity"],
                edges=edges,
                graph=graph,
            )
        paired_runs.append(run)

    paired_comparisons = {
        name: compare_paired_rows(
            paired_runs[0][name],
            paired_runs[1][name],
            policy["identity_epsilon_max_abs"],
        )
        for name in BASELINE_BATTERY
    }
    audit_semantic_comparisons = {
        name: compare_semantics_to_audit(
            paired_runs[0][name],
            audit["states"]["B_programs_installed_step_zero"][name]["per_input"],
        )
        for name in BASELINE_BATTERY
    }

    report = {
        "schema_version": 1,
        "protocol_id": registration["protocol_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=root
        ).strip(),
        "configuration": policy,
        "amendment": "PROTOCOL_AMENDMENT_2026-07-27.md",
        "note": (
            "Two current pinned runs are the numerical identity pair. The older "
            "audit is compared only for exact decisions and mismatch-ID sets."
        ),
        "paired_deterministic_comparisons": paired_comparisons,
        "old_audit_semantic_comparisons": audit_semantic_comparisons,
        "numerical_identity_reference": {
            name: paired_runs[0][name]
            for name in LOCKED_IDENTITY_INTERVENTIONS
        },
        "parameter_hashes": tensor_parameter_hashes(model),
        "pass": (
            all(value["pass"] for value in paired_comparisons.values())
            and all(value["pass"] for value in audit_semantic_comparisons.values())
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_baseline.json"
    if paired_path.exists():
        raise RuntimeError(f"Refusing to overwrite paired baseline: {paired_path}")
    paired_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise SystemExit("STOP: paired deterministic FP32 baseline failed")
    if args.preflight_only:
        return
    raise NotImplementedError("Calibration execution follows preflight implementation")


if __name__ == "__main__":
    main()
