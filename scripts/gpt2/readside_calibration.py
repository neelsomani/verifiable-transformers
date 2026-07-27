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
IDENTITY_INTERVENTIONS = (
    "native_full_forward",
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


def compare_to_audit(rows, audit_rows, epsilon):
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
    margin_max_abs = max(
        abs(row["candidate_margin"] - audit["candidate_margin"])
        for row, audit in zip(rows, audit_rows)
    )
    return {
        "rows": len(rows),
        "correct": sum(row["decision"] == row["target"] for row in rows),
        "decisions_exact": decisions_equal,
        "mismatch_ids_exact": mismatches == audit_mismatches,
        "mismatch_ids_sha256": canonical_hash(mismatches),
        "candidate_margin_max_abs_difference": margin_max_abs,
        "candidate_logits_sha256": canonical_hash(
            [row["candidate_logits"] for row in rows]
        ),
        "candidate_margins_sha256": canonical_hash(
            [row["candidate_margin"] for row in rows]
        ),
        "pass": (
            decisions_equal
            and mismatches == audit_mismatches
            and margin_max_abs <= epsilon
        ),
        "audit_candidate_logits_available": all(
            "candidate_logits" in row for row in audit_rows
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

    rows_by_name = {}
    comparisons = {}
    for name in IDENTITY_INTERVENTIONS:
        edges = None if name == "native_full_forward" else edge_sets[name]
        rows = evaluate_rows(
            model,
            values,
            policy["batch_size_for_identity"],
            edges=edges,
            graph=graph,
        )
        rows_by_name[name] = rows
        comparisons[name] = compare_to_audit(
            rows,
            audit["states"]["B_programs_installed_step_zero"][name]["per_input"],
            policy["identity_epsilon_max_abs"],
        )

    report = {
        "schema_version": 1,
        "protocol_id": registration["protocol_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=root
        ).strip(),
        "configuration": policy,
        "note": (
            "The source audit stores decisions and candidate margins but not the "
            "two raw candidate logits. Preflight therefore compares every stored "
            "decision, mismatch ID, and margin at epsilon, and records fresh raw-"
            "logit hashes for all subsequent same-path identity comparisons."
        ),
        "comparisons": comparisons,
        "parameter_hashes": tensor_parameter_hashes(model),
        "pass": all(value["pass"] for value in comparisons.values()),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise SystemExit("TERMINAL: deterministic FP32 audit preflight failed")
    if args.preflight_only:
        return
    raise NotImplementedError("Calibration execution follows preflight implementation")


if __name__ == "__main__":
    main()
