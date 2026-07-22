#!/usr/bin/env python3
"""Materialize the smallest exact-agreement circuit from a threshold sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import torch
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.compare_sweeps import load_sweep_results, recommend_threshold
from scripts.gpt2.behavior_domains import reference_program_targets
from scripts.gpt2.extract import (
    build_circuit_graph,
    controlled_forward,
    get_candidate_token_ids,
    load_behavior_examples,
    load_model_with_variants,
    select_last_real_logits,
)
from scripts.gpt2.synthesize_programs import projected_decisions


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
    parser.add_argument(
        "--model_path",
        default=None,
        help="Model used for an optional independent selection-domain check.",
    )
    parser.add_argument(
        "--domain_manifest",
        default=None,
        help=(
            "Optional development manifest. Candidate circuits must be exact "
            "against P(x) on this domain before minimum-edge selection."
        ),
    )
    parser.add_argument(
        "--candidate_extraction_manifest",
        default=None,
        help=(
            "Require every candidate circuit to have been extracted on this "
            "manifest. This prevents accidental exposure to prior gate rows."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
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


def circuit_edges(path: str) -> set[tuple[str, str]]:
    with open(os.path.join(path, "circuit.json")) as handle:
        circuit = json.load(handle)
    return {
        (edge["source"], edge["target"])
        if isinstance(edge, dict)
        else tuple(edge)
        for edge in circuit["edges"]
    }


def validate_candidate_exposure(results, manifest_path):
    if manifest_path is None:
        return None
    with open(manifest_path, "rb") as handle:
        expected_sha = hashlib.sha256(handle.read()).hexdigest()
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    for result in results:
        with open(os.path.join(result["path"], "circuit.json")) as handle:
            circuit = json.load(handle)
        domain = circuit.get("domain", {})
        if (
            domain.get("manifest_sha256") != expected_sha
            or domain.get("protocol_id") != manifest.get("protocol_id")
            or domain.get("split") != manifest.get("split")
        ):
            raise RuntimeError(
                f"Candidate {result['path']} was not extracted exclusively on "
                f"the locked {manifest.get('protocol_id')}/{manifest.get('split')} "
                "domain"
            )
    return {
        "path": os.path.abspath(manifest_path),
        "manifest_sha256": expected_sha,
        "protocol_id": manifest.get("protocol_id"),
        "split": manifest.get("split"),
    }


def validate_on_domain(args, results):
    if (args.model_path is None) != (args.domain_manifest is None):
        raise ValueError(
            "--model_path and --domain_manifest must be provided together"
        )
    if args.domain_manifest is None:
        return results, None
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_with_variants(args.model_path, device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    examples, provenance = load_behavior_examples(
        args.task, 0, args.domain_manifest
    )
    encoded = tokenizer(
        [example.prompt for example in examples],
        return_tensors="pt",
        padding=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    candidates = get_candidate_token_ids(args.task, tokenizer)
    targets = reference_program_targets(examples, tokenizer, candidates)
    full_decisions = projected_decisions(
        model, input_ids, attention_mask, candidates, args.batch_size
    )
    full_accuracy = float((full_decisions == targets).float().mean().item())
    if full_accuracy != 1.0:
        raise RuntimeError(
            f"Base model accuracy against P(x) on the selection domain is "
            f"{full_accuracy:.6f}; do not filter examples"
        )
    graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
    passing = []
    validations = []
    for result in results:
        decisions = []
        edges = circuit_edges(result["path"])
        with torch.no_grad():
            for start in range(0, input_ids.size(0), args.batch_size):
                ids = input_ids[start : start + args.batch_size]
                mask = attention_mask[start : start + args.batch_size]
                logits = controlled_forward(model, ids, mask, edges, graph)
                rows = select_last_real_logits(logits, mask)
                decisions.append(rows[:, candidates].argmax(dim=-1).cpu())
        decisions = torch.cat(decisions)
        accuracy = float((decisions == targets).float().mean().item())
        agreement = float(
            (decisions == full_decisions).float().mean().item()
        )
        mismatch_ids = [
            example.example_id
            for example, decision, target in zip(
                examples, decisions.tolist(), targets.tolist()
            )
            if decision != target
        ]
        validation = {
            "threshold": result["threshold"],
            "num_edges": result["num_edges"],
            "candidate_accuracy_against_P": accuracy,
            "projected_agreement_with_full": agreement,
            "correct": int((decisions == targets).sum().item()),
            "rows": int(targets.numel()),
            "mismatch_example_ids": mismatch_ids,
            "exact": accuracy == 1.0 and agreement == 1.0,
        }
        validations.append(validation)
        if validation["exact"]:
            candidate = dict(result)
            candidate["domain_validation"] = validation
            passing.append(candidate)
    domain_report = {
        "manifest": provenance,
        "full_model_accuracy_against_P": full_accuracy,
        "candidates": validations,
    }
    return passing, domain_report


def main() -> None:
    args = parse_args()
    results = load_sweep_results(args.sweep_dir, args.task)
    if not results:
        raise RuntimeError(f"No {args.task} sweep results in {args.sweep_dir}")
    extraction_domain = validate_candidate_exposure(
        results, args.candidate_extraction_manifest
    )
    results, domain_report = validate_on_domain(args, results)
    output_dir = os.path.join(args.output_root, args.task)
    os.makedirs(output_dir, exist_ok=True)
    if domain_report is not None:
        with open(os.path.join(output_dir, "selection_attempt.json"), "w") as handle:
            json.dump(
                {
                    "task": args.task,
                    "candidate_extraction_domain": extraction_domain,
                    "development_validation": domain_report,
                },
                handle,
                indent=2,
            )
    if not results:
        raise RuntimeError(
            f"No {args.task} threshold circuit is exact on the independent "
            "selection domain"
        )
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
    if domain_report is None:
        best = recommend_threshold(results, args.task)
    else:
        best = min(results, key=lambda row: (row["num_edges"], row["threshold"]))
    if best["projected_agreement"] < 1.0:
        raise RuntimeError("No threshold achieved exact projected agreement")
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
        "candidate_extraction_domain": extraction_domain,
        "domain_validation": (
            None if domain_report is None else best["domain_validation"]
        ),
        "domain_validation_candidates": domain_report,
        "selection_rule": (
            "exact projected agreement on the extraction domain and optional "
            "independent development domain; contain all task program heads "
            "when provided; then minimum edge count and lower-threshold tie-break"
        ),
    }
    with open(os.path.join(output_dir, "selection.json"), "w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
