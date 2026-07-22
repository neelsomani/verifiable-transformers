#!/usr/bin/env python3
"""Validate and materialize an exact circuit from threshold sweeps."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", action="append", required=True)
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
            "against P(x) on this domain before the configured selection rule."
        ),
    )
    parser.add_argument(
        "--candidate_extraction_manifest",
        action="append",
        default=[],
        help=(
            "Require every candidate circuit to have been extracted on this "
            "manifest. This prevents accidental exposure to prior gate rows."
        ),
    )
    parser.add_argument(
        "--selection_strategy",
        choices=("minimum_edges", "worst_case_margin"),
        default="minimum_edges",
    )
    parser.add_argument("--prior_development_manifest", default=None)
    parser.add_argument("--prior_failure_manifest", default=None)
    parser.add_argument(
        "--prior_failure_example_id", action="append", default=[]
    )
    parser.add_argument(
        "--attempt_name", default="selection_attempt.json"
    )
    parser.add_argument("--prior_selection", default=None)
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


def _manifest_provenance(manifest_path):
    with open(manifest_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    return {
        "path": os.path.abspath(manifest_path),
        "manifest_sha256": digest,
        "protocol_id": manifest.get("protocol_id"),
        "split": manifest.get("split"),
    }


def validate_candidate_exposure(results, manifest_paths):
    if not manifest_paths:
        return []
    if isinstance(manifest_paths, (str, os.PathLike)):
        manifest_paths = [manifest_paths]
    allowed = [_manifest_provenance(path) for path in manifest_paths]
    for result in results:
        with open(os.path.join(result["path"], "circuit.json")) as handle:
            circuit = json.load(handle)
        domain = circuit.get("domain", {})
        matches = [
            candidate
            for candidate in allowed
            if domain.get("manifest_sha256") == candidate["manifest_sha256"]
            and domain.get("protocol_id") == candidate["protocol_id"]
            and domain.get("split") == candidate["split"]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Candidate {result['path']} was not extracted exclusively on "
                "one of the locked candidate-development domains"
            )
        result["candidate_extraction_domain"] = matches[0]
    return allowed


def projected_scores(candidate_logits, targets):
    if candidate_logits.ndim != 2 or candidate_logits.size(1) != 2:
        raise ValueError("Robustness-first selection requires two candidates")
    targets = targets.to(candidate_logits.device)
    rows = torch.arange(candidate_logits.size(0), device=candidate_logits.device)
    decisions = candidate_logits.argmax(dim=-1)
    correct = candidate_logits[rows, targets]
    incorrect = candidate_logits[rows, 1 - targets]
    return decisions.cpu(), (correct - incorrect).float().cpu()


def full_projected_outputs(
    model, input_ids, attention_mask, candidates, targets, batch_size
):
    candidate_rows = []
    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start : start + batch_size]
            mask = attention_mask[start : start + batch_size]
            logits = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            rows = select_last_real_logits(logits, mask)
            candidate_rows.append(rows[:, candidates].float().cpu())
    return projected_scores(torch.cat(candidate_rows), targets)


def _prior_diagnostic_context(args, examples):
    prior_prompts = None
    if args.prior_development_manifest is not None:
        prior_examples, prior_provenance = load_behavior_examples(
            args.task, 0, args.prior_development_manifest
        )
        prior_prompts = {example.prompt for example in prior_examples}
    else:
        prior_provenance = None
    failure_by_prompt = {}
    if args.prior_failure_example_id:
        if args.prior_failure_manifest is None:
            raise ValueError(
                "--prior_failure_manifest is required with failure example IDs"
            )
        failure_examples, failure_provenance = load_behavior_examples(
            args.task, 0, args.prior_failure_manifest
        )
        failures_by_id = {
            example.example_id: example for example in failure_examples
        }
        for example_id in args.prior_failure_example_id:
            if example_id not in failures_by_id:
                raise ValueError(
                    f"Prior failure {example_id!r} is absent from its manifest"
                )
            failure = failures_by_id[example_id]
            failure_by_prompt[failure.prompt] = failure
    else:
        failure_provenance = None
    development_prompts = {example.prompt for example in examples}
    if prior_prompts is not None and not prior_prompts <= development_prompts:
        raise ValueError("Prior development rows are not all present in development")
    if not set(failure_by_prompt) <= development_prompts:
        raise ValueError("Prior failure rows are not all present in development")
    return {
        "prior_prompts": prior_prompts,
        "prior_provenance": prior_provenance,
        "failure_by_prompt": failure_by_prompt,
        "failure_provenance": failure_provenance,
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
    full_decisions, full_margins = full_projected_outputs(
        model,
        input_ids,
        attention_mask,
        candidates,
        targets,
        args.batch_size,
    )
    full_accuracy = float((full_decisions == targets).float().mean().item())
    if full_accuracy != 1.0:
        raise RuntimeError(
            f"Base model accuracy against P(x) on the selection domain is "
            f"{full_accuracy:.6f}; do not filter examples"
        )
    graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
    diagnostics = _prior_diagnostic_context(args, examples)
    prior_indices = None
    if diagnostics["prior_prompts"] is not None:
        prior_indices = torch.tensor(
            [
                index
                for index, example in enumerate(examples)
                if example.prompt in diagnostics["prior_prompts"]
            ],
            dtype=torch.long,
        )
    failure_indices = {
        index: diagnostics["failure_by_prompt"][example.prompt]
        for index, example in enumerate(examples)
        if example.prompt in diagnostics["failure_by_prompt"]
    }
    passing = []
    validations = []
    for result in results:
        candidate_rows = []
        edges = circuit_edges(result["path"])
        with torch.no_grad():
            for start in range(0, input_ids.size(0), args.batch_size):
                ids = input_ids[start : start + args.batch_size]
                mask = attention_mask[start : start + args.batch_size]
                logits = controlled_forward(model, ids, mask, edges, graph)
                rows = select_last_real_logits(logits, mask)
                candidate_rows.append(rows[:, candidates].float().cpu())
        decisions, margins = projected_scores(
            torch.cat(candidate_rows), targets
        )
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
            "path": result["path"],
            "threshold": result["threshold"],
            "num_edges": result["num_edges"],
            "candidate_extraction_domain": result.get(
                "candidate_extraction_domain"
            ),
            "candidate_accuracy_against_P": accuracy,
            "projected_agreement_with_full": agreement,
            "correct": int((decisions == targets).sum().item()),
            "rows": int(targets.numel()),
            "mismatch_example_ids": mismatch_ids,
            "minimum_signed_correct_margin": float(margins.min().item()),
            "mean_signed_correct_margin": float(margins.mean().item()),
            "exact": accuracy == 1.0 and agreement == 1.0,
        }
        if prior_indices is not None:
            prior_decisions = decisions[prior_indices]
            prior_targets = targets[prior_indices]
            prior_margins = margins[prior_indices]
            validation["prior_development"] = {
                "rows": int(prior_indices.numel()),
                "accuracy_against_P": float(
                    (prior_decisions == prior_targets).float().mean().item()
                ),
                "minimum_signed_correct_margin": float(
                    prior_margins.min().item()
                ),
                "exact": bool((prior_decisions == prior_targets).all().item()),
            }
        validation["retrospective_failure_examples"] = [
            {
                "previous_example_id": failure.example_id,
                "current_example_id": examples[index].example_id,
                "prompt": failure.prompt,
                "stratum": failure.stratum,
                "template_id": failure.template_id,
                "correct_token": failure.correct_token,
                "incorrect_token": failure.incorrect_token,
                "opener_token_index": failure.metadata.get(
                    "opener_token_index"
                ),
                "signed_correct_margin": float(margins[index].item()),
                "correct": bool(decisions[index] == targets[index]),
            }
            for index, failure in failure_indices.items()
        ]
        validations.append(validation)
        if validation["exact"]:
            candidate = dict(result)
            candidate["domain_validation"] = validation
            passing.append(candidate)
    domain_report = {
        "manifest": provenance,
        "full_model_accuracy_against_P": full_accuracy,
        "full_model_minimum_signed_correct_margin": float(
            full_margins.min().item()
        ),
        "prior_development_manifest": diagnostics["prior_provenance"],
        "prior_failure_manifest": diagnostics["failure_provenance"],
        "candidates": validations,
    }
    prior_exact = [
        validation
        for validation in validations
        if validation.get("prior_development", {}).get("exact") is True
    ]
    if prior_exact:
        prior_best = min(
            prior_exact,
            key=lambda row: (
                -row["prior_development"]["minimum_signed_correct_margin"],
                row["num_edges"],
                row["threshold"],
            ),
        )
        domain_report["retrospective_robustness_first_selection"] = {
            "path": prior_best["path"],
            "threshold": prior_best["threshold"],
            "num_edges": prior_best["num_edges"],
            "prior_development_minimum_signed_correct_margin": prior_best[
                "prior_development"
            ]["minimum_signed_correct_margin"],
            "failure_examples": prior_best[
                "retrospective_failure_examples"
            ],
            "would_avoid_recorded_failures": (
                all(
                    row["correct"]
                    for row in prior_best["retrospective_failure_examples"]
                )
                if prior_best["retrospective_failure_examples"]
                else None
            ),
        }
    if args.prior_selection is not None:
        with open(args.prior_selection) as handle:
            prior_selection = json.load(handle)
        prior_source = os.path.abspath(prior_selection["source"])
        matching = [
            validation
            for validation in validations
            if os.path.abspath(validation["path"]) == prior_source
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Prior selected candidate {prior_source} is absent from sweeps"
            )
        prior_validation = matching[0]
        domain_report["prior_selected_circuit_diagnostic"] = {
            "selection_path": os.path.abspath(args.prior_selection),
            "source": prior_source,
            "threshold": prior_validation["threshold"],
            "num_edges": prior_validation["num_edges"],
            "prior_development": prior_validation.get("prior_development"),
            "recorded_failure_examples": prior_validation[
                "retrospective_failure_examples"
            ],
        }
    return passing, domain_report


def select_best_candidate(results, task, strategy, *, independently_validated):
    if not independently_validated:
        return recommend_threshold(results, task)
    if strategy == "worst_case_margin":
        return min(
            results,
            key=lambda row: (
                -row["domain_validation"][
                    "minimum_signed_correct_margin"
                ],
                row["num_edges"],
                row["threshold"],
            ),
        )
    return min(results, key=lambda row: (row["num_edges"], row["threshold"]))


def main() -> None:
    args = parse_args()
    results = []
    seen_paths = set()
    for sweep_dir in args.sweep_dir:
        for result in load_sweep_results(sweep_dir, args.task):
            path = os.path.abspath(result["path"])
            if path in seen_paths:
                continue
            seen_paths.add(path)
            result["path"] = path
            results.append(result)
    if not results:
        raise RuntimeError(
            f"No {args.task} sweep results in {args.sweep_dir}"
        )
    extraction_domains = validate_candidate_exposure(
        results, args.candidate_extraction_manifest
    )
    results, domain_report = validate_on_domain(args, results)
    output_dir = os.path.join(args.output_root, args.task)
    os.makedirs(output_dir, exist_ok=True)
    if domain_report is not None:
        with open(os.path.join(output_dir, args.attempt_name), "w") as handle:
            json.dump(
                {
                    "task": args.task,
                    "candidate_extraction_domains": extraction_domains,
                    "selection_strategy": args.selection_strategy,
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
    best = select_best_candidate(
        results,
        args.task,
        args.selection_strategy,
        independently_validated=domain_report is not None,
    )
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
        "candidate_extraction_domain": best.get(
            "candidate_extraction_domain"
        ),
        "candidate_extraction_domains": extraction_domains,
        "selection_strategy": args.selection_strategy,
        "domain_validation": (
            None if domain_report is None else best["domain_validation"]
        ),
        "domain_validation_candidates": domain_report,
        "selection_rule": (
            "exact projected agreement on the extraction and independent "
            "development domains; contain all task program heads when provided; "
            + (
                "maximize worst-case signed correct-token margin, then minimum "
                "edge count and lower threshold"
                if args.selection_strategy == "worst_case_margin"
                else "minimum edge count and lower-threshold tie-break"
            )
        ),
    }
    with open(os.path.join(output_dir, "selection.json"), "w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
