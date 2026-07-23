#!/usr/bin/env python3
"""Select a jointly exact subset of synthesized GPT-2 programs.

Programs are proposed and accepted per head during synthesis.  This stage is
the separate composition check. Held-out mode adds programs using synthesis
data and evaluates the frozen result once on an untouched gate. Bounded mode
uses one declared finite domain and makes no held-out claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
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
from scripts.gpt2.synthesize_programs import projected_decisions
from scripts.programs import install_program_heads, load_programs, save_programs


TASKS = ("quote_close", "bracket_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--synthesis_results", required=True)
    parser.add_argument("--circuit_root", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--synthesis_manifest", default=None)
    parser.add_argument("--gate_manifest", default=None)
    parser.add_argument(
        "--bounded_manifest",
        default=None,
        help=(
            "Declared finite behavior domain. In this mode there is no gate "
            "split and no held-out-generalization claim."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--minimum_program_heads", type=int, default=1)
    parser.add_argument("--minimum_program_heads_per_task", type=int, default=1)
    parser.add_argument(
        "--require_all_circuit_heads",
        action="store_true",
        help="Require every attention head retained by every selected circuit.",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, default=list(TASKS)
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_domains(tokenizer, manifest_path, device, tasks=TASKS):
    result = {}
    provenance = {}
    for task in tasks:
        examples, task_provenance = load_behavior_examples(
            task, 0, manifest_path
        )
        encoded = tokenizer(
            [example.prompt for example in examples],
            return_tensors="pt",
            padding=True,
        )
        candidates = get_candidate_token_ids(task, tokenizer)
        result[task] = {
            "examples": examples,
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "candidates": candidates,
            "targets": reference_program_targets(
                examples, tokenizer, candidates
            ),
        }
        provenance[task] = task_provenance
    return result, provenance


def load_circuit_edges(circuit_root, tasks=TASKS):
    result = {}
    for task in tasks:
        with open(Path(circuit_root) / task / "circuit.json") as handle:
            circuit = json.load(handle)
        result[task] = {
            (edge["source"], edge["target"])
            if isinstance(edge, dict)
            else tuple(edge)
            for edge in circuit["edges"]
        }
    return result


def evaluate(model, domains, batch_size, edges_by_task=None):
    output = {}
    graph = (
        build_circuit_graph(model.config.n_layer, model.config.n_head)
        if edges_by_task is not None
        else None
    )
    for task, values in domains.items():
        if edges_by_task is None:
            decisions = projected_decisions(
                model,
                values["input_ids"],
                values["attention_mask"],
                values["candidates"],
                batch_size,
            )
        else:
            decision_batches = []
            with torch.no_grad():
                for start in range(0, values["input_ids"].size(0), batch_size):
                    input_ids = values["input_ids"][start : start + batch_size]
                    attention_mask = values["attention_mask"][
                        start : start + batch_size
                    ]
                    logits = controlled_forward(
                        model,
                        input_ids,
                        attention_mask,
                        edges_by_task[task],
                        graph,
                    )
                    rows = select_last_real_logits(logits, attention_mask)
                    decision_batches.append(
                        rows[:, values["candidates"]].argmax(dim=-1).cpu()
                    )
            decisions = torch.cat(decision_batches)
        targets = values["targets"]
        by_stratum = {}
        for stratum in sorted(
            {example.stratum for example in values["examples"]}
        ):
            indices = [
                index
                for index, example in enumerate(values["examples"])
                if example.stratum == stratum
            ]
            stratum_decisions = decisions[indices]
            stratum_targets = targets[indices]
            by_stratum[stratum] = {
                "accuracy_against_P": float(
                    (stratum_decisions == stratum_targets).float().mean().item()
                ),
                "correct": int(
                    (stratum_decisions == stratum_targets).sum().item()
                ),
                "rows": len(indices),
                "mismatch_example_ids": [
                    values["examples"][index].example_id
                    for index in indices
                    if decisions[index] != targets[index]
                ],
            }
        output[task] = {
            "accuracy_against_P": float(
                (decisions == targets).float().mean().item()
            ),
            "correct": int((decisions == targets).sum().item()),
            "rows": int(targets.numel()),
            "mismatch_example_ids": [
                example.example_id
                for example, decision, target in zip(
                    values["examples"], decisions.tolist(), targets.tolist()
                )
                if decision != target
            ],
            "by_stratum": by_stratum,
        }
    tasks = tuple(domains)
    output["exact_all_tasks"] = all(
        output[task]["accuracy_against_P"] == 1.0 for task in tasks
    )
    # Compatibility alias retained for existing two-task protocol artifacts.
    output["exact_both_tasks"] = output["exact_all_tasks"]
    return output


def evaluate_full_and_circuit(model, domains, circuits, batch_size):
    full = evaluate(model, domains, batch_size)
    circuit = evaluate(model, domains, batch_size, circuits)
    return {
        "full": full,
        "circuit": circuit,
        "exact_full_and_circuit": (
            full["exact_all_tasks"] and circuit["exact_all_tasks"]
        ),
    }


def candidate_order(programs, synthesis_results, tasks=TASKS):
    per_head = {}
    for task in tasks:
        for key, report in synthesis_results["tasks"][task].items():
            if key not in {f"{layer}.{head}" for layer, head in programs}:
                continue
            score = report.get("score", report)
            per_head.setdefault(key, {"tasks": [], "support_iou": 0.0})
            per_head[key]["tasks"].append(task)
            per_head[key]["support_iou"] = max(
                per_head[key]["support_iou"],
                float(score.get("support_iou", 0.0)),
            )
    # Prefer programs supported by both behaviors, then stronger attention-map
    # fits.  Lexicographic layer/head order makes ties deterministic.
    return sorted(
        programs,
        key=lambda head: (
            -len(per_head.get(f"{head[0]}.{head[1]}", {}).get("tasks", [])),
            -per_head.get(f"{head[0]}.{head[1]}", {}).get("support_iou", 0.0),
            head,
        ),
    ), per_head


def main() -> None:
    args = parse_args()
    tasks = tuple(dict.fromkeys(args.tasks))
    bounded_mode = args.bounded_manifest is not None
    if bounded_mode and (
        args.synthesis_manifest is not None or args.gate_manifest is not None
    ):
        raise ValueError(
            "--bounded_manifest cannot be combined with synthesis/gate manifests"
        )
    if not bounded_mode and (
        args.synthesis_manifest is None or args.gate_manifest is None
    ):
        raise ValueError(
            "Provide --bounded_manifest, or provide both --synthesis_manifest "
            "and --gate_manifest"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = load_model_with_variants(args.model_path, device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    selection_manifest = (
        args.bounded_manifest if bounded_mode else args.synthesis_manifest
    )
    synthesis_domains, synthesis_provenance = build_domains(
        tokenizer, selection_manifest, device, tasks
    )
    gate_domains = gate_provenance = None
    if not bounded_mode:
        gate_domains, gate_provenance = build_domains(
            tokenizer, args.gate_manifest, device, tasks
        )
    circuits = load_circuit_edges(args.circuit_root, tasks)
    base_synthesis = evaluate_full_and_circuit(
        base, synthesis_domains, circuits, args.batch_size
    )
    base_gate = (
        None
        if bounded_mode
        else evaluate_full_and_circuit(
            base, gate_domains, circuits, args.batch_size
        )
    )
    if (
        not base_synthesis["exact_full_and_circuit"]
        or (base_gate is not None and not base_gate["exact_full_and_circuit"])
    ):
        failures = []
        split_reports = [("bounded" if bounded_mode else "synthesis", base_synthesis)]
        if base_gate is not None:
            split_reports.append(("gate", base_gate))
        for split, report in split_reports:
            for forward in ("full", "circuit"):
                for task in tasks:
                    task_report = report[forward][task]
                    if task_report["accuracy_against_P"] != 1.0:
                        failures.append(
                            {
                                "split": split,
                                "forward": forward,
                                "task": task,
                                "accuracy_against_P": task_report[
                                    "accuracy_against_P"
                                ],
                                "correct": task_report["correct"],
                                "rows": task_report["rows"],
                                "mismatch_example_ids": task_report[
                                    "mismatch_example_ids"
                                ],
                                "by_stratum": task_report["by_stratum"],
                            }
                        )
        report = {
            "stage": "base_full_and_circuit_preflight",
            "success": False,
            "failure_reason": (
                "base full model or synthesis-selected circuit is not exact "
                "against P(x) on the locked behavior domain"
            ),
            "selection_or_filtering_performed": False,
            "model_path": args.model_path,
            "circuit_root": args.circuit_root,
            "mode": "bounded_domain" if bounded_mode else "held_out_gate",
            "tasks": list(tasks),
            "bounded_manifest": (
                synthesis_provenance if bounded_mode else None
            ),
            "synthesis_manifest": (
                None if bounded_mode else synthesis_provenance
            ),
            "gate_manifest": gate_provenance,
            "base_bounded": base_synthesis if bounded_mode else None,
            "base_synthesis": None if bounded_mode else base_synthesis,
            "base_gate": base_gate,
            "failures": failures,
        }
        with open(output_dir / "joint_program_report.json", "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps(report, indent=2))
        raise SystemExit(2)

    programs = load_programs(args.programs)
    with open(args.synthesis_results) as handle:
        synthesis_results = json.load(handle)
    order, head_metadata = candidate_order(programs, synthesis_results, tasks)
    variant = "sparsemax"
    model_info = Path(args.model_path) / "model_info.json"
    if model_info.exists():
        with open(model_info) as handle:
            variant = json.load(handle).get("attn_variant", variant)

    selected = []
    rejected = []
    trials = []

    def try_subset(heads):
        candidate = copy.deepcopy(base)
        subset = {head: programs[head] for head in heads}
        install_program_heads(candidate, subset, attention_variant=variant)
        metrics = evaluate_full_and_circuit(
            candidate.eval(), synthesis_domains, circuits, args.batch_size
        )
        del candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    for head in order:
        trial_heads = selected + [head]
        metrics = try_subset(trial_heads)
        accepted = bool(metrics["exact_full_and_circuit"])
        trials.append(
            {
                "pass": "forward_add",
                "head": f"{head[0]}.{head[1]}",
                "selected_count_before": len(selected),
                "accepted": accepted,
                "synthesis": metrics,
            }
        )
        if accepted:
            selected.append(head)
        else:
            rejected.append(head)

    # Interactions can change after later additions.  Reconsider each rejected
    # head once, still using only the declared selection domain.
    still_rejected = []
    for head in rejected:
        metrics = try_subset(selected + [head])
        accepted = bool(metrics["exact_full_and_circuit"])
        trials.append(
            {
                "pass": "forward_readd",
                "head": f"{head[0]}.{head[1]}",
                "selected_count_before": len(selected),
                "accepted": accepted,
                "synthesis": metrics,
            }
        )
        if accepted:
            selected.append(head)
        else:
            still_rejected.append(head)

    selected_programs = {head: programs[head] for head in selected}
    selected_by_task = {
        task: [
            f"{layer}.{head}"
            for layer, head in selected
            if task
            in head_metadata.get(f"{layer}.{head}", {}).get("tasks", [])
        ]
        for task in tasks
    }
    required_heads_by_task = {
        task: sorted(
            {
                node.replace("attn_", "").replace("_h_", ".")
                for edge in circuits[task]
                for node in edge
                if node.startswith("attn_")
            }
        )
        for task in tasks
    }
    selected_head_names = {f"{layer}.{head}" for layer, head in selected}
    all_circuit_heads_replaced = all(
        set(required_heads_by_task[task]) <= selected_head_names
        for task in tasks
    )
    final_model = copy.deepcopy(base)
    install_program_heads(
        final_model, selected_programs, attention_variant=variant
    )
    final_synthesis = evaluate_full_and_circuit(
        final_model.eval(), synthesis_domains, circuits, args.batch_size
    )
    # Held-out mode opens the gate only after selection. Bounded mode has no
    # gate: final_synthesis is exhaustive evaluation of the declared D.
    final_gate = (
        None
        if bounded_mode
        else evaluate_full_and_circuit(
            final_model, gate_domains, circuits, args.batch_size
        )
    )
    success = (
        len(selected) >= args.minimum_program_heads
        and all(
            len(selected_by_task[task]) >= args.minimum_program_heads_per_task
            for task in tasks
        )
        and final_synthesis["exact_full_and_circuit"]
        and (final_gate is None or final_gate["exact_full_and_circuit"])
        and (
            not args.require_all_circuit_heads or all_circuit_heads_replaced
        )
    )
    save_programs(selected_programs, output_dir / "programs_selected.json")
    report = {
        "method": "deterministic_forward_add_then_one_readd",
        "mode": "bounded_domain" if bounded_mode else "held_out_gate",
        "tasks": list(tasks),
        "claim_scope": (
            "exact only on the declared finite bounded domain; no held-out claim"
            if bounded_mode
            else "selection split plus one untouched held-out gate"
        ),
        "selection_or_filtering_performed": True,
        "selection_split": "bounded" if bounded_mode else "synthesis",
        "gate_split_used_for_selection": None if bounded_mode else False,
        "reference_target": "explicit_reference_program_P(x)",
        "model_path": args.model_path,
        "circuit_root": args.circuit_root,
        "source_programs": args.programs,
        "bounded_manifest": synthesis_provenance if bounded_mode else None,
        "synthesis_manifest": None if bounded_mode else synthesis_provenance,
        "gate_manifest": gate_provenance,
        "base_bounded": base_synthesis if bounded_mode else None,
        "base_synthesis": None if bounded_mode else base_synthesis,
        "base_gate": base_gate,
        "candidate_heads": [f"{a}.{b}" for a, b in order],
        "candidate_head_metadata": head_metadata,
        "selected_heads": [f"{a}.{b}" for a, b in selected],
        "selected_heads_by_task": selected_by_task,
        "required_circuit_heads_by_task": required_heads_by_task,
        "require_all_circuit_heads": args.require_all_circuit_heads,
        "all_circuit_heads_replaced": all_circuit_heads_replaced,
        "rejected_heads": [f"{a}.{b}" for a, b in still_rejected],
        "trials": trials,
        "final_bounded": final_synthesis if bounded_mode else None,
        "final_synthesis": None if bounded_mode else final_synthesis,
        "final_gate": final_gate,
        "minimum_program_heads": args.minimum_program_heads,
        "minimum_program_heads_per_task": args.minimum_program_heads_per_task,
        "success": success,
    }
    with open(output_dir / "joint_program_report.json", "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
