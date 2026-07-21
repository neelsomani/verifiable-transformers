#!/usr/bin/env python3
"""Select and gate a jointly exact subset of synthesized GPT-2 programs.

Programs are proposed and accepted per head during synthesis.  This stage is
the separate composition check: add programs greedily while requiring exact
P(x) accuracy on both synthesis tasks after every addition, then evaluate the
frozen result once on the untouched gate split.  Gate failures are reported,
never adapted to.
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
    parser.add_argument("--synthesis_manifest", required=True)
    parser.add_argument("--gate_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--minimum_program_heads", type=int, default=1)
    parser.add_argument("--minimum_program_heads_per_task", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_domains(tokenizer, manifest_path, device):
    result = {}
    provenance = {}
    for task in TASKS:
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


def load_circuit_edges(circuit_root):
    result = {}
    for task in TASKS:
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
        }
    output["exact_both_tasks"] = all(
        output[task]["accuracy_against_P"] == 1.0 for task in TASKS
    )
    return output


def evaluate_full_and_circuit(model, domains, circuits, batch_size):
    full = evaluate(model, domains, batch_size)
    circuit = evaluate(model, domains, batch_size, circuits)
    return {
        "full": full,
        "circuit": circuit,
        "exact_full_and_circuit": (
            full["exact_both_tasks"] and circuit["exact_both_tasks"]
        ),
    }


def candidate_order(programs, synthesis_results):
    per_head = {}
    for task in TASKS:
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
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = load_model_with_variants(args.model_path, device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    synthesis_domains, synthesis_provenance = build_domains(
        tokenizer, args.synthesis_manifest, device
    )
    gate_domains, gate_provenance = build_domains(
        tokenizer, args.gate_manifest, device
    )
    circuits = load_circuit_edges(args.circuit_root)
    base_synthesis = evaluate_full_and_circuit(
        base, synthesis_domains, circuits, args.batch_size
    )
    base_gate = evaluate_full_and_circuit(
        base, gate_domains, circuits, args.batch_size
    )
    if (
        not base_synthesis["exact_full_and_circuit"]
        or not base_gate["exact_full_and_circuit"]
    ):
        raise RuntimeError(
            "The base full model or its selected circuit is not exact against "
            "P(x) on the preregistered v2 domain. Preserve this result; do not "
            "filter the failing prompts."
        )

    programs = load_programs(args.programs)
    with open(args.synthesis_results) as handle:
        synthesis_results = json.load(handle)
    order, head_metadata = candidate_order(programs, synthesis_results)
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
    # head once, still using only the synthesis split.
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
        for task in TASKS
    }
    final_model = copy.deepcopy(base)
    install_program_heads(
        final_model, selected_programs, attention_variant=variant
    )
    final_synthesis = evaluate_full_and_circuit(
        final_model.eval(), synthesis_domains, circuits, args.batch_size
    )
    # This is the first and only use of the untouched gate split in selection.
    # Its outcome is reported and cannot alter the selected subset.
    final_gate = evaluate_full_and_circuit(
        final_model, gate_domains, circuits, args.batch_size
    )
    success = (
        len(selected) >= args.minimum_program_heads
        and all(
            len(selected_by_task[task]) >= args.minimum_program_heads_per_task
            for task in TASKS
        )
        and final_synthesis["exact_full_and_circuit"]
        and final_gate["exact_full_and_circuit"]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_programs(selected_programs, output_dir / "programs_selected.json")
    report = {
        "method": "deterministic_forward_add_then_one_readd",
        "selection_split": "synthesis",
        "gate_split_used_for_selection": False,
        "reference_target": "explicit_reference_program_P(x)",
        "model_path": args.model_path,
        "circuit_root": args.circuit_root,
        "source_programs": args.programs,
        "synthesis_manifest": synthesis_provenance,
        "gate_manifest": gate_provenance,
        "base_synthesis": base_synthesis,
        "base_gate": base_gate,
        "candidate_heads": [f"{a}.{b}" for a, b in order],
        "candidate_head_metadata": head_metadata,
        "selected_heads": [f"{a}.{b}" for a, b in selected],
        "selected_heads_by_task": selected_by_task,
        "rejected_heads": [f"{a}.{b}" for a, b in still_rejected],
        "trials": trials,
        "final_synthesis": final_synthesis,
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
