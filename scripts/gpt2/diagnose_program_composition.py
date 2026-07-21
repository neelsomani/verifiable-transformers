#!/usr/bin/env python3
"""Record per-input full/circuit behavior before and after program healing."""

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
from scripts.programs import install_program_heads, load_programs


TASKS = ("quote_close", "bracket_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--circuit_root", required=True)
    parser.add_argument("--domain_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--healed_model", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def circuit_edges(circuit_root):
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


def domains(tokenizer, manifest, device):
    result = {}
    provenance = {}
    for task in TASKS:
        examples, task_provenance = load_behavior_examples(task, 0, manifest)
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


def model_rows(model, values, batch_size, edges=None, graph=None):
    decisions = []
    margins = []
    with torch.no_grad():
        for start in range(0, values["input_ids"].size(0), batch_size):
            input_ids = values["input_ids"][start : start + batch_size]
            attention_mask = values["attention_mask"][start : start + batch_size]
            if edges is None:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            else:
                logits = controlled_forward(
                    model,
                    input_ids,
                    attention_mask,
                    edges,
                    graph,
                )
            candidate_logits = select_last_real_logits(
                logits, attention_mask
            )[:, values["candidates"]]
            targets = values["targets"][start : start + batch_size].to(
                candidate_logits.device
            )
            other = 1 - targets
            row_index = torch.arange(targets.numel(), device=targets.device)
            decisions.append(candidate_logits.argmax(dim=-1).cpu())
            margins.append(
                (
                    candidate_logits[row_index, targets]
                    - candidate_logits[row_index, other]
                ).float().cpu()
            )
    return torch.cat(decisions), torch.cat(margins)


def evaluate_configuration(model, task_domains, batch_size, edges_by_task=None):
    output = {}
    graph = (
        build_circuit_graph(model.config.n_layer, model.config.n_head)
        if edges_by_task is not None
        else None
    )
    for task, values in task_domains.items():
        decisions, margins = model_rows(
            model,
            values,
            batch_size,
            None if edges_by_task is None else edges_by_task[task],
            graph,
        )
        targets = values["targets"]
        output[task] = {
            "accuracy_against_P": float(
                (decisions == targets).float().mean().item()
            ),
            "mismatch_example_ids": [
                example.example_id
                for example, decision, target in zip(
                    values["examples"], decisions.tolist(), targets.tolist()
                )
                if decision != target
            ],
            "rows": {
                example.example_id: {
                    "decision_candidate_index": int(decision),
                    "target_candidate_index": int(target),
                    "correct": bool(decision == target),
                    "signed_correct_margin": float(margin),
                }
                for example, decision, target, margin in zip(
                    values["examples"],
                    decisions.tolist(),
                    targets.tolist(),
                    margins.tolist(),
                )
            },
        }
    return output


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = load_model_with_variants(args.base_model, device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    task_domains, provenance = domains(tokenizer, args.domain_manifest, device)
    circuits = circuit_edges(args.circuit_root)
    programs = load_programs(args.programs)
    program_nodes = {
        (layer, head): f"attn_{layer}_h_{head}" for layer, head in programs
    }
    variant = "sparsemax"
    info_path = Path(args.base_model) / "model_info.json"
    if info_path.exists():
        with open(info_path) as handle:
            variant = json.load(handle).get("attn_variant", variant)

    configurations = {
        "M0_base_full": evaluate_configuration(
            base, task_domains, args.batch_size
        ),
        "C_M0_base_circuit_only": evaluate_configuration(
            base, task_domains, args.batch_size, circuits
        ),
    }

    programmed = copy.deepcopy(base)
    install_program_heads(programmed, programs, attention_variant=variant)
    configurations["MP_programmed_unhealed_full"] = evaluate_configuration(
        programmed, task_domains, args.batch_size
    )
    configurations["C_MP_programmed_unhealed_circuit_only"] = (
        evaluate_configuration(programmed, task_domains, args.batch_size, circuits)
    )

    task_union_heads = {}
    for source_task in TASKS:
        subset = {
            head: programs[head]
            for head, node in program_nodes.items()
            if any(node in edge for edge in circuits[source_task])
        }
        task_union_heads[source_task] = [f"{a}.{b}" for a, b in sorted(subset)]
        candidate = copy.deepcopy(base)
        install_program_heads(candidate, subset, attention_variant=variant)
        configurations[f"MP_{source_task}_program_union_full"] = (
            evaluate_configuration(candidate, task_domains, args.batch_size)
        )
        del candidate

    if args.healed_model:
        healed = load_model_with_variants(args.healed_model, device).eval()
        configurations["MH_healed_full"] = evaluate_configuration(
            healed, task_domains, args.batch_size
        )
        configurations["C_MH_healed_circuit_only"] = evaluate_configuration(
            healed, task_domains, args.batch_size, circuits
        )

    # Materialize prompt metadata once instead of repeating it in every model
    # configuration.  Configuration rows join on example_id.
    examples = {
        task: {
            example.example_id: {
                "prompt": example.prompt,
                "correct_token": example.correct_token,
                "incorrect_token": example.incorrect_token,
                "stratum": example.stratum,
                "template_id": example.template_id,
                "metadata": example.metadata,
            }
            for example in values["examples"]
        }
        for task, values in task_domains.items()
    }
    output = {
        "base_model": args.base_model,
        "healed_model": args.healed_model,
        "programs": args.programs,
        "circuit_root": args.circuit_root,
        "domain": provenance,
        "reference_target": "explicit_reference_program_P(x)",
        "task_specific_program_unions": task_union_heads,
        "examples": examples,
        "configurations": configurations,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    summary = {
        name: {
            task: {
                "accuracy_against_P": report[task]["accuracy_against_P"],
                "mismatches": len(report[task]["mismatch_example_ids"]),
            }
            for task in TASKS
        }
        for name, report in configurations.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
