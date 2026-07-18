#!/usr/bin/env python3
"""Check GPT-2 program-head necessity and neural-path bypass after healing."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.extract import (
    BEHAVIOR_GENERATORS,
    build_circuit_graph,
    controlled_forward,
    get_candidate_token_ids,
    load_model_with_variants,
    select_last_real_logits,
)
from scripts.programs import load_programs


HEAD_PATTERN = re.compile(r"^attn_(\d+)_h_(\d+)$")
TASKS = ("quote_close", "bracket_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--circuit_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_examples", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def circuit_edges(circuit: dict) -> set[tuple[str, str]]:
    return {
        (edge["source"], edge["target"])
        if isinstance(edge, dict)
        else tuple(edge)
        for edge in circuit["edges"]
    }


def projected_metrics(reference, candidate, attention_mask, candidates):
    reference_last = select_last_real_logits(reference, attention_mask)[:, candidates]
    candidate_last = select_last_real_logits(candidate, attention_mask)[:, candidates]
    reference_decision = reference_last.argmax(dim=-1)
    candidate_decision = candidate_last.argmax(dim=-1)
    return {
        "projected_agreement": float(
            (reference_decision == candidate_decision).float().mean().item()
        ),
        "mean_candidate_kl": float(
            torch.nn.functional.kl_div(
                torch.log_softmax(candidate_last, dim=-1),
                torch.softmax(reference_last, dim=-1),
                reduction="batchmean",
            ).item()
        ),
    }


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_with_variants(args.model_path, device).eval()
    programs = load_programs(os.path.join(args.model_path, "programs.json"))
    program_nodes = {
        f"attn_{layer}_h_{head}" for layer, head in programs
    }
    graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
    full_edges = graph.get_edges()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    reports = {}

    for task in TASKS:
        with open(os.path.join(args.circuit_root, task, "circuit.json")) as handle:
            circuit = json.load(handle)
        retained = circuit_edges(circuit)
        intended = sorted(
            node for node in program_nodes if any(node in edge for edge in retained)
        )
        if not intended:
            raise RuntimeError(f"{task} circuit contains no program head")
        examples = BEHAVIOR_GENERATORS[task](args.num_examples)
        encoded = tokenizer(
            [example.prompt for example in examples],
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        candidates = get_candidate_token_ids(task, tokenizer)
        with torch.no_grad():
            reference = controlled_forward(
                model, input_ids, attention_mask, full_edges, graph
            )
            circuit_logits = controlled_forward(
                model, input_ids, attention_mask, retained, graph
            )
        individual = {}
        for node in intended:
            edges = {edge for edge in full_edges if edge[0] != node}
            with torch.no_grad():
                logits = controlled_forward(
                    model, input_ids, attention_mask, edges, graph
                )
            metrics = projected_metrics(
                reference, logits, attention_mask, candidates
            )
            individual[node] = {
                "metrics": metrics,
                "necessary": metrics["projected_agreement"] < 1.0,
            }
        without_intended = {
            edge for edge in full_edges if edge[0] not in intended
        }
        without_all = {
            edge for edge in full_edges if edge[0] not in program_nodes
        }
        with torch.no_grad():
            intended_logits = controlled_forward(
                model, input_ids, attention_mask, without_intended, graph
            )
            all_logits = controlled_forward(
                model, input_ids, attention_mask, without_all, graph
            )
        intended_metrics = projected_metrics(
            reference, intended_logits, attention_mask, candidates
        )
        reports[task] = {
            "intended_program_heads": intended,
            "circuit": projected_metrics(
                reference, circuit_logits, attention_mask, candidates
            ),
            "individual_ablations": individual,
            "without_intended_program_heads": intended_metrics,
            "without_all_program_heads": projected_metrics(
                reference, all_logits, attention_mask, candidates
            ),
            "necessary": all(value["necessary"] for value in individual.values()),
            "non_bypassed": intended_metrics["projected_agreement"] < 1.0,
        }

    output = {
        "model_path": args.model_path,
        "circuit_root": args.circuit_root,
        "program_nodes": sorted(program_nodes),
        "num_examples": args.num_examples,
        "tasks": reports,
        "migration_pass": all(
            report["circuit"]["projected_agreement"] == 1.0
            and report["necessary"]
            and report["non_bypassed"]
            for report in reports.values()
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))
    if not output["migration_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
