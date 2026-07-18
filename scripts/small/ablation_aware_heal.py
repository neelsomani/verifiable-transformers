#!/usr/bin/env python3
"""Heal programs while suppressing paths that can bypass the intended head."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.circuits import CircuitGraph, controlled_forward
from scripts.programs import install_program_heads, load_programs, save_programs
from scripts.small import get_eval_dataset, vocab
from scripts.small.config import SmallVerifiableConfig
from scripts.small.extract import evaluate_circuit, load_model


TASKS = ("quote_close", "bracket_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--circuit_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--perfect_evals", type=int, default=3)
    parser.add_argument("--non_circuit_keep_probability", type=float, default=0.25)
    parser.add_argument("--circuit_loss_weight", type=float, default=1.0)
    parser.add_argument("--sampled_loss_weight", type=float, default=1.0)
    parser.add_argument("--bypass_penalty_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_edges(path: str) -> set[tuple[str, str]]:
    with open(path) as handle:
        circuit = json.load(handle)
    return {
        (edge["source"], edge["target"])
        if isinstance(edge, dict)
        else tuple(edge)
        for edge in circuit["edges"]
    }


def task_batch(task: str, device: torch.device):
    examples = get_eval_dataset(task)
    return (
        torch.tensor([example["input_ids"] for example in examples], device=device),
        torch.tensor([example["target"] for example in examples], device=device),
    )


def candidate_uniform_penalty(logits: torch.Tensor, task: str) -> torch.Tensor:
    candidates = sorted(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task]))
    candidate_logits = logits[:, -1, candidates]
    uniform = torch.full_like(candidate_logits, 1.0 / len(candidates))
    return F.kl_div(
        F.log_softmax(candidate_logits, dim=-1), uniform, reduction="batchmean"
    )


def migration_metrics(model, graph, full_edges, circuits, intended, batches, device):
    reports = {}
    for task in TASKS:
        intended_nodes = set(intended[task])
        without_heads = {
            edge for edge in full_edges if edge[0] not in intended_nodes
        }
        individual = {}
        core_individual = {}
        for node in intended[task]:
            without_one = {edge for edge in full_edges if edge[0] != node}
            metrics = evaluate_circuit(model, task, without_one, graph, device)
            individual[node] = {
                "metrics": metrics,
                "necessary": metrics["agreement"] < 1.0,
            }
            core_without_one = {
                edge for edge in circuits[task] if edge[0] != node
            }
            core_metrics = evaluate_circuit(
                model, task, core_without_one, graph, device
            )
            core_individual[node] = {
                "metrics": core_metrics,
                "necessary": core_metrics["agreement"] < 1.0,
            }
        reports[task] = {
            "full": evaluate_circuit(model, task, full_edges, graph, device),
            "individual_ablations": individual,
            "without_intended_program_heads": evaluate_circuit(
                model, task, without_heads, graph, device
            ),
            "preregistered_circuit": evaluate_circuit(
                model, task, circuits[task], graph, device
            ),
            "preregistered_circuit_individual_ablations": core_individual,
        }
    return reports


def main() -> None:
    args = parse_args()
    if not 0 <= args.non_circuit_keep_probability <= 1:
        raise ValueError("non_circuit_keep_probability must be in [0, 1]")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(os.path.dirname(args.checkpoint), "config.json")
    config = SmallVerifiableConfig.load(config_path)
    programs = load_programs(args.programs)
    program_nodes = {
        (layer, head): f"attn_{layer}_h_{head}" for layer, head in programs
    }
    graph = CircuitGraph(config.n_layers, config.n_heads)
    full_edges = graph.get_edges()
    circuits = {
        task: load_edges(os.path.join(args.circuit_root, task, "circuit.json"))
        for task in TASKS
    }
    intended = {}
    for task, edges in circuits.items():
        retained = sorted(
            node for node in program_nodes.values() if any(node in edge for edge in edges)
        )
        if not retained:
            raise RuntimeError(
                f"Expected at least one intended program head for {task}, got {retained}"
            )
        intended[task] = retained

    model = load_model(args.checkpoint, config, device)
    install_program_heads(model, programs, attention_variant=config.attn_variant)
    batches = {task: task_batch(task, device) for task in TASKS}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    consecutive_passes = 0
    started = time.perf_counter()

    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        full_inputs = torch.cat([batches[task][0] for task in TASKS])
        full_targets = torch.cat([batches[task][1] for task in TASKS])
        full_loss = F.cross_entropy(
            model(input_ids=full_inputs).logits[:, -1, :], full_targets
        )
        circuit_loss = full_loss.new_zeros(())
        sampled_loss = full_loss.new_zeros(())
        bypass_penalty = full_loss.new_zeros(())

        for task in TASKS:
            input_ids, targets = batches[task]
            core_edges = circuits[task]
            core_logits = controlled_forward(model, input_ids, core_edges, graph)
            circuit_loss = circuit_loss + F.cross_entropy(
                core_logits[:, -1, :], targets
            )

            sampled_edges = set(core_edges)
            for edge in full_edges - core_edges:
                if random.random() < args.non_circuit_keep_probability:
                    sampled_edges.add(edge)
            sampled_logits = controlled_forward(
                model, input_ids, sampled_edges, graph
            )
            sampled_loss = sampled_loss + F.cross_entropy(
                sampled_logits[:, -1, :], targets
            )

            intended_nodes = set(intended[task])
            ablation_edge_sets = [
                {edge for edge in full_edges if edge[0] not in intended_nodes},
                {edge for edge in core_edges if edge[0] not in intended_nodes},
            ]
            ablation_edge_sets.extend(
                {edge for edge in full_edges if edge[0] != node}
                for node in intended[task]
            )
            ablation_edge_sets.extend(
                {edge for edge in core_edges if edge[0] != node}
                for node in intended[task]
            )
            for ablated_edges in ablation_edge_sets:
                bypass_logits = controlled_forward(
                    model, input_ids, ablated_edges, graph
                )
                bypass_penalty = bypass_penalty + candidate_uniform_penalty(
                    bypass_logits, task
                )

        loss = (
            full_loss
            + args.circuit_loss_weight * circuit_loss
            + args.sampled_loss_weight * sampled_loss
            + args.bypass_penalty_weight * bypass_penalty
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite ablation-aware healing loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.max_steps:
            reports = migration_metrics(
                model, graph, full_edges, circuits, intended, batches, device
            )
            passed = all(
                report["full"]["candidate_accuracy"] == 1.0
                and report["without_intended_program_heads"]["agreement"] < 1.0
                and all(
                    ablation["necessary"]
                    for ablation in report["individual_ablations"].values()
                )
                and report["preregistered_circuit"]["candidate_accuracy"] == 1.0
                and all(
                    ablation["necessary"]
                    for ablation in report[
                        "preregistered_circuit_individual_ablations"
                    ].values()
                )
                for report in reports.values()
            )
            consecutive_passes = consecutive_passes + 1 if passed else 0
            record = {
                "step": step,
                "loss": float(loss.detach().item()),
                "full_loss": float(full_loss.detach().item()),
                "circuit_loss": float(circuit_loss.detach().item()),
                "sampled_loss": float(sampled_loss.detach().item()),
                "bypass_penalty": float(bypass_penalty.detach().item()),
                "migration": reports,
                "passed": passed,
                "consecutive_passes": consecutive_passes,
            }
            history.append(record)
            print(
                f"step={step} loss={record['loss']:.5f} "
                + " ".join(
                    f"{task}:full={reports[task]['full']['candidate_accuracy']:.3f},"
                    "ablated="
                    f"{reports[task]['without_intended_program_heads']['agreement']:.3f}"
                    for task in TASKS
                )
            )
            if consecutive_passes >= args.perfect_evals:
                break

    success = consecutive_passes >= args.perfect_evals
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(args.output_dir, "config.json"))
    vocab.save_vocab(os.path.join(args.output_dir, "vocab.json"))
    result = {
        "source_checkpoint": args.checkpoint,
        "programs": args.programs,
        "circuit_root": args.circuit_root,
        "programs_frozen": True,
        "method": (
            "sampled_non_circuit_ablation_with_combined_and_individual_"
            "program_bypass_suppression_on_full_and_preregistered_circuits"
        ),
        "intended_program_heads": intended,
        "success": success,
        "elapsed_seconds": time.perf_counter() - started,
        "hyperparameters": vars(args),
        "history": history,
    }
    with open(os.path.join(args.output_dir, "healing_results.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    if not success:
        raise RuntimeError("Ablation-aware healing did not pass the B3/B4 criteria")

    final_dir = os.path.join(args.output_dir, "checkpoint-final")
    model.save_pretrained(final_dir)
    save_programs(programs, os.path.join(final_dir, "programs.json"))
    print(f"Saved ablation-aware healed model to {final_dir}")


if __name__ == "__main__":
    main()
