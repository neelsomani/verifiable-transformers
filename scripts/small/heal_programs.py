#!/usr/bin/env python3
"""Freeze synthesized attention programs and heal the remaining small model."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.programs import install_program_heads, load_programs, save_programs
from scripts.small import SmallVerifiableDataset, vocab
from scripts.small.config import SmallVerifiableConfig
from scripts.small.extract import load_model
from scripts.small.train import evaluate_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--learning_rates", type=float, nargs="+", default=[3e-3, 1e-3, 3e-4]
    )
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--perfect_evals", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def exhaustive_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = SmallVerifiableDataset(task_sampling="all")
    examples = dataset.all_examples
    return (
        torch.tensor([example["input_ids"] for example in examples], device=device),
        torch.tensor([example["target"] for example in examples], device=device),
    )


def evaluate(model, device: torch.device) -> dict:
    return {
        task: evaluate_task(model, task, device)
        for task in ("quote_close", "bracket_type")
    }


def train_attempt(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    learning_rate: float,
    max_steps: int,
    eval_every: int,
    perfect_evals: int,
    weight_decay: float,
) -> tuple[bool, list[dict]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    consecutive_perfect = 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids).logits[:, -1, :]
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite healing loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == max_steps:
            metrics = evaluate(model, input_ids.device)
            perfect = all(
                value["candidate_accuracy"] == 1.0 for value in metrics.values()
            )
            consecutive_perfect = consecutive_perfect + 1 if perfect else 0
            record = {
                "step": step,
                "loss": float(loss.detach().item()),
                "metrics": metrics,
                "all_tasks_perfect": perfect,
                "consecutive_perfect_evals": consecutive_perfect,
            }
            history.append(record)
            print(
                f"lr={learning_rate:g} step={step}: loss={record['loss']:.6f}, "
                f"quote={metrics['quote_close']['candidate_accuracy']:.4f}, "
                f"bracket={metrics['bracket_type']['candidate_accuracy']:.4f}"
            )
            if consecutive_perfect >= perfect_evals:
                return True, history
    return False, history


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(os.path.dirname(args.checkpoint), "config.json")
    config = SmallVerifiableConfig.load(config_path)
    programs = load_programs(args.programs)
    input_ids, targets = exhaustive_batch(device)
    attempts = []
    winning_model = None
    started = time.perf_counter()

    for learning_rate in args.learning_rates:
        model = load_model(args.checkpoint, config, device)
        install_program_heads(
            model, programs, attention_variant=config.attn_variant
        )
        initial_metrics = evaluate(model, device)
        success, history = train_attempt(
            model,
            input_ids,
            targets,
            learning_rate=learning_rate,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            perfect_evals=args.perfect_evals,
            weight_decay=args.weight_decay,
        )
        attempts.append(
            {
                "learning_rate": learning_rate,
                "initial_metrics": initial_metrics,
                "success": success,
                "history": history,
            }
        )
        if success:
            winning_model = model
            break

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(args.output_dir, "config.json"))
    vocab.save_vocab(os.path.join(args.output_dir, "vocab.json"))
    result = {
        "source_checkpoint": args.checkpoint,
        "programs": args.programs,
        "program_heads": [f"{layer}.{head}" for layer, head in sorted(programs)],
        "programs_frozen": True,
        "success": winning_model is not None,
        "acceptance": {
            "candidate_accuracy_each_task": 1.0,
            "consecutive_evaluations": args.perfect_evals,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "attempts": attempts,
    }
    with open(os.path.join(args.output_dir, "healing_results.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    if winning_model is None:
        raise RuntimeError("No healing attempt achieved the B3 exit criterion")

    final_dir = os.path.join(args.output_dir, "checkpoint-final")
    winning_model.save_pretrained(final_dir)
    save_programs(programs, os.path.join(final_dir, "programs.json"))
    print(f"Saved healed program model to {final_dir}")


if __name__ == "__main__":
    main()
