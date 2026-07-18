#!/usr/bin/env python3
"""Fine-tune a standard-LN small model through complete norm removal."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.norm_removal import (
    fold_attenuated_layernorms,
    install_attenuated_layernorms,
    update_attenuation_schedule,
)
from scripts.small import SmallVerifiableDataset, collate_fn, vocab
from scripts.small.config import SmallVerifiableConfig
from scripts.small.extract import load_model
from scripts.small.train import evaluate_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--calibration_steps", type=int, default=40)
    parser.add_argument("--transition_steps", type=int, default=40)
    parser.add_argument("--gap_steps", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--ema_momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_both(model, device: torch.device) -> dict:
    return {
        task: evaluate_task(model, task, device)
        for task in ("quote_close", "bracket_type")
    }


def exhaustive_logits(model, device: torch.device) -> torch.Tensor:
    dataset = SmallVerifiableDataset(task_sampling="all")
    input_ids = torch.tensor(
        [example["input_ids"] for example in dataset.all_examples],
        dtype=torch.long,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        return model(input_ids).logits.detach().cpu()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(os.path.dirname(args.checkpoint), "config.json")
    config = SmallVerifiableConfig.load(config_path)
    if config.norm_variant != "layer_norm":
        raise ValueError(
            f"Expected a standard-LN checkpoint, got norm_variant={config.norm_variant}"
        )
    model = load_model(args.checkpoint, config, device)
    entries = install_attenuated_layernorms(model, momentum=args.ema_momentum)
    model.to(device)

    dataset = SmallVerifiableDataset(task_sampling="all")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    batches = cycle(loader)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    schedule_end = (
        args.calibration_steps
        + (len(entries) - 1) * args.gap_steps
        + args.transition_steps
    )
    history = []
    perfect_after_removal = 0
    print(f"Device: {device}")
    print(f"Norm schedule: {[entry.name for entry in entries]}")
    print(f"All norms fully attenuated at step {schedule_end}")

    for step in range(args.max_steps + 1):
        schedule = update_attenuation_schedule(
            entries,
            step,
            calibration_steps=args.calibration_steps,
            transition_steps=args.transition_steps,
            gap_steps=args.gap_steps,
        )
        if step > 0:
            model.train()
            batch = next(batches)
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids).logits[:, -1, :]
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite removal loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            loss = torch.tensor(float("nan"))

        if step % args.eval_every == 0 or step == schedule_end:
            metrics = evaluate_both(model, device)
            all_perfect = all(
                value["candidate_accuracy"] == 1.0 for value in metrics.values()
            )
            record = {
                "step": step,
                "loss": None if step == 0 else float(loss.item()),
                "schedule": schedule,
                "fixed_std": {
                    entry.name: float(entry.module.fixed_std.item()) for entry in entries
                },
                "metrics": metrics,
                "all_perfect": all_perfect,
            }
            history.append(record)
            summary = ", ".join(
                f"{task}={value['candidate_accuracy']:.3f}"
                for task, value in metrics.items()
            )
            print(f"step={step:4d} loss={record['loss']} {summary}")

            if step >= schedule_end and all_perfect:
                perfect_after_removal += 1
            else:
                perfect_after_removal = 0
            if perfect_after_removal >= 3:
                break

    final_metrics = evaluate_both(model, device)
    success = all(
        value["candidate_accuracy"] == 1.0 for value in final_metrics.values()
    ) and all(float(entry.module.attenuation.item()) == 1.0 for entry in entries)
    if not success:
        failure = {
            "status": "failed",
            "schedule_end": schedule_end,
            "history": history,
            "final_metrics": final_metrics,
        }
        with open(os.path.join(args.output_dir, "removal_metrics.json"), "w") as handle:
            json.dump(failure, handle, indent=2)
        raise RuntimeError("LayerNorm removal did not retain 100% candidate accuracy")

    before_fold = exhaustive_logits(model, device)
    fold_attenuated_layernorms(model)
    after_fold = exhaustive_logits(model, device)
    fold_max_abs_diff = float((before_fold - after_fold).abs().max().item())
    if fold_max_abs_diff > 1e-4:
        raise RuntimeError(f"Affine fold changed logits by {fold_max_abs_diff}")

    config.norm_variant = "none"
    config.tie_embeddings = False
    config.save(os.path.join(args.output_dir, "config.json"))
    vocab.save_vocab(os.path.join(args.output_dir, "vocab.json"))
    final_dir = os.path.join(args.output_dir, "checkpoint-final")
    model.save_pretrained(final_dir)
    final_metrics = evaluate_both(model, device)
    result = {
        "status": "passed",
        "method": "sequential_fixed_std_attenuation_and_affine_folding",
        "source_checkpoint": args.checkpoint,
        "schedule_end": schedule_end,
        "fold_max_abs_diff": fold_max_abs_diff,
        "final_metrics": final_metrics,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "removal_metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"LayerNorm removal PASSED; fold max abs diff={fold_max_abs_diff:.6g}")
    print(f"Saved norm-free checkpoint to {final_dir}")


if __name__ == "__main__":
    main()
