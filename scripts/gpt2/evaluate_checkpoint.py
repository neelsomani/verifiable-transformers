#!/usr/bin/env python3
"""Measure a variant checkpoint on the registered processed OWT validation set."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.extract import load_model_with_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--processed_dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size_per_device", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model_with_variants(args.model_path, "cpu")
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = load_from_disk(args.processed_dataset_dir)
    eval_dataset = datasets["validation"]
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(args.max_eval_samples, len(eval_dataset)))
        )

    output_parent = os.path.dirname(os.path.abspath(args.output))
    training_args = TrainingArguments(
        output_dir=output_parent,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=args.batch_size_per_device,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )
    metrics = trainer.evaluate()
    eval_loss = float(metrics["eval_loss"])
    result = {
        "model_path": os.path.abspath(args.model_path),
        "processed_dataset_dir": os.path.abspath(args.processed_dataset_dir),
        "eval_examples": len(eval_dataset),
        "eval_loss": eval_loss,
        "eval_perplexity": math.exp(eval_loss),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    if trainer.is_world_process_zero():
        os.makedirs(output_parent, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
