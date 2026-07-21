#!/usr/bin/env python3
"""Build and validate deterministic GPT-2 behavior-domain v2 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from transformers import GPT2TokenizerFast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.behavior_domains import (
    PROTOCOL_ID,
    TASKS,
    canonical_json_sha256,
    generate_v2_splits,
    legacy_examples,
    write_domain_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--config", default="configs/gpt2_behavior_domain_v2.json"
    )
    return parser.parse_args()


def tokenizer_fingerprint(tokenizer) -> dict:
    vocab = tokenizer.get_vocab()
    digest = hashlib.sha256(
        json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": tokenizer.name_or_path,
        "vocab_size": len(vocab),
        "vocab_sha256": digest,
    }


def validate_examples(tokenizer, tasks, requirements, *, enforce_position_range=True) -> dict:
    result = {}
    for task in TASKS:
        rows = tasks[task]
        prompts = [row.prompt for row in rows]
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"{task} manifest contains repeated prompts")
        positions = set()
        contextual_openers = set()
        for row in rows:
            for token in (row.correct_token, row.incorrect_token):
                if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
                    raise ValueError(
                        f"Candidate {token!r} for {row.example_id} is not one token"
                    )
            encoded = tokenizer(
                row.prompt,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            opener_index = int(row.metadata["opener_char_index"])
            covering = [
                index
                for index, (start, stop) in enumerate(encoded.offset_mapping)
                if start <= opener_index < stop
            ]
            if len(covering) != 1:
                raise ValueError(
                    f"Could not align opener for {row.example_id}: {covering}"
                )
            token_index = covering[0]
            positions.add(token_index)
            contextual_openers.add(int(encoded.input_ids[token_index]))
            row.metadata["opener_token_index"] = token_index
            row.metadata["opener_context_token_id"] = int(
                encoded.input_ids[token_index]
            )
            row.metadata["prompt_token_count"] = len(encoded.input_ids)
        minimum = int(requirements["minimum_distinct_opener_token_positions"])
        if enforce_position_range and len(positions) < minimum:
            raise ValueError(
                f"{task} has {len(positions)} opener positions; requires {minimum}"
            )
        result[task] = {
            "candidate_tokens_are_single_token": True,
            "opener_alignment_checked": True,
            "opener_token_positions": sorted(positions),
            "distinct_opener_token_positions": len(positions),
            "contextual_opener_token_ids": sorted(contextual_openers),
        }
    return result


def main() -> None:
    args = parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("The config does not declare the v2 behavior protocol")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    splits = generate_v2_splits(
        seed=int(config["seed"]),
        split_sizes={
            name: int(size)
            for name, size in config["split_sizes_per_task"].items()
        },
    )
    requirements = config["requirements"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_meta = tokenizer_fingerprint(tokenizer)
    index = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "config": str(Path(args.config).resolve()),
        "config_sha256": canonical_json_sha256(config),
        "tokenizer": tokenizer_meta,
        "manifests": {},
    }
    for split, tasks in splits.items():
        validation = validate_examples(tokenizer, tasks, requirements)
        path = output_dir / f"{split}.json"
        manifest = write_domain_manifest(
            path,
            split=split,
            examples=tasks,
            config=config,
            tokenizer_metadata=tokenizer_meta,
            validation=validation,
        )
        index["manifests"][split] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "summary": manifest["summary"],
        }

    legacy_tasks = {
        task: legacy_examples(task, int(config["legacy_regression_rows_per_task"]))
        for task in TASKS
    }
    legacy_validation = validate_examples(
        tokenizer,
        legacy_tasks,
        requirements,
        enforce_position_range=False,
    )
    legacy_path = output_dir / "legacy_regression.json"
    legacy_manifest = write_domain_manifest(
        legacy_path,
        split="legacy_regression",
        examples=legacy_tasks,
        config=config,
        tokenizer_metadata=tokenizer_meta,
        validation=legacy_validation,
    )
    index["manifests"]["legacy_regression"] = {
        "path": str(legacy_path.resolve()),
        "sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
        "summary": legacy_manifest["summary"],
    }
    index_path = output_dir / "manifest_index.json"
    with open(index_path, "w") as handle:
        json.dump(index, handle, indent=2)
        handle.write("\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
