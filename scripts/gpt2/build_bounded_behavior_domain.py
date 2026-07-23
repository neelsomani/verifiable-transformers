#!/usr/bin/env python3
"""Build the fixed Phase-C bounded domain from the two frozen v4 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.behavior_domains import (
    TASKS,
    canonical_json_sha256,
    load_domain_manifest,
    write_domain_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/gpt2_behavior_domain_bounded_v1.json"
    )
    parser.add_argument(
        "--output", default="artifacts/gpt2-behavior-domain-bounded-v1/domain.json"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build(config_path: Path, output_path: Path, root: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config.get("protocol_id") != "gpt2_behavior_domain_bounded_v1":
        raise ValueError("Bounded builder requires the bounded-v1 protocol ID")
    sources = []
    for source in config["source_manifests"]:
        path = resolve(root, source["path"])
        actual_sha = sha256(path)
        if actual_sha != source["sha256"]:
            raise ValueError(
                f"Frozen source manifest changed: {path}: "
                f"{actual_sha} != {source['sha256']}"
            )
        manifest = load_domain_manifest(path)
        if manifest.get("protocol_id") != "gpt2_behavior_domain_v4":
            raise ValueError(f"Bounded source is not protocol v4: {path}")
        if manifest.get("split") != source["name"]:
            raise ValueError(
                f"Bounded source split mismatch: {path}: "
                f"{manifest.get('split')!r} != {source['name']!r}"
            )
        sources.append((source, path, manifest))

    vocab_hashes = {manifest["tokenizer"]["vocab_sha256"] for _, _, manifest in sources}
    if len(vocab_hashes) != 1:
        raise ValueError("Bounded sources use different tokenizer vocabularies")

    combined = {}
    validation = {}
    expected_rows = int(config["bounded_rows_per_task"])
    for task in TASKS:
        rows = [
            row
            for _, _, manifest in sources
            for row in manifest["loaded_examples"][task]
        ]
        prompts = [row.prompt for row in rows]
        example_ids = [row.example_id for row in rows]
        if len(rows) != expected_rows:
            raise ValueError(f"{task} bounded rows: {len(rows)} != {expected_rows}")
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"{task} bounded domain repeats prompts")
        if len(set(example_ids)) != len(example_ids):
            raise ValueError(f"{task} bounded domain repeats example IDs")
        counts = Counter(row.stratum for row in rows)
        if dict(counts) != config["required_strata_rows"][task]:
            raise ValueError(
                f"{task} bounded stratum counts changed: "
                f"{dict(counts)} != {config['required_strata_rows'][task]}"
            )
        positions = sorted(
            {int(row.metadata["opener_token_index"]) for row in rows}
        )
        validation[task] = {
            "source_manifests_revalidated": True,
            "unique_prompts": True,
            "unique_example_ids": True,
            "candidate_tokens_are_single_token": True,
            "opener_alignment_checked_in_sources": True,
            "opener_token_positions": positions,
            "strata": dict(counts),
        }
        combined[task] = rows

    payload = write_domain_manifest(
        output_path,
        split="bounded",
        examples=combined,
        config=config,
        tokenizer_metadata=sources[0][2]["tokenizer"],
        validation=validation,
        protocol_id=config["protocol_id"],
    )
    for task in TASKS:
        actual = payload["summary"][task]["prompt_set_sha256"]
        expected = config["locked_prompt_set_sha256"][task]
        if actual != expected:
            raise ValueError(
                f"Locked bounded digest changed for {task}: {actual} != {expected}"
            )
    payload.update(
        {
            "claim_type": config["claim_type"],
            "held_out_gate": False,
            "claim_scope": config["relationship_to_v4"],
            "source_manifests": [
                {
                    "name": source["name"],
                    "path": source["path"],
                    "sha256": source["sha256"],
                    "prompt_set_sha256": {
                        task: manifest["summary"][task]["prompt_set_sha256"]
                        for task in TASKS
                    },
                }
                for source, path, manifest in sources
            ],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    index = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "config": display_path(root, config_path),
        "config_sha256": canonical_json_sha256(config),
        "manifest": display_path(root, output_path),
        "manifest_sha256": sha256(output_path),
        "summary": payload["summary"],
        "held_out_gate": False,
    }
    (output_path.parent / "manifest_index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    return index


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = resolve(root, args.config)
    output_path = resolve(root, args.output)
    print(json.dumps(build(config_path, output_path, root), indent=2))


if __name__ == "__main__":
    main()
