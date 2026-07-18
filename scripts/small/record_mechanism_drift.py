#!/usr/bin/env python3
"""Freeze the pre-chase quote-close mechanism-drift observation as JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/mechanism_drift.json")
    return parser.parse_args()


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def edges(circuit: dict) -> list[list[str]]:
    return [
        [
            edge["source"] if isinstance(edge, dict) else edge[0],
            edge["target"] if isinstance(edge, dict) else edge[1],
        ]
        for edge in circuit["edges"]
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    before_path = Path("artifacts/small_norm_free_circuits/quote_close/circuit.json")
    after_path = Path(
        "artifacts/small_program_healed_ablation_aware_circuits/quote_close/circuit.json"
    )
    migration_path = Path(
        "artifacts/small_program_healed_ablation_aware_circuits/migration_report.json"
    )
    programs_path = Path(
        "artifacts/small_program_healed_ablation_aware/checkpoint-final/programs.json"
    )
    plain_failure_path = Path(
        "artifacts/small_program_healed_circuits/migration_report.json"
    )
    before = load(before_path)
    after = load(after_path)
    migration = load(migration_path)
    programs = load(programs_path)
    program_nodes = {
        "attn_{}_h_{}".format(*key.split(".")) for key in programs
    }
    after_nodes = {node for edge in edges(after) for node in edge}
    new_neural_heads = sorted(
        node
        for node in after_nodes
        if node.startswith("attn_") and node not in program_nodes
    )
    expected_before = [
        ["emb", "attn_0_h_0"],
        ["attn_0_h_0", "mlp_0"],
        ["mlp_0", "mlp_1"],
        ["mlp_1", "logits"],
    ]
    if {tuple(value) for value in edges(before)} != {
        tuple(value) for value in expected_before
    }:
        raise RuntimeError("Norm-free quote circuit no longer matches the pre-chase path")
    if "attn_1_h_1" not in new_neural_heads:
        raise RuntimeError("Expected downstream neural dependency attn_1_h_1 is absent")
    if not migration["migration_pass"]:
        raise RuntimeError("Ablation-aware migration did not pass")

    payload = {
        "status": "OBSERVED_BEFORE_PROGRAM_CHASE",
        "task": "quote_close",
        "before": {
            "checkpoint": "artifacts/small_norm_free/checkpoint-final",
            "circuit": str(before_path),
            "path": ["emb", "attn_0_h_0", "mlp_0", "mlp_1", "logits"],
            "edges": edges(before),
        },
        "after": {
            "checkpoint": "artifacts/small_program_healed_ablation_aware/checkpoint-final",
            "circuit": str(after_path),
            "edges": edges(after),
            "installed_program_heads": sorted(program_nodes),
            "new_downstream_neural_dependencies": new_neural_heads,
        },
        "migration": {
            "report": str(migration_path),
            "passed": True,
            "quote_program_necessary": migration["tasks"]["quote_close"]["necessary"],
            "quote_program_non_bypassed": migration["tasks"]["quote_close"][
                "non_bypassed"
            ],
        },
        "interpretation": (
            "Ablation-aware healing made the installed quote program necessary and "
            "non-bypassed, but the selected exact circuit migrated to a new downstream "
            "neural attention dependency at attn_1_h_1."
        ),
        "plain_heal_failure_preservation": {
            "path": str(plain_failure_path),
            "sha256_before_chase": sha256(plain_failure_path),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
