#!/usr/bin/env python3
"""Validate and summarize the bounded small-model program-head chase."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FINAL_RUN = Path("artifacts/small_program_healed_chase_round1_core_aware")
FINAL_CIRCUITS = Path(
    "artifacts/small_program_healed_chase_round1_core_aware_circuits"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/program_chase_report.json")
    parser.add_argument(
        "--cost_table", default="artifacts/small-unified-cost-table.json"
    )
    return parser.parse_args()


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circuit_heads(circuit: dict) -> set[str]:
    nodes = set()
    for edge in circuit["edges"]:
        source = edge["source"] if isinstance(edge, dict) else edge[0]
        target = edge["target"] if isinstance(edge, dict) else edge[1]
        nodes.update((source, target))
    return {node for node in nodes if node.startswith("attn_")}


def formal_statuses(verification: dict) -> dict[str, str]:
    return {
        value["property"]: value["status"]
        for value in verification["properties"]
        if value["property"].startswith("projected_")
    }


def main() -> None:
    args = parse_args()
    drift = load(Path("artifacts/mechanism_drift.json"))
    synthesis = load(Path("artifacts/small_program_chase_round1/synthesis_results.json"))
    migration = load(FINAL_CIRCUITS / "migration_report.json")
    programs = load(FINAL_RUN / "checkpoint-final" / "programs.json")
    program_nodes = {
        "attn_{}_h_{}".format(*key.split(".")) for key in programs
    }
    cost_table = load(Path(args.cost_table))
    tasks = {}
    for task in ("quote_close", "bracket_type"):
        circuit_path = FINAL_CIRCUITS / task / "circuit.json"
        verification_path = FINAL_CIRCUITS / task / "verification" / "verification_results.json"
        circuit = load(circuit_path)
        verification = load(verification_path)
        active = circuit_heads(circuit)
        neural = sorted(active - program_nodes)
        statuses = formal_statuses(verification)
        sanity = {
            value["property"]: value["status"]
            for value in verification["properties"]
            if value["property"] in {"pytorch_circuit_validation", "smt_pytorch_sanity"}
        }
        tasks[task] = {
            "circuit": str(circuit_path),
            "verification": str(verification_path),
            "active_program_heads": sorted(active & program_nodes),
            "active_neural_heads": neural,
            "zero_attention_bilinear_terms": not neural,
            "sanity_statuses": sanity,
            "formal_statuses": statuses,
            "all_four_properties_verified": len(statuses) == 4
            and all(status == "VERIFIED" for status in statuses.values()),
            "migration": migration["tasks"][task],
        }

    plain_path = Path(drift["plain_heal_failure_preservation"]["path"])
    plain_hash_after = sha256(plain_path)
    expected_hash = drift["plain_heal_failure_preservation"]["sha256_before_chase"]
    program_cost_rows = [
        row
        for row in cost_table["rows"]
        if row["row_type"] == "selected_circuit" and row["run"] == "program_healed"
    ]
    success = (
        synthesis["tasks"]["quote_close"]["1.1"]["accepted"]
        and migration["migration_pass"]
        and all(value["zero_attention_bilinear_terms"] for value in tasks.values())
        and all(value["all_four_properties_verified"] for value in tasks.values())
        and plain_hash_after == expected_hash
        and len(program_cost_rows) == 2
    )
    payload = {
        "status": "SUCCESS_ZERO_BILINEAR_ONE_ROUND" if success else "FAILED_AUDIT",
        "rounds_used": 1,
        "maximum_rounds_allowed": 2,
        "initial_drift_artifact": "artifacts/mechanism_drift.json",
        "rounds": [
            {
                "round": 1,
                "target": "attn_1_h_1",
                "synthesis": "artifacts/small_program_chase_round1/synthesis_results.json",
                "accepted": synthesis["tasks"]["quote_close"]["1.1"]["accepted"],
                "projected_agreement": synthesis["tasks"]["quote_close"]["1.1"][
                    "projected_agreement"
                ],
                "final_healing": str(FINAL_RUN / "healing_results.json"),
                "final_checkpoint": str(FINAL_RUN / "checkpoint-final"),
            }
        ],
        "diagnostic_healing_attempts": [
            "artifacts/small_program_healed_chase_round1/healing_results.json",
            "artifacts/small_program_healed_chase_round1_individual/healing_results.json",
        ],
        "final_program_heads": sorted(program_nodes),
        "migration_report": str(FINAL_CIRCUITS / "migration_report.json"),
        "migration_pass": migration["migration_pass"],
        "tasks": tasks,
        "regenerated_cost_rows": program_cost_rows,
        "plain_heal_failure_preservation": {
            "path": str(plain_path),
            "sha256_before_chase": expected_hash,
            "sha256_after_chase": plain_hash_after,
            "unchanged": plain_hash_after == expected_hash,
        },
    }
    if not success:
        raise RuntimeError(json.dumps(payload, indent=2))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
