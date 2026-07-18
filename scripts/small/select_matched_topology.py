#!/usr/bin/env python3
"""Select exact-agreement BandNorm/norm-free circuits with equal edge counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("bandnorm", "norm_free")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandnorm_root", required=True)
    parser.add_argument("--norm_free_root", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def exact(entries: list[dict]) -> list[dict]:
    return [entry for entry in entries if entry["metrics"]["agreement"] == 1.0]


def best_at_edge_count(entries: list[dict], edge_count: int) -> dict:
    return min(
        (entry for entry in entries if entry["n_edges"] == edge_count),
        key=lambda entry: (
            entry["metrics"]["mean_candidate_kl"],
            entry["threshold"],
        ),
    )


def write_selected(output_dir: Path, entry: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("circuit.json", entry["circuit"]),
        ("edge_log.json", entry["edge_log"]),
    ):
        with (output_dir / name).open("w") as handle:
            json.dump(value, handle, indent=2)
    with (output_dir / "selection.json").open("w") as handle:
        json.dump(
            {
                "selection_rule": (
                    "exact projected agreement, shared edge count, then lowest "
                    "candidate KL and threshold"
                ),
                "threshold": entry["threshold"],
                "n_edges": entry["n_edges"],
                "metrics": entry["metrics"],
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    roots = {
        "bandnorm": Path(args.bandnorm_root),
        "norm_free": Path(args.norm_free_root),
    }
    output_root = Path(args.output_root)
    manifest = {
        "comparison_scope": (
            "Only selected pairs with identical task and circuit edge count are "
            "eligible for raw quantitative comparison."
        ),
        "tasks": {},
    }
    for task in ("quote_close", "bracket_type"):
        sweeps = {
            variant: exact(load(root / task / "threshold_sweep.json"))
            for variant, root in roots.items()
        }
        edge_counts = {
            variant: sorted({entry["n_edges"] for entry in entries})
            for variant, entries in sweeps.items()
        }
        common = sorted(set(edge_counts["bandnorm"]) & set(edge_counts["norm_free"]))
        if not common:
            manifest["tasks"][task] = {
                "status": "SKIPPED_NO_EQUAL_EDGE_TOPOLOGY",
                "eligible_edge_counts": edge_counts,
            }
            continue

        edge_count = common[0]
        selections = {
            variant: best_at_edge_count(entries, edge_count)
            for variant, entries in sweeps.items()
        }
        for variant, entry in selections.items():
            write_selected(output_root / task / variant, entry)
        manifest["tasks"][task] = {
            "status": "SELECTED",
            "matched_edge_count": edge_count,
            "eligible_edge_counts": edge_counts,
            "variants": {
                variant: {
                    "source_sweep": str(roots[variant] / task / "threshold_sweep.json"),
                    "threshold": entry["threshold"],
                    "n_edges": entry["n_edges"],
                    "metrics": entry["metrics"],
                }
                for variant, entry in selections.items()
            },
        }

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "selection.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
