#!/usr/bin/env python3
"""Build the unified GPT-2 component cost table, including pending run outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os


DOCUMENTED_ROWS = [
    {
        "component": "baseline",
        "owt_eval_loss": 3.1340,
        "owt_loss_delta": 0.0,
        "owt_eval_perplexity": math.exp(3.1340),
        "owt_perplexity_delta": 0.0,
        "relative_owt_perplexity_increase": 0.0,
        "wikitext_perplexity": 52.9820,
        "wikitext_perplexity_delta": 0.0,
        "replacement_fraction": None,
        "status": "measured",
    },
    {
        "component": "sparsemax",
        "owt_eval_loss": 3.1973,
        "owt_loss_delta": 0.0633,
        "owt_eval_perplexity": math.exp(3.1973),
        "owt_perplexity_delta": math.exp(3.1973) - math.exp(3.1340),
        "relative_owt_perplexity_increase": math.exp(0.0633) - 1.0,
        "wikitext_perplexity": 55.7227,
        "wikitext_perplexity_delta": 2.7407,
        "replacement_fraction": None,
        "status": "measured",
    },
    {
        "component": "layernorm+sparsemax+leaky_relu",
        "owt_eval_loss": 3.1968865394592285,
        "owt_loss_delta": 3.1968865394592285 - 3.1340,
        "owt_eval_perplexity": 24.456267913785197,
        "owt_perplexity_delta": 24.456267913785197 - math.exp(3.1340),
        "relative_owt_perplexity_increase": (
            24.456267913785197 / math.exp(3.1340) - 1.0
        ),
        "wikitext_perplexity": 57.18547510376165,
        "wikitext_perplexity_delta": 57.18547510376165 - 52.9820,
        "replacement_fraction": None,
        "status": "measured; removal input; A4 run 1 of 2",
    },
    {
        "component": "bandnorm+sparsemax+leaky_relu",
        "owt_eval_loss": 3.3300,
        "owt_loss_delta": 0.1960,
        "owt_eval_perplexity": math.exp(3.3300),
        "owt_perplexity_delta": math.exp(3.3300) - math.exp(3.1340),
        "relative_owt_perplexity_increase": math.exp(0.1960) - 1.0,
        "wikitext_perplexity": 62.11,
        "wikitext_perplexity_delta": 9.128,
        "replacement_fraction": None,
        "status": "measured",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--removal_metrics", default="artifacts/gpt2-norm-free/removal_metrics.json")
    parser.add_argument("--program_metrics", default="artifacts/gpt2-program-healed/healing_results.json")
    parser.add_argument("--synthesis_metrics", default="artifacts/gpt2-programs/synthesis_results.json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def load_if_present(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    rows = list(DOCUMENTED_ROWS)
    removal = load_if_present(args.removal_metrics)
    if removal is None:
        rows.append(
            {
                "component": "layernorm_removal",
                "owt_eval_loss": None,
                "owt_loss_delta": None,
                "owt_eval_perplexity": None,
                "owt_perplexity_delta": None,
                "relative_owt_perplexity_increase": None,
                "wikitext_perplexity": None,
                "wikitext_perplexity_delta": None,
                "replacement_fraction": None,
                "status": "pending_A4",
            }
        )
    else:
        rows.append(
            {
                "component": "layernorm_removal",
                "owt_eval_loss": removal["post_fold_eval_loss"],
                "owt_loss_delta": removal.get("removal_loss_delta"),
                "owt_eval_perplexity": removal["post_fold_perplexity"],
                "owt_perplexity_delta": (
                    None
                    if removal.get("baseline_eval_loss") is None
                    else removal["post_fold_perplexity"]
                    - math.exp(removal["baseline_eval_loss"])
                ),
                "relative_owt_perplexity_increase": (
                    None
                    if removal.get("removal_loss_delta") is None
                    else math.exp(removal["removal_loss_delta"]) - 1.0
                ),
                "wikitext_perplexity": None,
                "wikitext_perplexity_delta": None,
                "replacement_fraction": None,
                "status": removal.get("decision", "measured"),
            }
        )

    programs = load_if_present(args.program_metrics)
    synthesis = load_if_present(args.synthesis_metrics)
    if programs is None:
        rows.append(
            {
                "component": "program_heads",
                "owt_eval_loss": None,
                "owt_loss_delta": None,
                "owt_eval_perplexity": None,
                "owt_perplexity_delta": None,
                "relative_owt_perplexity_increase": None,
                "wikitext_perplexity": None,
                "wikitext_perplexity_delta": None,
                "replacement_fraction": (
                    None if synthesis is None else synthesis.get("replacement_fraction")
                ),
                "status": "pending_C4",
            }
        )
    else:
        reference_loss = math.log(programs["reference_eval_perplexity"])
        final_perplexity = programs["final_eval_perplexity"]
        reference_perplexity = programs["reference_eval_perplexity"]
        rows.append(
            {
                "component": "program_heads",
                "owt_eval_loss": programs["final_eval_loss"],
                "owt_loss_delta": programs["final_eval_loss"] - reference_loss,
                "owt_eval_perplexity": final_perplexity,
                "owt_perplexity_delta": final_perplexity - reference_perplexity,
                "relative_owt_perplexity_increase": (
                    final_perplexity / reference_perplexity - 1.0
                ),
                "wikitext_perplexity": None,
                "wikitext_perplexity_delta": None,
                "replacement_fraction": (
                    None if synthesis is None else synthesis.get("replacement_fraction")
                ),
                "status": "accepted" if programs["success"] else "rejected",
            }
        )

    output = {
        "documented_source": "docs/SCALABILITY.md",
        "currency": {
            "primary": "OpenWebText validation loss delta",
            "secondary": "OpenWebText validation perplexity delta",
            "external_validation": "WikiText-103 validation perplexity delta",
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(output, handle, indent=2)
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
