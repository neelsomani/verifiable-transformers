#!/usr/bin/env python3
"""Apply the preregistered A4 gate and materialize the Phase-C base choice."""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--removal_metrics", required=True)
    parser.add_argument("--norm_free_model", required=True)
    parser.add_argument("--bandnorm_model", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def valid_checkpoint(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, "config.json"))
        and (
            os.path.isfile(os.path.join(path, "model.safetensors"))
            or os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        )
        and os.path.isfile(os.path.join(path, "model_info.json"))
    )


def select(metrics: dict, norm_free_model: str, bandnorm_model: str) -> dict:
    if metrics.get("status") != "passed":
        raise RuntimeError("A4 removal did not complete successfully")
    delta = metrics.get("removal_loss_delta")
    gate = metrics.get("bandnorm_loss_delta_gate")
    if delta is None or gate is None:
        raise RuntimeError("A4 decision requires measured removal delta and gate")
    expected = "norm_free" if float(delta) < float(gate) else "bandnorm"
    if metrics.get("decision") != expected:
        raise RuntimeError(
            f"Removal artifact decision {metrics.get('decision')!r} contradicts "
            f"the preregistered rule, which yields {expected!r}"
        )
    selected = norm_free_model if expected == "norm_free" else bandnorm_model
    if not valid_checkpoint(selected):
        raise FileNotFoundError(
            f"Selected {expected} checkpoint is incomplete: {selected}"
        )
    return {
        "decision": expected,
        "selected_model": os.path.abspath(selected),
        "removal_loss_delta": float(delta),
        "bandnorm_loss_delta_gate": float(gate),
        "decision_rule": "norm_free iff removal_loss_delta < bandnorm_loss_delta_gate",
        "removal_metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    with open(args.removal_metrics) as handle:
        metrics = json.load(handle)
    output = select(metrics, args.norm_free_model, args.bandnorm_model)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
