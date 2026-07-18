#!/usr/bin/env python3
"""Synthesize restricted programs for retained small-model circuit heads."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.programs import (
    CommandProgramProposer,
    SynthesisHarness,
    install_program_heads,
)
from scripts.small import get_eval_dataset, vocab
from scripts.small.config import SmallVerifiableConfig
from scripts.small.extract import load_model
from scripts.small.train import create_small_model


HEAD_PATTERN = re.compile(r"^attn_(\d+)_h_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--circuit_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--healable_agreement", type=float, default=1.0)
    parser.add_argument(
        "--lm_proposer_command",
        default=None,
        help=(
            "Optional Hayes-style LM command. The restricted prompt is sent "
            "on stdin and the command must emit program JSON on stdout."
        ),
    )
    parser.add_argument("--lm_proposer_rounds", type=int, default=2)
    parser.add_argument("--lm_proposer_timeout_seconds", type=float, default=120.0)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Restrict synthesis to <task>:<layer>.<head>; may be repeated.",
    )
    return parser.parse_args()


def retained_heads(circuit: dict) -> set[tuple[int, int]]:
    heads = set()
    for edge in circuit["edges"]:
        source = edge["source"] if isinstance(edge, dict) else edge[0]
        target = edge["target"] if isinstance(edge, dict) else edge[1]
        for node in (source, target):
            match = HEAD_PATTERN.match(node)
            if match:
                heads.add((int(match.group(1)), int(match.group(2))))
    return heads


def counterfactual_softmax_model(
    checkpoint: str, config: SmallVerifiableConfig
):
    softmax_config = copy.deepcopy(config)
    softmax_config.attn_variant = "softmax"
    model = create_small_model(softmax_config)
    state = load_file(os.path.join(checkpoint, "model.safetensors"))
    model.load_state_dict(state)
    return model.eval()


def parse_targets(values: list[str]) -> dict[str, set[tuple[int, int]]]:
    targets: dict[str, set[tuple[int, int]]] = {}
    for value in values:
        task, raw_head = value.split(":", 1)
        if task not in {"quote_close", "bracket_type"}:
            raise ValueError(f"Unknown task in --target: {task}")
        layer, head = (int(part) for part in raw_head.split("."))
        targets.setdefault(task, set()).add((layer, head))
    return targets


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = SmallVerifiableConfig.load(
        os.path.join(os.path.dirname(args.checkpoint), "config.json")
    )
    base = load_model(args.checkpoint, config, torch.device("cpu"))
    existing_path = os.path.join(args.checkpoint, "programs.json")
    if os.path.exists(existing_path):
        with open(existing_path) as handle:
            all_programs = json.load(handle)
        softmax_model = None
        counterfactual_unavailable_reason = (
            "checkpoint already contains program heads, so the raw neural "
            "softmax counterfactual is not architecture-compatible"
        )
    else:
        all_programs = {}
        softmax_model = counterfactual_softmax_model(args.checkpoint, config)
        counterfactual_unavailable_reason = None
    requested_targets = parse_targets(args.target)
    task_results = {}
    proposer = (
        CommandProgramProposer(
            args.lm_proposer_command,
            timeout_seconds=args.lm_proposer_timeout_seconds,
        )
        if args.lm_proposer_command
        else None
    )

    for task in ("quote_close", "bracket_type"):
        if requested_targets and task not in requested_targets:
            continue
        circuit_path = os.path.join(args.circuit_root, task, "circuit.json")
        with open(circuit_path) as handle:
            circuit = json.load(handle)
        retained = retained_heads(circuit)
        heads = requested_targets.get(task, retained)
        missing = heads - retained
        if missing:
            raise RuntimeError(
                f"Requested heads are absent from {task} circuit: {sorted(missing)}"
            )
        examples = get_eval_dataset(task)
        input_ids = torch.tensor([example["input_ids"] for example in examples])
        candidates = sorted(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task]))
        with torch.no_grad():
            base_output = base(input_ids, output_attentions=True)
            softmax_output = (
                softmax_model(input_ids, output_attentions=True)
                if softmax_model is not None
                else None
            )
            full_decision = base_output.logits[:, -1, candidates].argmax(dim=-1)

        head_results = {}
        for layer, head in sorted(heads):
            target = base_output.attentions[layer][:, head]

            def projected_evaluator(program):
                candidate_model = copy.deepcopy(base)
                install_program_heads(
                    candidate_model,
                    {(layer, head): program},
                    attention_variant="sparsemax",
                )
                candidate_model.eval()
                with torch.no_grad():
                    decision = candidate_model(input_ids).logits[:, -1, candidates].argmax(dim=-1)
                return float((decision == full_decision).float().mean().item())

            harness = SynthesisHarness(
                healable_projected_agreement=args.healable_agreement,
                proposer=proposer,
                proposer_rounds=args.lm_proposer_rounds,
            )
            result = harness.synthesize(
                input_ids,
                target,
                projected_evaluator=projected_evaluator,
            )
            key = f"{layer}.{head}"
            if result.accepted:
                all_programs[key] = result.program.to_dict()
            softmax_attention = (
                softmax_output.attentions[layer][:, head]
                if softmax_output is not None
                else None
            )
            report = result.to_dict()
            report.update(
                {
                    "target_zero_fraction_sparsemax": float((target == 0).float().mean().item()),
                    "target_zero_fraction_softmax_counterfactual": (
                        float((softmax_attention == 0).float().mean().item())
                        if softmax_attention is not None
                        else None
                    ),
                    "mean_support_size_sparsemax": float(
                        (target > 1e-7).sum(dim=-1).float().mean().item()
                    ),
                    "mean_support_size_softmax_counterfactual": (
                        float(
                            (softmax_attention > 1e-7)
                            .sum(dim=-1)
                            .float()
                            .mean()
                            .item()
                        )
                        if softmax_attention is not None
                        else None
                    ),
                    "softmax_counterfactual_unavailable_reason": (
                        counterfactual_unavailable_reason
                        if softmax_attention is None
                        else None
                    ),
                }
            )
            head_results[key] = report
            print(
                f"{task} head {key}: accepted={result.accepted}, "
                f"IoU={result.score.support_iou:.4f}, "
                f"projected={result.score.projected_agreement:.4f}"
            )
        task_results[task] = head_results

    output = {
        "checkpoint": args.checkpoint,
        "circuit_root": args.circuit_root,
        "acceptance_metric": "projected_agreement",
        "healable_agreement": args.healable_agreement,
        "lm_proposer": proposer.provenance() if proposer is not None else None,
        "lm_proposer_rounds": args.lm_proposer_rounds,
        "requested_targets": args.target,
        "existing_programs_preserved": sorted(
            key for key in all_programs if key not in {
                head for result in task_results.values() for head in result
                if result[head]["accepted"]
            }
        ),
        "programs": all_programs,
        "tasks": task_results,
    }
    with open(os.path.join(args.output_dir, "synthesis_results.json"), "w") as handle:
        json.dump(output, handle, indent=2)
    with open(os.path.join(args.output_dir, "programs.json"), "w") as handle:
        json.dump(all_programs, handle, indent=2)


if __name__ == "__main__":
    main()
