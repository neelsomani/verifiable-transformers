#!/usr/bin/env python3
"""Synthesize restricted programs for retained GPT-2 circuit heads."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

import torch
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.extract import (
    get_candidate_token_ids,
    load_behavior_examples,
    load_model_with_variants,
)
from scripts.gpt2.behavior_domains import reference_program_targets
from scripts.programs import (
    CommandProgramProposer,
    SynthesisHarness,
    install_program_heads,
)


HEAD_PATTERN = re.compile(r"^attn_(\d+)_h_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--circuit_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_examples", type=int, default=128)
    parser.add_argument(
        "--domain_manifest",
        default=None,
        help="Locked synthesis-domain manifest; legacy repeated rows if omitted.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--healable_agreement", type=float, default=1.0)
    parser.add_argument("--device", default=None)
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
    return parser.parse_args()


def retained_heads(circuit: dict) -> set[tuple[int, int]]:
    result = set()
    for edge in circuit["edges"]:
        source = edge["source"] if isinstance(edge, dict) else edge[0]
        target = edge["target"] if isinstance(edge, dict) else edge[1]
        for node in (source, target):
            match = HEAD_PATTERN.match(node)
            if match:
                result.add((int(match.group(1)), int(match.group(2))))
    return result


def projected_decisions(
    model,
    input_ids,
    attention_mask,
    candidates,
    batch_size,
) -> torch.Tensor:
    results = []
    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start : start + batch_size]
            mask = attention_mask[start : start + batch_size]
            logits = model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
            ).logits
            last = mask.sum(dim=1) - 1
            rows = logits[torch.arange(ids.size(0), device=ids.device), last]
            results.append(rows[:, candidates].argmax(dim=-1).cpu())
    return torch.cat(results)


def collect_attention(
    model,
    input_ids,
    attention_mask,
    layer,
    head,
    batch_size,
) -> torch.Tensor:
    results = []
    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            output = model(
                input_ids=input_ids[start : start + batch_size],
                attention_mask=attention_mask[start : start + batch_size],
                output_attentions=True,
                use_cache=False,
            )
            results.append(output.attentions[layer][:, head].cpu())
    return torch.cat(results)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_with_variants(args.model_path, device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    os.makedirs(args.output_dir, exist_ok=True)
    all_programs = {}
    task_results = {}
    resistant_heads = []
    conflicting_heads = []
    blocked_heads = set()
    proposer = (
        CommandProgramProposer(
            args.lm_proposer_command,
            timeout_seconds=args.lm_proposer_timeout_seconds,
        )
        if args.lm_proposer_command
        else None
    )
    domain_provenance = {}
    base_accuracy_against_reference = {}

    for task in ("quote_close", "bracket_type"):
        with open(os.path.join(args.circuit_root, task, "circuit.json")) as handle:
            circuit = json.load(handle)
        heads = retained_heads(circuit)
        examples, task_domain = load_behavior_examples(
            task, args.num_examples, args.domain_manifest
        )
        domain_provenance[task] = task_domain
        encoded = tokenizer(
            [example.prompt for example in examples],
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        candidates = get_candidate_token_ids(task, tokenizer)
        base_decisions = projected_decisions(
            model,
            input_ids,
            attention_mask,
            candidates,
            args.batch_size,
        )
        reference = reference_program_targets(examples, tokenizer, candidates)
        base_accuracy = float((base_decisions == reference).float().mean().item())
        base_accuracy_against_reference[task] = base_accuracy
        if base_accuracy != 1.0:
            raise RuntimeError(
                f"{task} base-model accuracy against P(x) is {base_accuracy:.6f}; "
                "v2 forbids filtering examples or synthesizing against an "
                "incorrect base reference"
            )
        task_heads = {}

        for layer, head in sorted(heads):
            target = collect_attention(
                model,
                input_ids,
                attention_mask,
                layer,
                head,
                args.batch_size,
            )
            candidate_model = copy.deepcopy(model)
            # Construct physical Q/K deletion once, then swap only the frozen DSL
            # object between candidates evaluated by the harness.
            placeholder = SynthesisHarness()._base_candidates(
                input_ids, attention_mask
            )[0]
            install_program_heads(
                candidate_model,
                {(layer, head): placeholder},
                attention_variant="sparsemax",
            )
            candidate_attention = candidate_model.transformer.h[layer].attn

            def projected_evaluator(program):
                candidate_attention.programs[head] = program
                decisions = projected_decisions(
                    candidate_model,
                    input_ids,
                    attention_mask,
                    candidates,
                    args.batch_size,
                )
                return float((decisions == reference).float().mean().item())

            result = SynthesisHarness(
                healable_projected_agreement=args.healable_agreement,
                proposer=proposer,
                proposer_rounds=args.lm_proposer_rounds,
            ).synthesize(
                input_ids.cpu(),
                target,
                projected_evaluator=projected_evaluator,
                attention_mask=attention_mask.cpu(),
            )
            key = f"{layer}.{head}"
            program_dict = result.program.to_dict()
            if not result.accepted:
                resistant_heads.append({"task": task, "head": key})
                all_programs.pop(key, None)
                blocked_heads.add(key)
            elif key not in blocked_heads and key not in all_programs:
                all_programs[key] = program_dict
            elif key in all_programs and all_programs[key] != program_dict:
                # A physical head can host only one frozen program. Do not
                # silently choose one task's synthesis over the other.
                conflicting_heads.append(
                    {
                        "head": key,
                        "task": task,
                        "reason": "task domains synthesized different programs",
                    }
                )
                all_programs.pop(key, None)
                blocked_heads.add(key)
            report = result.to_dict()
            valid_entries = (
                attention_mask.bool().cpu().unsqueeze(-1)
                & attention_mask.bool().cpu().unsqueeze(1)
            )
            report["target_zero_fraction"] = float(
                (target[valid_entries] == 0).float().mean().item()
            )
            task_heads[key] = report
            print(
                f"{task} head {key}: accepted={result.accepted}, "
                f"IoU={result.score.support_iou:.4f}, "
                f"projected={result.score.projected_agreement:.4f}"
            )
        task_results[task] = task_heads

    output = {
        "model_path": args.model_path,
        "circuit_root": args.circuit_root,
        "num_examples": {
            task: domain_provenance[task]["rows"] for task in domain_provenance
        },
        "domain_manifest": args.domain_manifest,
        "domain": domain_provenance,
        "reference_target": "explicit_reference_program_P(x)",
        "base_accuracy_against_reference": base_accuracy_against_reference,
        "acceptance_metric": "exact_projected_agreement",
        "healable_agreement": args.healable_agreement,
        "lm_proposer": proposer.provenance() if proposer is not None else None,
        "lm_proposer_rounds": args.lm_proposer_rounds,
        "programs": all_programs,
        "replacement_fraction": len(all_programs) / model.config.n_head / model.config.n_layer,
        "success": bool(all_programs),
        "resistant_heads": resistant_heads,
        "conflicting_heads": conflicting_heads,
        "tasks": task_results,
    }
    with open(os.path.join(args.output_dir, "synthesis_results.json"), "w") as handle:
        json.dump(output, handle, indent=2)
    with open(os.path.join(args.output_dir, "programs.json"), "w") as handle:
        json.dump(all_programs, handle, indent=2)
    if not all_programs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
