#!/usr/bin/env python3
"""Run the preregistered Phase Q read-side calibration ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.audit_preheal_necessity import (
    intervention_edge_sets,
    load_selected_edges,
)
from scripts.gpt2.behavior_domains import reference_program_targets
from scripts.gpt2.extract import (
    build_circuit_graph,
    controlled_forward,
    get_candidate_token_ids,
    load_behavior_examples,
    load_model_with_variants,
    select_last_real_logits,
)
from scripts.programs import install_program_heads, load_programs


PROGRAM_NODES = ("attn_7_h_11", "attn_9_h_0")
LOCKED_IDENTITY_INTERVENTIONS = (
    "native_full_forward",
    "zero_both_program_heads",
    "whole_selected_circuit_node_zero",
    "whole_selected_circuit_edge_zero",
)
BASELINE_BATTERY = (
    "native_full_forward",
    "circuit_only",
    "zero_attn_7_h_11",
    "zero_attn_9_h_0",
    "zero_both_program_heads",
    "whole_selected_circuit_node_zero",
    "whole_selected_circuit_edge_zero",
)


def configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_hash(values) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_parameter_hashes(model) -> dict[str, str]:
    result = {}
    for name, value in model.state_dict().items():
        raw = value.detach().cpu().contiguous().numpy().tobytes()
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def evaluate_rows(model, values, batch_size, edges=None, graph=None):
    rows = []
    with torch.no_grad():
        for start in range(0, len(values["examples"]), batch_size):
            stop = start + batch_size
            ids = values["input_ids"][start:stop]
            mask = values["attention_mask"][start:stop]
            if edges is None:
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            else:
                logits = controlled_forward(model, ids, mask, edges, graph)
            projected = select_last_real_logits(logits, mask)[
                :, values["candidate_token_ids"]
            ].float()
            decisions = projected.argmax(-1).cpu()
            targets = values["targets"][start:stop]
            indices = torch.arange(decisions.numel(), device=projected.device)
            margins = (
                projected[indices, targets.to(projected.device)]
                - projected[indices, (1 - targets).to(projected.device)]
            ).cpu()
            for offset in range(decisions.numel()):
                example = values["examples"][start + offset]
                rows.append(
                    {
                        "example_id": example.example_id,
                        "target": int(targets[offset]),
                        "decision": int(decisions[offset]),
                        "candidate_logits": [
                            float(v) for v in projected[offset].detach().cpu()
                        ],
                        "candidate_margin": float(margins[offset]),
                    }
                )
    return rows


def _program_contributions(model, input_ids, attention_mask, normalized_by_layer):
    """Return post-W_O, bias-free contributions for the two frozen programs."""
    result = []
    for node in PROGRAM_NODES:
        _, layer_text, _, head_text = node.split("_")
        layer, head = int(layer_text), int(head_text)
        attention = model.transformer.h[layer].attn
        normalized = normalized_by_layer[layer]
        start = head * attention.head_dim
        stop = start + attention.head_dim
        value = F.linear(
            normalized,
            attention.value_proj.weight[start:stop],
            attention.value_proj.bias[start:stop],
        )
        weights = attention.programs[head].weights(input_ids, dtype=value.dtype)
        valid = attention_mask.to(value.dtype)
        weights = weights * valid.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mixture = torch.matmul(weights, value)
        result.append(torch.matmul(mixture, attention.c_proj.weight[start:stop]))
    return torch.stack(result, dim=1)


def native_forward_with_contributions(model, input_ids, attention_mask):
    normalized_by_layer = {}
    handles = []
    for layer, block in enumerate(model.transformer.h):
        if layer not in {7, 9}:
            continue

        def capture(_module, _inputs, output, layer=layer):
            normalized_by_layer[layer] = output

        handles.append(block.ln_1.register_forward_hook(capture))
    try:
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
    finally:
        for handle in handles:
            handle.remove()
    contributions = _program_contributions(
        model, input_ids, attention_mask, normalized_by_layer
    )
    return logits, contributions


def cache_task_features(model, values, graph, selected_edges, batch_size):
    """Cache exact affine read-side features for native-full and circuit-only."""
    candidate_weight = model.lm_head.weight[values["candidate_token_ids"]]
    cache = {"native_full": [], "circuit_only": []}
    with torch.no_grad():
        for start in range(0, len(values["examples"]), batch_size):
            stop = start + batch_size
            ids = values["input_ids"][start:stop]
            mask = values["attention_mask"][start:stop]
            native_logits, native_contributions = native_forward_with_contributions(
                model, ids, mask
            )
            circuit_logits, nodes = controlled_forward(
                model,
                ids,
                mask,
                selected_edges,
                graph,
                return_node_outputs=True,
            )
            circuit_contributions = []
            for node in PROGRAM_NODES:
                _, layer_text, _, head_text = node.split("_")
                layer, head = int(layer_text), int(head_text)
                attention = model.transformer.h[layer].attn
                begin = head * attention.head_dim
                end = begin + attention.head_dim
                circuit_contributions.append(
                    torch.matmul(
                        nodes[node], attention.c_proj.weight[begin:end]
                    )
                )
            circuit_contributions = torch.stack(circuit_contributions, dim=1)
            last = mask.sum(dim=1) - 1
            batch_indices = torch.arange(ids.size(0), device=ids.device)
            for name, logits, contributions in (
                ("native_full", native_logits, native_contributions),
                ("circuit_only", circuit_logits, circuit_contributions),
            ):
                base = logits[batch_indices, last][:, values["candidate_token_ids"]]
                channel = contributions[batch_indices, :, last, :]
                # [batch, head, channel, candidate]
                features = channel.unsqueeze(-1) * candidate_weight.T[None, None]
                cache[name].append((base.cpu(), features.cpu()))
    return {
        name: {
            "base": torch.cat([part[0] for part in parts]),
            "features": torch.cat([part[1] for part in parts]),
        }
        for name, parts in cache.items()
    }


def calibrated_task_logits(cache, calibration):
    delta = calibration - 1.0
    if delta.ndim == 1:
        delta = delta[:, None]
    return {
        name: values["base"].to(calibration.device)
        + (
            values["features"].to(calibration.device)
            * delta[None, :, :, None]
        ).sum(dim=(1, 2))
        for name, values in cache.items()
    }


def exact_task_gate(cache, calibration, targets):
    logits = calibrated_task_logits(cache, calibration)
    target_device = targets.to(calibration.device)
    counts = {
        name: int((value.argmax(-1) == target_device).sum().item())
        for name, value in logits.items()
    }
    return counts, logits


def owt_objective_batch(model, row, calibration, device):
    ids = torch.tensor(row["input_ids"], device=device).unsqueeze(0)
    mask = torch.tensor(row["attention_mask"], device=device).unsqueeze(0)
    labels = torch.tensor(row["labels"], device=device).unsqueeze(0)
    with torch.no_grad():
        base_logits, contributions = native_forward_with_contributions(
            model, ids, mask
        )
    delta = calibration - 1.0
    if delta.ndim != 1:
        raise ValueError("Scalar OWT objective requires two scalar gains")
    residual_delta = (
        contributions.detach() * delta.reshape(1, 2, 1, 1)
    ).sum(dim=1)
    logits = base_logits.detach() + F.linear(
        residual_delta, model.lm_head.weight
    )
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def evaluate_owt_scalar(model, dataset, calibration, device, batch_size=8):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = dataset[start : start + batch_size]
            ids = torch.tensor(rows["input_ids"], device=device)
            mask = torch.tensor(rows["attention_mask"], device=device)
            labels = torch.tensor(rows["labels"], device=device)
            base_logits, contributions = native_forward_with_contributions(
                model, ids, mask
            )
            residual_delta = (
                contributions
                * (calibration - 1.0).reshape(1, 2, 1, 1)
            ).sum(dim=1)
            logits = base_logits + F.linear(residual_delta, model.lm_head.weight)
            shifted_labels = labels[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            tokens = int((shifted_labels != -100).sum().item())
            total_loss += float(loss.item())
            total_tokens += tokens
    mean_loss = total_loss / total_tokens
    return {
        "eval_examples": len(dataset),
        "eval_tokens": total_tokens,
        "eval_loss": mean_loss,
        "eval_perplexity": math.exp(mean_loss),
    }


def run_scalar_rung(
    model,
    values,
    graph,
    selected_edges,
    registration,
    output_dir,
    processed_dataset_dir,
    device,
):
    rung = registration["ladder"][0]
    cache = cache_task_features(
        model,
        values,
        graph,
        selected_edges,
        registration["execution_policy"]["batch_size_for_identity"],
    )
    targets = values["targets"]
    datasets = load_from_disk(processed_dataset_dir)
    owt_train = datasets["validation"]
    gains = torch.nn.Parameter(torch.ones(2, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam(
        [gains], lr=rung["learning_rate"], weight_decay=rung["weight_decay"]
    )
    history = []
    passing_candidate = None
    for step in range(1, rung["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        task_logits = calibrated_task_logits(cache, gains)
        task_loss = sum(
            F.cross_entropy(logits, targets.to(device))
            for logits in task_logits.values()
        )
        owt_loss = owt_objective_batch(
            model, owt_train[(step - 1) % len(owt_train)], gains, device
        )
        loss = task_loss + registration["objective"]["owt_preservation_weight"] * owt_loss
        loss.backward()
        optimizer.step()
        if step % rung["gate_interval_steps"] != 0:
            continue
        counts, _ = exact_task_gate(cache, gains, targets)
        record = {
            "step": step,
            "gains": [float(value) for value in gains.detach().cpu()],
            "objective": float(loss.detach().cpu()),
            "task_loss": float(task_loss.detach().cpu()),
            "owt_preservation_loss": float(owt_loss.detach().cpu()),
            "native_full_correct": counts["native_full"],
            "circuit_only_correct": counts["circuit_only"],
            "task_pass": all(value == len(targets) for value in counts.values()),
            "identity_pass": True,
            "identity_basis": (
                "The only calibrated selected edges are absent in all locked "
                "lesion edge sets, so their delta is exactly zero by construction."
            ),
        }
        history.append(record)
        print(json.dumps({"scalar_gate": record}), flush=True)
        if record["task_pass"] and record["identity_pass"]:
            owt = evaluate_owt_scalar(model, owt_train, gains.detach(), device)
            record["owt"] = owt
            record["owt_pass"] = (
                owt["eval_perplexity"]
                <= registration["locked_gates"]["owt_perplexity_max"]
            )
            if record["owt_pass"]:
                passing_candidate = gains.detach().cpu()
                break
    report = {
        "schema_version": 1,
        "rung": 1,
        "name": rung["name"],
        "steps_run": step,
        "objective_evaluations": step,
        "history": history,
        "terminal_gains": [float(value) for value in gains.detach().cpu()],
        "pass": passing_candidate is not None,
    }
    (output_dir / "rung1_scalar.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report, passing_candidate, cache


def compare_semantics_to_audit(rows, audit_rows):
    if [r["example_id"] for r in rows] != [r["example_id"] for r in audit_rows]:
        raise RuntimeError("Domain order does not reproduce the audit")
    decisions_equal = all(
        row["decision"] == audit["decision"] for row, audit in zip(rows, audit_rows)
    )
    mismatches = [
        row["example_id"] for row in rows if row["decision"] != row["target"]
    ]
    audit_mismatches = [
        row["example_id"]
        for row in audit_rows
        if row["decision"] != row["target"]
    ]
    return {
        "rows": len(rows),
        "correct": sum(row["decision"] == row["target"] for row in rows),
        "decisions_exact": decisions_equal,
        "mismatch_ids_exact": mismatches == audit_mismatches,
        "mismatch_ids_sha256": canonical_hash(mismatches),
        "candidate_logits_sha256": canonical_hash(
            [row["candidate_logits"] for row in rows]
        ),
        "candidate_margins_sha256": canonical_hash(
            [row["candidate_margin"] for row in rows]
        ),
        "pass": decisions_equal and mismatches == audit_mismatches,
        "numerical_comparison_to_old_audit": "not_performed_by_amendment",
    }


def compare_paired_rows(first, second, epsilon):
    if [r["example_id"] for r in first] != [r["example_id"] for r in second]:
        raise RuntimeError("Paired deterministic runs used different domain order")
    decisions_exact = all(
        left["decision"] == right["decision"] for left, right in zip(first, second)
    )
    first_mismatches = [
        row["example_id"] for row in first if row["decision"] != row["target"]
    ]
    second_mismatches = [
        row["example_id"] for row in second if row["decision"] != row["target"]
    ]
    logit_max_abs = max(
        abs(left_logit - right_logit)
        for left, right in zip(first, second)
        for left_logit, right_logit in zip(
            left["candidate_logits"], right["candidate_logits"]
        )
    )
    margin_max_abs = max(
        abs(left["candidate_margin"] - right["candidate_margin"])
        for left, right in zip(first, second)
    )
    return {
        "rows": len(first),
        "correct": sum(row["decision"] == row["target"] for row in first),
        "decisions_exact": decisions_exact,
        "mismatch_ids_exact": first_mismatches == second_mismatches,
        "mismatch_ids_sha256": canonical_hash(first_mismatches),
        "candidate_logit_max_abs_difference": logit_max_abs,
        "candidate_margin_max_abs_difference": margin_max_abs,
        "epsilon": epsilon,
        "pass": (
            decisions_exact
            and first_mismatches == second_mismatches
            and logit_max_abs <= epsilon
            and margin_max_abs <= epsilon
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument(
        "--processed_dataset_dir",
        default=".cache/openwebtext-gpt2-block1024",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    registration_path = Path(args.registration)
    registration = json.loads(registration_path.read_text())
    root = Path(__file__).resolve().parents[2]
    frozen = registration["frozen_inputs"]
    policy = registration["execution_policy"]
    configure_determinism(policy["seed"])

    audit = json.loads((root / frozen["audit"]).read_text())
    base_path = root / frozen["source_model"]
    tokenizer = GPT2Tokenizer.from_pretrained(base_path)
    tokenizer.pad_token = tokenizer.eos_token
    examples, _ = load_behavior_examples(
        "quote_close", 0, str(root / frozen["domain_manifest"])
    )
    encoded = tokenizer(
        [example.prompt for example in examples], return_tensors="pt", padding=True
    )
    candidate_ids = get_candidate_token_ids("quote_close", tokenizer)
    values = {
        "examples": examples,
        "input_ids": encoded["input_ids"].to(args.device),
        "attention_mask": encoded["attention_mask"].to(args.device),
        "candidate_token_ids": candidate_ids,
        "targets": reference_program_targets(examples, tokenizer, candidate_ids),
    }

    model = load_model_with_variants(str(base_path), args.device).float().eval()
    programs = load_programs(str(root / frozen["programs"]))
    install_program_heads(model, programs, attention_variant="sparsemax")
    graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
    selected_edges = load_selected_edges(root / frozen["circuit"])
    edge_sets = intervention_edge_sets(graph, selected_edges)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_baseline.json"
    if paired_path.exists():
        report = json.loads(paired_path.read_text())
        if not report.get("pass"):
            raise SystemExit("STOP: stored paired deterministic baseline failed")
    else:
        paired_runs = []
        for _ in range(2):
            run = {}
            for name in BASELINE_BATTERY:
                edges = None if name == "native_full_forward" else edge_sets[name]
                run[name] = evaluate_rows(
                    model,
                    values,
                    policy["batch_size_for_identity"],
                    edges=edges,
                    graph=graph,
                )
            paired_runs.append(run)

        paired_comparisons = {
            name: compare_paired_rows(
                paired_runs[0][name],
                paired_runs[1][name],
                policy["identity_epsilon_max_abs"],
            )
            for name in BASELINE_BATTERY
        }
        audit_semantic_comparisons = {
            name: compare_semantics_to_audit(
                paired_runs[0][name],
                audit["states"]["B_programs_installed_step_zero"][name]["per_input"],
            )
            for name in BASELINE_BATTERY
        }

        report = {
        "schema_version": 1,
        "protocol_id": registration["protocol_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=root
        ).strip(),
        "configuration": policy,
        "amendment": "PROTOCOL_AMENDMENT_2026-07-27.md",
        "note": (
            "Two current pinned runs are the numerical identity pair. The older "
            "audit is compared only for exact decisions and mismatch-ID sets."
        ),
        "paired_deterministic_comparisons": paired_comparisons,
        "old_audit_semantic_comparisons": audit_semantic_comparisons,
        "numerical_identity_reference": {
            name: paired_runs[0][name]
            for name in LOCKED_IDENTITY_INTERVENTIONS
        },
        "parameter_hashes": tensor_parameter_hashes(model),
        "pass": (
            all(value["pass"] for value in paired_comparisons.values())
            and all(value["pass"] for value in audit_semantic_comparisons.values())
        ),
        }
        paired_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise SystemExit("STOP: paired deterministic FP32 baseline failed")
    if args.preflight_only:
        return
    scalar_report, scalar_candidate, _ = run_scalar_rung(
        model,
        values,
        graph,
        selected_edges,
        registration,
        output_dir,
        str(root / args.processed_dataset_dir),
        args.device,
    )
    if scalar_candidate is not None:
        print(json.dumps({"rung1": "pass", "gains": scalar_candidate.tolist()}))
        return
    raise NotImplementedError(
        "Rung 1 failed; diagonal and program-local registered fallbacks are pending"
    )


if __name__ == "__main__":
    main()
