#!/usr/bin/env python3
"""Continue Phase Q under the registered constrained-calibration amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from safetensors.torch import load_file
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.audit_preheal_necessity import (
    internal_circuit_nodes,
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
from scripts.gpt2.readside_calibration import (
    canonical_hash,
    configure_determinism,
    tensor_parameter_hashes,
)
from scripts.programs import (
    hard_prune_attention_heads,
    install_program_heads,
    load_programs,
)


PROGRAM_711 = (7, 11)
PROGRAM_90 = (9, 0)
PROGRAM_NODES = ("attn_7_h_11", "attn_9_h_0")
OWT_BUDGET = 28.617593822841776


def _load_programmed_source(root: Path, registration: dict, device: str):
    frozen = registration["frozen_inputs"]
    source = root / frozen["source_model"]
    model = load_model_with_variants(str(source), device).float().eval()
    programs = load_programs(str(root / frozen["programs"]))
    if sorted(programs) != [PROGRAM_711, PROGRAM_90]:
        raise RuntimeError(f"Unexpected program set: {sorted(programs)}")
    install_program_heads(model, programs, attention_variant="sparsemax")
    return model, programs


def _copy_program_local_slice(target, source_state, layer: int, head: int) -> list[str]:
    attention = target.transformer.h[layer].attn
    start = head * attention.head_dim
    stop = start + attention.head_dim
    prefix = f"transformer.h.{layer}.attn."
    copied = []
    with torch.no_grad():
        for suffix, parameter, axis in (
            ("value_proj.weight", attention.value_proj.weight, 0),
            ("value_proj.bias", attention.value_proj.bias, 0),
            ("c_proj.weight", attention.c_proj.weight, 0),
        ):
            name = prefix + suffix
            index = [slice(None)] * parameter.ndim
            index[axis] = slice(start, stop)
            parameter[tuple(index)].copy_(
                source_state[name][tuple(index)].to(parameter.device)
            )
            copied.append(name)
    return copied


def load_rung3_candidate(
    root: Path,
    registration: dict,
    device: str,
    *,
    lean: bool,
):
    model, programs = _load_programmed_source(root, registration, device)
    checkpoint_path = (
        root
        / "artifacts/gpt2-phase-q-readside-calibration/rung3_checkpoint/model.safetensors"
    )
    rung3 = load_file(str(checkpoint_path), device="cpu")
    if lean:
        copied = _copy_program_local_slice(model, rung3, *PROGRAM_711)
        hard_prune_attention_heads(model, {PROGRAM_90})
    else:
        incompatible = model.load_state_dict(rung3, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Rung-3 checkpoint architecture mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        copied = [
            f"transformer.h.{layer}.attn.{suffix}"
            for layer, _head in (PROGRAM_711, PROGRAM_90)
            for suffix in ("value_proj.weight", "value_proj.bias", "c_proj.weight")
        ]
    return model.eval(), programs, rung3, copied


def _domain_values(root: Path, registration: dict, tokenizer, device: str):
    frozen = registration["frozen_inputs"]
    examples, provenance = load_behavior_examples(
        "quote_close", 0, str(root / frozen["domain_manifest"])
    )
    encoded = tokenizer(
        [example.prompt for example in examples],
        return_tensors="pt",
        padding=True,
    )
    candidates = get_candidate_token_ids("quote_close", tokenizer)
    return {
        "examples": examples,
        "provenance": provenance,
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "candidate_token_ids": candidates,
        "targets": reference_program_targets(examples, tokenizer, candidates),
    }


def evaluate_path(model, values, batch_size: int, *, edges=None, graph=None):
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
            targets = values["targets"][start:stop].to(projected.device)
            decisions = projected.argmax(-1)
            index = torch.arange(len(decisions), device=projected.device)
            margins = projected[index, targets] - projected[index, 1 - targets]
            for offset in range(len(decisions)):
                rows.append(
                    {
                        "example_id": values["examples"][start + offset].example_id,
                        "target": int(targets[offset]),
                        "decision": int(decisions[offset]),
                        "margin": float(margins[offset]),
                        "candidate_logits": [
                            float(value) for value in projected[offset].cpu()
                        ],
                    }
                )
    correct = sum(row["target"] == row["decision"] for row in rows)
    return {
        "rows": len(rows),
        "correct": correct,
        "mismatch_ids": [
            row["example_id"] for row in rows if row["target"] != row["decision"]
        ],
        "decisions_sha256": canonical_hash([row["decision"] for row in rows]),
        "candidate_logits_sha256": canonical_hash(
            [row["candidate_logits"] for row in rows]
        ),
        "margins_sha256": canonical_hash([row["margin"] for row in rows]),
        "minimum_margin": min(row["margin"] for row in rows),
        "mean_margin": sum(row["margin"] for row in rows) / len(rows),
    }


def evaluate_owt(model, dataset, device: str, batch_size: int = 8):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = dataset[start : start + batch_size]
            ids = torch.tensor(rows["input_ids"], device=device)
            mask = torch.tensor(rows["attention_mask"], device=device)
            labels = torch.tensor(rows["labels"], device=device)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            shifted = labels[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                shifted.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss)
            total_tokens += int((shifted != -100).sum())
    mean_loss = total_loss / total_tokens
    return {
        "eval_examples": len(dataset),
        "eval_tokens": total_tokens,
        "eval_loss": mean_loss,
        "eval_perplexity": math.exp(mean_loss),
    }


def _integrity_report(model, baseline_hashes: dict, rung3: dict, copied: list[str]):
    hashes = tensor_parameter_hashes(model)
    allowed = {}
    for layer, head in (PROGRAM_711,):
        attention = model.transformer.h[layer].attn
        start, stop = head * attention.head_dim, (head + 1) * attention.head_dim
        for suffix, parameter in (
            ("value_proj.weight", attention.value_proj.weight),
            ("value_proj.bias", attention.value_proj.bias),
            ("c_proj.weight", attention.c_proj.weight),
        ):
            name = f"transformer.h.{layer}.attn.{suffix}"
            allowed[name] = {
                "slice": [start, stop],
                "rung3_tensor_sha256": hashlib.sha256(
                    rung3[name].contiguous().numpy().tobytes()
                ).hexdigest(),
                "candidate_tensor_sha256": hashes[name],
            }
    changed = sorted(
        name
        for name, digest in hashes.items()
        if name in baseline_hashes and digest != baseline_hashes[name]
    )
    unexpected = sorted(set(changed) - set(copied))
    return {
        "state_dict_hashes_sha256": canonical_hash(hashes),
        "changed_tensor_names_against_programmed_source": changed,
        "permitted_tensor_names": sorted(set(copied)),
        "unexpected_changed_tensor_names": unexpected,
        "non_permitted_integrity_pass": not unexpected,
        "hard_pruning_mask": {"9": [0]},
        "program_local_slices": allowed,
    }


def run_lean_selection(args) -> None:
    root = Path(__file__).resolve().parents[2]
    registration = json.loads(Path(args.registration).read_text())
    configure_determinism(registration["execution_policy"]["seed"])
    frozen = registration["frozen_inputs"]
    source_path = root / frozen["source_model"]
    tokenizer = GPT2Tokenizer.from_pretrained(source_path)
    tokenizer.pad_token = tokenizer.eos_token
    values = _domain_values(root, registration, tokenizer, args.device)
    graph = build_circuit_graph(12, 12)
    original_edges = load_selected_edges(root / frozen["circuit"])
    lean_edges = {
        edge for edge in original_edges if "attn_9_h_0" not in edge
    }
    full_edges = {
        edge for edge in graph.all_edges if edge[0] != "attn_9_h_0"
    }
    full_without_711 = {
        edge for edge in full_edges if edge[0] != "attn_7_h_11"
    }
    internal = internal_circuit_nodes(lean_edges)
    whole_circuit_zero = {
        edge for edge in full_edges if edge[0] not in internal
    }

    source, _ = _load_programmed_source(root, registration, args.device)
    baseline_hashes = tensor_parameter_hashes(source)
    del source
    model, programs, rung3, copied = load_rung3_candidate(
        root, registration, args.device, lean=True
    )
    integrity = _integrity_report(model, baseline_hashes, rung3, copied)
    batch_size = registration["execution_policy"]["batch_size_for_identity"]
    paths = {
        "full": evaluate_path(model, values, batch_size),
        "controlled_full_pruned": evaluate_path(
            model, values, batch_size, edges=full_edges, graph=graph
        ),
        "circuit_only": evaluate_path(
            model, values, batch_size, edges=lean_edges, graph=graph
        ),
        "full_zero_attn_7_h_11": evaluate_path(
            model, values, batch_size, edges=full_without_711, graph=graph
        ),
        "whole_selected_circuit_node_zero": evaluate_path(
            model, values, batch_size, edges=whole_circuit_zero, graph=graph
        ),
    }
    owt = evaluate_owt(
        model,
        load_from_disk(root / args.processed_dataset_dir)["validation"],
        args.device,
    )
    integrity["program_keys_in_memory"] = [
        f"{layer}.{head}" for layer, head in sorted(programs)
    ]
    integrity["active_program_heads"] = ["7.11"]
    integrity["hard_pruned_program_heads"] = ["9.0"]
    integrity["pruned_head_has_no_neural_fallback"] = (
        0 not in model.transformer.h[9].attn.neural_heads
        and 0 in model.transformer.h[9].attn.hard_pruned_heads
    )
    gates = {
        "full_exact": paths["full"]["correct"] == 1280,
        "controlled_full_exact": paths["controlled_full_pruned"]["correct"] == 1280,
        "circuit_only_exact": paths["circuit_only"]["correct"] == 1280,
        "attn_7_h_11_necessary": paths["full_zero_attn_7_h_11"]["correct"] < 1280,
        "whole_circuit_necessary": (
            paths["whole_selected_circuit_node_zero"]["correct"] < 1280
        ),
        "owt_within_budget": owt["eval_perplexity"] <= OWT_BUDGET,
        "integrity": (
            integrity["non_permitted_integrity_pass"]
            and integrity["pruned_head_has_no_neural_fallback"]
        ),
    }
    selected = "lean_single_program" if all(gates.values()) else "two_program_rung3"
    report = {
        "schema_version": 1,
        "protocol_amendment": (
            "artifacts/gpt2-phase-q-readside-calibration/"
            "CAUSAL_CRITERION_AMENDMENT_2026-07-27.md"
        ),
        "candidate": "lean_single_program",
        "domain": values["provenance"],
        "original_circuit_edge_count": len(original_edges),
        "lean_circuit_edge_count": len(lean_edges),
        "lean_circuit_edges": sorted(map(list, lean_edges)),
        "paths": paths,
        "owt": owt,
        "owt_budget": OWT_BUDGET,
        "integrity": integrity,
        "gates": gates,
        "pass": all(gates.values()),
        "selected_flagship": selected,
        "training_or_tuning_steps": 0,
    }
    output = Path(args.output_dir) / "lean_candidate_selection.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("lean-selection",),
    )
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--processed_dataset_dir",
        default=".cache/openwebtext-gpt2-block1024",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.command == "lean-selection":
        run_lean_selection(parsed)
