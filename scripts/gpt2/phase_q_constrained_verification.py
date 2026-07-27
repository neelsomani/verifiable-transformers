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
    cleanup_graph,
    compute_projected_agreement,
    find_circuit,
    controlled_forward,
    get_candidate_token_ids,
    load_behavior_examples,
    load_model_with_variants,
    projected_trim_circuit,
    select_last_real_logits,
)
from scripts.gpt2.readside_calibration import (
    canonical_hash,
    compare_paired_rows,
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
EXTRACTION_THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


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


def _projected_summary(logits, reference, values):
    last = select_last_real_logits(logits, values["attention_mask"])[
        :, values["candidate_token_ids"]
    ].float()
    targets = values["targets"].to(last.device)
    decisions = last.argmax(-1)
    indices = torch.arange(len(decisions), device=last.device)
    margins = last[indices, targets] - last[indices, 1 - targets]
    return {
        "rows": len(decisions),
        "correct": int((decisions == targets).sum()),
        "projected_agreement": compute_projected_agreement(
            reference,
            logits,
            values["attention_mask"],
            values["candidate_token_ids"],
        ),
        "minimum_signed_correct_margin": float(margins.min()),
        "mean_signed_correct_margin": float(margins.mean()),
        "decisions_sha256": canonical_hash(decisions.cpu().tolist()),
        "margins_sha256": canonical_hash(margins.cpu().tolist()),
    }


def run_reextract(args) -> None:
    root = Path(__file__).resolve().parents[2]
    registration = json.loads(Path(args.registration).read_text())
    configure_determinism(registration["execution_policy"]["seed"])
    frozen = registration["frozen_inputs"]
    tokenizer = GPT2Tokenizer.from_pretrained(root / frozen["source_model"])
    tokenizer.pad_token = tokenizer.eos_token
    values = _domain_values(root, registration, tokenizer, args.device)
    model, programs, _rung3, _copied = load_rung3_candidate(
        root, registration, args.device, lean=False
    )
    graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
    prompts = [example.prompt for example in values["examples"]]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)
    with torch.no_grad():
        reference = controlled_forward(
            model, input_ids, attention_mask, set(graph.all_edges), graph
        )

    output_dir = Path(args.output_dir) / "reextraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for threshold in EXTRACTION_THRESHOLDS:
        print(f"REEXTRACTION threshold={threshold}", flush=True)
        edges, edge_log = find_circuit(
            model=model,
            tokenizer=tokenizer,
            examples=values["examples"],
            graph=graph,
            threshold=threshold,
            metric="candidate_kl",
            task="quote_close",
            min_agreement=1.0,
            ablation_mode="zero",
            ablation_cache=None,
            device=args.device,
            initial_edges=set(graph.all_edges),
            verbose=False,
        )
        edges = cleanup_graph(edges, graph)
        edges, trim_log = projected_trim_circuit(
            model,
            graph,
            edges,
            input_ids,
            attention_mask,
            reference,
            values["candidate_token_ids"],
            1.0,
        )
        edge_log.extend(trim_log)
        with torch.no_grad():
            logits = controlled_forward(
                model, input_ids, attention_mask, edges, graph
            )
        summary = _projected_summary(logits, reference, values)
        retained_program_heads = sorted(
            node
            for node in PROGRAM_NODES
            if any(node == parent for parent, _child in edges)
        )
        compact = {
            "threshold": threshold,
            "metric": "candidate_kl",
            "min_agreement": 1.0,
            "ablation": "zero",
            "num_edges": len(edges),
            "edges": sorted(map(list, edges)),
            "retained_program_heads": retained_program_heads,
            "summary": summary,
            "edge_log_sha256": canonical_hash(edge_log),
            "edge_log_entries": len(edge_log),
        }
        attempt_path = output_dir / f"threshold_{threshold:g}.json"
        attempt_path.write_text(json.dumps(compact, indent=2) + "\n")
        attempts.append(compact)
        print(
            json.dumps(
                {
                    "threshold": threshold,
                    "edges": len(edges),
                    "correct": summary["correct"],
                    "agreement": summary["projected_agreement"],
                    "retained_program_heads": retained_program_heads,
                }
            ),
            flush=True,
        )

    eligible = [
        attempt
        for attempt in attempts
        if attempt["summary"]["correct"] == 1280
        and attempt["summary"]["projected_agreement"] == 1.0
    ]
    if not eligible:
        raise RuntimeError("No re-extraction candidate achieved exact agreement")
    selected = min(
        eligible, key=lambda value: (value["num_edges"], value["threshold"])
    )
    report = {
        "schema_version": 1,
        "fixed_flagship": "two_program_rung3",
        "domain": values["provenance"],
        "thresholds": list(EXTRACTION_THRESHOLDS),
        "selection_rule": (
            "exact 1280/1280 and projected agreement 1.0; minimum edge count; "
            "lower-threshold tie-break"
        ),
        "attempts": [
            {
                key: value
                for key, value in attempt.items()
                if key not in {"edges"}
            }
            for attempt in attempts
        ],
        "selected": selected,
        "programs_installed": [
            f"{layer}.{head}" for layer, head in sorted(programs)
        ],
        "forbidden_heads": [],
        "training_or_tuning_steps": 0,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"selected": selected}, indent=2))


def run_causal_replay(args) -> None:
    root = Path(__file__).resolve().parents[2]
    registration = json.loads(Path(args.registration).read_text())
    configure_determinism(registration["execution_policy"]["seed"])
    frozen = registration["frozen_inputs"]
    tokenizer = GPT2Tokenizer.from_pretrained(root / frozen["source_model"])
    tokenizer.pad_token = tokenizer.eos_token
    values = _domain_values(root, registration, tokenizer, args.device)
    model, programs, _rung3, copied = load_rung3_candidate(
        root, registration, args.device, lean=False
    )
    graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
    full = set(graph.all_edges)
    selected_payload = json.loads(
        (
            Path(args.output_dir) / "reextraction/selection.json"
        ).read_text()
    )["selected"]
    selected = {tuple(edge) for edge in selected_payload["edges"]}
    internal = internal_circuit_nodes(selected)
    registered_selected = load_selected_edges(root / frozen["circuit"])
    registered_internal = internal_circuit_nodes(registered_selected)
    edge_sets = {
        "controlled_full": full,
        "circuit_only": selected,
        "full_zero_attn_7_h_11": {
            edge for edge in full if edge[0] != "attn_7_h_11"
        },
        "core_zero_attn_7_h_11": {
            edge for edge in selected if edge[0] != "attn_7_h_11"
        },
        "full_zero_attn_9_h_0": {
            edge for edge in full if edge[0] != "attn_9_h_0"
        },
        "core_zero_attn_9_h_0": {
            edge for edge in selected if edge[0] != "attn_9_h_0"
        },
        "zero_both_program_heads": {
            edge for edge in full if edge[0] not in PROGRAM_NODES
        },
        "core_zero_joint": {
            edge for edge in selected if edge[0] not in PROGRAM_NODES
        },
        "whole_selected_circuit_node_zero": {
            edge for edge in full if edge[0] not in internal
        },
        "registered_whole_selected_circuit_node_zero": {
            edge for edge in full if edge[0] not in registered_internal
        },
        "whole_selected_circuit_edge_zero": full - selected,
    }
    batch_size = registration["execution_policy"]["batch_size_for_identity"]
    raw = {
        "native_full_forward": None,
        **{name: None for name in edge_sets},
    }
    summaries = {}
    native_rows = []
    # Keep raw rows only long enough to perform registered bitwise comparisons.
    def rows_for(edges=None):
        rows = []
        with torch.no_grad():
            for start in range(0, len(values["examples"]), batch_size):
                stop = start + batch_size
                ids = values["input_ids"][start:stop]
                mask = values["attention_mask"][start:stop]
                logits = (
                    model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                    if edges is None
                    else controlled_forward(model, ids, mask, edges, graph)
                )
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
                            "example_id": values["examples"][
                                start + offset
                            ].example_id,
                            "target": int(targets[offset]),
                            "decision": int(decisions[offset]),
                            "candidate_logits": [
                                float(value) for value in projected[offset].cpu()
                            ],
                            "candidate_margin": float(margins[offset]),
                        }
                    )
        return rows

    native_rows = rows_for()
    raw["native_full_forward"] = native_rows
    for name, edges in edge_sets.items():
        raw[name] = rows_for(edges)
    for name, rows in raw.items():
        correct = sum(row["decision"] == row["target"] for row in rows)
        mismatches = [
            row["example_id"] for row in rows if row["decision"] != row["target"]
        ]
        summaries[name] = {
            "rows": len(rows),
            "correct": correct,
            "mismatch_ids": mismatches,
            "decisions_sha256": canonical_hash(
                [row["decision"] for row in rows]
            ),
            "candidate_logits_sha256": canonical_hash(
                [row["candidate_logits"] for row in rows]
            ),
            "candidate_margins_sha256": canonical_hash(
                [row["candidate_margin"] for row in rows]
            ),
            "minimum_margin": min(row["candidate_margin"] for row in rows),
            "mean_margin": (
                sum(row["candidate_margin"] for row in rows) / len(rows)
            ),
        }

    paired = json.loads(
        (
            Path(args.output_dir) / "paired_baseline.json"
        ).read_text()
    )["numerical_identity_reference"]
    epsilon = registration["execution_policy"]["identity_epsilon_max_abs"]
    identity = {
        "zero_both_program_heads": compare_paired_rows(
            raw["zero_both_program_heads"],
            paired["zero_both_program_heads"],
            epsilon,
        ),
        "whole_selected_circuit_node_zero": compare_paired_rows(
            raw["registered_whole_selected_circuit_node_zero"],
            paired["whole_selected_circuit_node_zero"],
            epsilon,
        ),
    }
    baseline_hashes = json.loads(
        (Path(args.output_dir) / "paired_baseline.json").read_text()
    )["parameter_hashes"]
    hashes = tensor_parameter_hashes(model)
    changed = sorted(
        name
        for name, digest in hashes.items()
        if baseline_hashes.get(name) != digest
    )
    permitted = sorted(set(copied))
    integrity = {
        "changed_tensor_names": changed,
        "permitted_tensor_names": permitted,
        "unexpected_changed_tensor_names": sorted(set(changed) - set(permitted)),
        "pass": set(changed) <= set(permitted),
        "state_dict_hashes_sha256": canonical_hash(hashes),
    }
    owt_registered = json.loads(
        (Path(args.output_dir) / "rung3_program_local.json").read_text()
    )["history"][-1]["owt"]
    gates = {
        "full_exact": summaries["native_full_forward"]["correct"] == 1280,
        "controlled_full_exact": summaries["controlled_full"]["correct"] == 1280,
        "circuit_only_exact": summaries["circuit_only"]["correct"] == 1280,
        "joint_program_set_necessary": (
            summaries["zero_both_program_heads"]["correct"] < 1280
        ),
        "whole_selected_circuit_necessary": (
            summaries["whole_selected_circuit_node_zero"]["correct"] < 1280
        ),
        "registered_whole_circuit_necessary": (
            summaries["registered_whole_selected_circuit_node_zero"]["correct"]
            < 1280
        ),
        "leakage_identity": all(value["pass"] for value in identity.values()),
        "parameter_integrity": integrity["pass"],
        "owt_within_budget": (
            owt_registered["eval_perplexity"] <= OWT_BUDGET
        ),
    }
    report = {
        "schema_version": 1,
        "criterion": "constrained_calibration_set_level",
        "fixed_flagship": "two_program_rung3",
        "domain": values["provenance"],
        "selected_circuit": {
            "num_edges": len(selected),
            "edges": sorted(map(list, selected)),
            "internal_nodes": internal,
            "retained_program_heads": selected_payload[
                "retained_program_heads"
            ],
        },
        "paths": summaries,
        "identity_against_paired_baseline": identity,
        "parameter_integrity": integrity,
        "owt": {
            **owt_registered,
            "reuse_basis": (
                "The chosen fixed checkpoint is byte-identical to the rung-3 "
                "checkpoint evaluated before this amendment; extraction and "
                "lesions do not mutate parameters."
            ),
        },
        "owt_budget": OWT_BUDGET,
        "mechanism_interpretation": {
            "attn_7_h_11": "circuit-internally load-bearing",
            "attn_9_h_0": "auxiliary/redundant and absent from selected core",
        },
        "gates": gates,
        "pass_before_smt_edge_necessity": all(gates.values()),
        "pending_gate": "SMT circuit-internal edge necessity",
        "training_or_tuning_steps": 0,
        "programs_installed": [
            f"{layer}.{head}" for layer, head in sorted(programs)
        ],
    }
    output = Path(args.output_dir) / "amended_causal_replay.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass_before_smt_edge_necessity"]:
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("lean-selection", "reextract", "causal-replay"),
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
    elif parsed.command == "reextract":
        run_reextract(parsed)
    elif parsed.command == "causal-replay":
        run_causal_replay(parsed)
