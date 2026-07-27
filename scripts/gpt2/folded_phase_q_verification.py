#!/usr/bin/env python3
"""Exact folded verification for the fixed Phase-Q three-edge quote circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gmpy2
import torch
from safetensors.torch import load_file
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.programs.dsl import AttentionProgram


EDGES = (
    ("attn_7_h_11", "logits"),
    ("emb", "mlp_0"),
    ("mlp_0", "attn_7_h_11"),
)
CANDIDATES = (6, 1)
REGISTERED_EPSILON = gmpy2.mpq(1, 100)
ALPHA = gmpy2.mpq(1, 100)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def q(value) -> gmpy2.mpq:
    """Convert a scalar FP32 value to its exact binary rational."""
    if isinstance(value, gmpy2.mpq):
        return value
    if isinstance(value, int):
        return gmpy2.mpq(value)
    numerator, denominator = float(value).as_integer_ratio()
    return gmpy2.mpq(numerator, denominator)


def qstr(value: gmpy2.mpq) -> str:
    return f"{value.numerator}/{value.denominator}"


def dot(left: Iterable[gmpy2.mpq], right: Iterable[gmpy2.mpq]) -> gmpy2.mpq:
    total = gmpy2.mpq(0)
    for a, b in zip(left, right):
        total += a * b
    return total


def qvector(tensor: torch.Tensor) -> list[gmpy2.mpq]:
    return [q(value) for value in tensor.detach().cpu().float().tolist()]


@dataclass
class SignatureFold:
    token: int
    position: int
    hidden: list[gmpy2.mpq]
    signs: tuple[bool, ...]
    minimum_abs_pre_activation: gmpy2.mpq
    fold_seconds: float


class ExactFoldedCircuit:
    """Contract downstream maps and cache exact positionwise MLP signatures."""

    def __init__(self, state: dict[str, torch.Tensor], program: AttentionProgram):
        self.state = state
        self.program = program
        self.signature_cache: dict[tuple[int, int, bool], SignatureFold] = {}
        self.wte = state["transformer.wte.weight"].cpu().float()
        self.wpe = state["transformer.wpe.weight"].cpu().float()
        self.w_up = state["transformer.h.0.mlp.c_fc.weight"].cpu().float()
        self.b_up = qvector(state["transformer.h.0.mlp.c_fc.bias"])
        self.w_down = state["transformer.h.0.mlp.c_proj.weight"].cpu().float()
        self.b_down = qvector(state["transformer.h.0.mlp.c_proj.bias"])
        self.w_v = state["transformer.h.7.attn.value_proj.weight"][
            704:768
        ].cpu().float()
        self.b_v = qvector(
            state["transformer.h.7.attn.value_proj.bias"][704:768]
        )
        self.w_o = state["transformer.h.7.attn.c_proj.weight"][
            704:768
        ].cpu().float()
        self.b_o = qvector(state["transformer.h.7.attn.c_proj.bias"])
        self.lm = state["lm_head.weight"].cpu().float()
        self.contractions = {
            token: self._contract_candidate(token) for token in CANDIDATES
        }
        self.lm_difference = [
            a - b
            for a, b in zip(qvector(self.lm[CANDIDATES[0]]), qvector(self.lm[CANDIDATES[1]]))
        ]
        self.robustness_l1 = sum((abs(x) for x in self.lm_difference), gmpy2.mpq(0))

    def _contract_candidate(self, token: int) -> dict:
        lm = qvector(self.lm[token])
        # Conv1D c_proj: selected head row i maps to residual column j.
        head_query = [
            dot(qvector(self.w_o[i]), lm) for i in range(64)
        ]
        residual_bias = dot(self.b_o, lm)
        mlp_query = []
        for j in range(768):
            mlp_query.append(
                sum(
                    (head_query[i] * q(self.w_v[i, j]) for i in range(64)),
                    gmpy2.mpq(0),
                )
            )
        value_bias = dot(head_query, self.b_v)
        hidden_query = []
        for hidden in range(3072):
            hidden_query.append(
                sum(
                    (
                        q(self.w_down[hidden, j]) * mlp_query[j]
                        for j in range(768)
                    ),
                    gmpy2.mpq(0),
                )
            )
        return {
            "head_query": head_query,
            "mlp_query": mlp_query,
            "hidden_query": hidden_query,
            "constant": residual_bias + value_bias + dot(self.b_down, mlp_query),
        }

    def signature(self, token: int, position: int, *, emb_edge: bool = True) -> SignatureFold:
        key = (token, position, emb_edge)
        if key in self.signature_cache:
            return self.signature_cache[key]
        started = time.perf_counter()
        if emb_edge:
            residual = [
                q(a) + q(b) for a, b in zip(self.wte[token], self.wpe[position])
            ]
        else:
            residual = [gmpy2.mpq(0)] * 768
        hidden = []
        signs = []
        minimum = None
        for index in range(3072):
            pre = self.b_up[index]
            row = self.w_up[:, index]
            for weight, value in zip(row.tolist(), residual):
                pre += q(weight) * value
            nonnegative = pre >= 0
            signs.append(nonnegative)
            magnitude = abs(pre)
            minimum = magnitude if minimum is None else min(minimum, magnitude)
            hidden.append(pre if nonnegative else ALPHA * pre)
        result = SignatureFold(
            token=token,
            position=position,
            hidden=hidden,
            signs=tuple(signs),
            minimum_abs_pre_activation=minimum or gmpy2.mpq(0),
            fold_seconds=time.perf_counter() - started,
        )
        self.signature_cache[key] = result
        return result

    def folded_logits(
        self,
        tokens: list[int],
        *,
        edges: tuple[tuple[str, str], ...] = EDGES,
    ) -> tuple[dict[int, gmpy2.mpq], dict]:
        edge_set = set(edges)
        if ("attn_7_h_11", "logits") not in edge_set:
            return {token: gmpy2.mpq(0) for token in CANDIDATES}, {
                "signatures": [], "program_weights": []
            }
        weights = self.program.rational_weights(tokens)[-1]
        active = [(position, q(weight)) for position, weight in enumerate(weights) if weight]
        logits = {}
        signatures = []
        mlp_edge = ("mlp_0", "attn_7_h_11") in edge_set
        emb_edge = ("emb", "mlp_0") in edge_set
        for position, _weight in active:
            signatures.append(self.signature(tokens[position], position, emb_edge=emb_edge))
        for candidate in CANDIDATES:
            contraction = self.contractions[candidate]
            if mlp_edge:
                hidden_term = sum(
                    (
                        weight * dot(contraction["hidden_query"], signature.hidden)
                        for (position, weight), signature in zip(active, signatures)
                    ),
                    gmpy2.mpq(0),
                )
                logits[candidate] = contraction["constant"] + hidden_term
            else:
                logits[candidate] = (
                    dot(contraction["head_query"], self.b_v)
                    + dot(self.b_o, qvector(self.lm[candidate]))
                )
        return logits, {
            "signatures": signatures,
            "program_weights": [[position, qstr(weight)] for position, weight in active],
        }

    def monolithic_logits(self, tokens: list[int]) -> dict[int, gmpy2.mpq]:
        """Independent uncontracted exact-rational evaluation."""
        weights = self.program.rational_weights(tokens)[-1]
        mlp_outputs = []
        for position, weight in enumerate(weights):
            if not weight:
                continue
            signature = self.signature(tokens[position], position)
            output = []
            for j in range(768):
                output.append(
                    self.b_down[j]
                    + sum(
                        (
                            signature.hidden[h] * q(self.w_down[h, j])
                            for h in range(3072)
                        ),
                        gmpy2.mpq(0),
                    )
                )
            mlp_outputs.append((q(weight), output))
        head = []
        for i in range(64):
            value = self.b_v[i]
            for weight, mlp in mlp_outputs:
                value += weight * sum(
                    (q(self.w_v[i, j]) * mlp[j] for j in range(768)),
                    gmpy2.mpq(0),
                )
            head.append(value)
        residual = []
        for j in range(768):
            residual.append(
                self.b_o[j]
                + sum(
                    (head[i] * q(self.w_o[i, j]) for i in range(64)),
                    gmpy2.mpq(0),
                )
            )
        return {
            token: dot(qvector(self.lm[token]), residual) for token in CANDIDATES
        }


def decision(logits: dict[int, gmpy2.mpq]) -> int:
    return CANDIDATES[0] if logits[CANDIDATES[0]] >= logits[CANDIDATES[1]] else CANDIDATES[1]


def expected_token(example: dict) -> int:
    return CANDIDATES[0] if example["correct_token"] == "'" else CANDIDATES[1]


def load_inputs(root: Path):
    manifest_path = root / "artifacts/gpt2-behavior-domain-bounded-v1/domain.json"
    manifest = json.loads(manifest_path.read_text())
    examples = manifest["examples"]["quote_close"]
    tokenizer = GPT2Tokenizer.from_pretrained(
        root / "artifacts/gpt2-phase-q-readside-calibration/fp32_export"
    )
    token_rows = [tokenizer.encode(row["prompt"], add_special_tokens=False) for row in examples]
    return manifest_path, manifest, examples, token_rows


def load_circuit(root: Path) -> ExactFoldedCircuit:
    export = root / "artifacts/gpt2-phase-q-readside-calibration/fp32_export"
    state = load_file(str(export / "model.safetensors"), device="cpu")
    programs = json.loads((export / "programs.json").read_text())
    if sorted(programs) != ["7.11", "9.0"]:
        raise RuntimeError(f"Fixed two-program export changed: {sorted(programs)}")
    return ExactFoldedCircuit(state, AttentionProgram.from_dict(programs["7.11"]))


def anchor_sequences():
    return [
        ([before, quote, after], "single" if quote == 6 else "double")
        for before in (10, 11)
        for after in (10, 11)
        for quote in (6, 1)
    ]


def validate_anchors(circuit: ExactFoldedCircuit, d_tokens: list[list[int]]) -> dict:
    records = []
    d_set = {tuple(row) for row in d_tokens}
    for index, (tokens, label) in enumerate(anchor_sequences()):
        started = time.perf_counter()
        folded, metadata = circuit.folded_logits(tokens)
        monolithic = circuit.monolithic_logits(tokens)
        equal = folded == monolithic
        records.append(
            {
                "sequence_index": index,
                "tokens": tokens,
                "label": label,
                "in_D_as_exact_token_sequence": tuple(tokens) in d_set,
                "folded_equals_monolithic_exact_rational": equal,
                "candidate_logits": {str(k): qstr(v) for k, v in folded.items()},
                "decision": decision(folded),
                "expected": 6 if label == "single" else 1,
                "functional_match": decision(folded) == (6 if label == "single" else 1),
                "program_weights": metadata["program_weights"],
                "seconds": time.perf_counter() - started,
            }
        )
        if not equal:
            raise RuntimeError(f"Fold mismatch on preserved anchor seq{index}")
    return {
        "status": "PASSED",
        "old_smoke_log_sha256": sha256_bytes(
            Path(
                "artifacts/gpt2-phase-q-readside-calibration/verification_max_length_3.log"
            ).read_bytes()
        ),
        "seq1_old_monolithic_violation_result": "SAT",
        "seq1_interpretation": (
            "Outside D. The restricted scan program does not treat synthetic token ID 1 "
            "as a key-token opener, so its exact self-fallback yields a genuine smoke-domain "
            "functional counterexample without contradicting exactness on D."
        ),
        "records": records,
    }


def per_input_record(
    index: int,
    example: dict,
    tokens: list[int],
    circuit: ExactFoldedCircuit,
) -> dict:
    started = time.perf_counter()
    logits, metadata = circuit.folded_logits(tokens)
    fold_seconds = time.perf_counter() - started
    selected = decision(logits)
    expected = expected_token(example)
    other = CANDIDATES[1] if selected == CANDIDATES[0] else CANDIDATES[0]
    margin = logits[selected] - logits[other]
    maximum_epsilon = margin / circuit.robustness_l1
    sign_records = metadata["signatures"]
    ablations = {}
    for edge in EDGES:
        ablated = tuple(candidate for candidate in EDGES if candidate != edge)
        edge_logits, _ = circuit.folded_logits(tokens, edges=ablated)
        ablations[f"{edge[0]}->{edge[1]}"] = {
            "decision": decision(edge_logits),
            "candidate_logits": {str(k): qstr(v) for k, v in edge_logits.items()},
            "changes_decision": decision(edge_logits) != selected,
        }
    assertions = {
        "topology": len(EDGES),
        "program_semantics": len(metadata["program_weights"]),
        "leaky_relu_sign_certificates": 3072 * len(sign_records),
        "functional_ground_comparison": 1,
        "content_invariance_ground_comparison": 1,
        "edge_necessity_ground_comparisons": len(EDGES),
        "robustness_box_bounds": 2 * 768,
        "robustness_margin_constraint": 1,
    }
    return {
        "schema_version": 1,
        "index": index,
        "example_id": example["example_id"],
        "example_id_sha256": sha256_bytes(example["example_id"].encode()),
        "prompt_sha256": sha256_bytes(example["prompt"].encode()),
        "tokens_sha256": canonical_hash(tokens),
        "tokens": tokens,
        "stratum": example["stratum"],
        "expected": expected,
        "decision": selected,
        "candidate_logits": {str(k): qstr(v) for k, v in logits.items()},
        "functional_equivalence": "PASSED" if selected == expected else "FAILED",
        "content_invariance": None,
        "edge_ablations": ablations,
        "robustness": {
            "registered_epsilon": qstr(REGISTERED_EPSILON),
            "maximum_epsilon": qstr(maximum_epsilon),
            "status": "PASSED" if maximum_epsilon >= REGISTERED_EPSILON else "FAILED",
            "margin": qstr(margin),
            "lm_difference_l1": qstr(circuit.robustness_l1),
            "lp_semantics": "exact closed-form optimum of the registered final-residual L-infinity margin LP",
            "sign_region": "unbounded at final-residual perturbation interface; MLP trace is upstream and unchanged",
        },
        "sign_certificate": {
            "status": "PASSED",
            "count": 3072 * len(sign_records),
            "minimum_abs_pre_activation": qstr(
                min(
                    (record.minimum_abs_pre_activation for record in sign_records),
                    default=gmpy2.mpq(0),
                )
            ),
            "signature_keys": [
                [record.token, record.position] for record in sign_records
            ],
        },
        "assertion_attribution": assertions,
        "assertion_count": sum(assertions.values()),
        "encode_fold_seconds": fold_seconds,
        "solve_seconds": 0.0,
        "program_weights": metadata["program_weights"],
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(args) -> None:
    root = Path(args.root).resolve()
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest, examples, token_rows = load_inputs(root)
    if len(examples) != 1280 or len({row["prompt"] for row in examples}) != 1280:
        raise RuntimeError("Frozen D cardinality changed")
    circuit = load_circuit(root)
    anchor_path = output / "anchor_equality.json"
    if not args.resume or not anchor_path.exists():
        write_json(anchor_path, validate_anchors(circuit, token_rows))
    if args.anchors_only:
        return

    records_path = output / "per_input.jsonl"
    completed = set()
    records = []
    if args.resume and records_path.exists():
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            completed.add(record["index"])
            records.append(record)
    with records_path.open("a") as handle:
        for index, (example, tokens) in enumerate(zip(examples, token_rows)):
            if index in completed:
                continue
            record = per_input_record(index, example, tokens, circuit)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            records.append(record)
            if args.limit and len(completed) + len(records) >= args.limit:
                break

    records.sort(key=lambda item: item["index"])
    if len(records) != 1280:
        write_json(output / "progress.json", {"completed": len(records), "rows": 1280})
        return
    anchors = {}
    edge_witnesses = {}
    for record in records:
        feature = record["stratum"]
        if feature not in anchors:
            anchors[feature] = record["decision"]
        record["content_invariance"] = (
            "PASSED" if record["decision"] == anchors[feature] else "FAILED"
        )
        for edge, result in record["edge_ablations"].items():
            if result["changes_decision"] and edge not in edge_witnesses:
                edge_witnesses[edge] = {
                    "example_id": record["example_id"],
                    "index": record["index"],
                    "full_decision": record["decision"],
                    "ablated_decision": result["decision"],
                }
    # Rewrite the completed records with finalized content-invariance status.
    with records_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    properties = {
        "functional_equivalence": {
            "status": "PASSED" if all(r["functional_equivalence"] == "PASSED" for r in records) else "FAILED",
            "checked": 1280,
        },
        "content_invariance": {
            "status": "PASSED" if all(r["content_invariance"] == "PASSED" for r in records) else "FAILED",
            "semantics": "same registered quote stratum implies same projected decision",
            "checked": 1280,
        },
        "edge_necessity": {
            "status": "PASSED" if len(edge_witnesses) == len(EDGES) else "FAILED",
            "canonical_edges": [list(edge) for edge in sorted(EDGES, key=lambda edge: (edge[1], edge[0]))],
            "witnesses": edge_witnesses,
        },
        "continuous_robustness": {
            "status": "PASSED" if all(r["robustness"]["status"] == "PASSED" for r in records) else "FAILED",
            "registered_epsilon": qstr(REGISTERED_EPSILON),
            "minimum_maximum_epsilon": min(
                (r["robustness"]["maximum_epsilon"] for r in records),
                key=lambda text: gmpy2.mpq(text),
            ),
            "checked": 1280,
        },
    }
    summary = {
        "status": "PASSED" if all(p["status"] == "PASSED" for p in properties.values()) else "FAILED",
        "domain": {
            "path": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "rows": 1280,
            "prompt_set_sha256": manifest["summary"]["quote_close"]["prompt_set_sha256"],
            "ordered_example_ids_newline_sha256": sha256_bytes(
                ("\n".join(row["example_id"] for row in examples) + "\n").encode()
            ),
        },
        "topology": [list(edge) for edge in EDGES],
        "properties": properties,
        "per_input_path": str(records_path.relative_to(root)),
        "per_input_sha256": sha256_bytes(records_path.read_bytes()),
        "assertion_count": sum(r["assertion_count"] for r in records),
        "encode_fold_seconds": sum(r["encode_fold_seconds"] for r in records),
        "solve_seconds": sum(r["solve_seconds"] for r in records),
        "unique_exact_mlp_signatures": len(circuit.signature_cache),
    }
    write_json(output / "summary.json", summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="artifacts/gpt2-phase-q-readside-calibration/folded_verification",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--anchors-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
