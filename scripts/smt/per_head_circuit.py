"""SMT encoding for circuits whose attention nodes are pre-W_O heads."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from z3 import ArithRef, Solver, Sum, RealVal

from scripts.programs.dsl import AttentionProgram
from .attribution import increment_norm_instances, record_solver_delta
from .encoders import (
    encode_mlp,
    encode_mlp_with_trace,
    encode_signed_l1_band_norm,
    encode_signed_l1_band_norm_with_trace,
    encode_sparsemax,
    encode_sparsemax_with_support,
    z3_real,
)


_HEAD_RE = re.compile(r"^attn_(\d+)_h_(\d+)$")


def _zero(seq_len: int, width: int) -> List[List[ArithRef]]:
    return [[RealVal(0) for _ in range(width)] for _ in range(seq_len)]


def _parse_head(node: str) -> tuple[int, int] | None:
    match = _HEAD_RE.match(node)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def _residual_for(
    child: str,
    node_outputs: Dict[str, List[List[ArithRef]]],
    circuit_edges: Set[Tuple[str, str]],
    model_weights: Dict[str, Any],
    seq_len: int,
) -> List[List[ArithRef]]:
    d_model = model_weights["d_model"]
    n_heads = model_weights["n_heads"]
    head_dim = d_model // n_heads
    residual = _zero(seq_len, d_model)
    selected_by_layer: Dict[int, Dict[int, List[List[ArithRef]]]] = {}

    for parent, target in circuit_edges:
        if target != child or parent not in node_outputs:
            continue
        parsed = _parse_head(parent)
        if parsed is None:
            parent_output = node_outputs[parent]
            for position in range(seq_len):
                for coord in range(d_model):
                    residual[position][coord] += parent_output[position][coord]
        else:
            layer, head = parsed
            selected_by_layer.setdefault(layer, {})[head] = node_outputs[parent]

    for layer, selected in selected_by_layer.items():
        W_o = model_weights[f"attn_{layer}_W_o"]
        b_o = model_weights[f"attn_{layer}_b_o"]
        for position in range(seq_len):
            concatenated: List[ArithRef] = []
            for head in range(n_heads):
                concatenated.extend(
                    selected[head][position]
                    if head in selected
                    else [RealVal(0)] * head_dim
                )
            for output_coord in range(d_model):
                residual[position][output_coord] += Sum(
                    [
                        z3_real(W_o[output_coord][source_coord])
                        * concatenated[source_coord]
                        for source_coord in range(d_model)
                    ]
                ) + z3_real(b_o[output_coord])
    return residual


def _normalize(
    residual: List[List[ArithRef]],
    model_weights: Dict[str, Any],
    gamma_key: str,
    beta_key: str,
    solver: Solver,
    ctx_prefix: str,
    trace: Dict[str, Any] | None,
    assertion_profile: Dict[str, Any] | None,
) -> List[List[ArithRef]]:
    norm_variant = model_weights.get("norm_variant", "layer_norm")
    if norm_variant == "none":
        return residual
    if norm_variant != "signed_l1_band_norm":
        raise ValueError(f"No exact SMT encoding for norm variant {norm_variant!r}")

    d_model = model_weights["d_model"]
    gamma = model_weights[gamma_key]
    beta = model_weights[beta_key]
    half_low = model_weights["half_low"]
    half_high = model_weights["half_high"]
    pos_fallback = [1.0 if index % 2 == 0 else 0.0 for index in range(d_model)]
    neg_fallback = [0.0 if index % 2 == 0 else 1.0 for index in range(d_model)]
    result = []
    for position, vector in enumerate(residual):
        increment_norm_instances(assertion_profile)
        context = f"{ctx_prefix}_norm_p{position}"
        before = len(solver.assertions())
        if trace is None:
            normalized = encode_signed_l1_band_norm(
                vector,
                gamma,
                beta,
                half_low,
                half_high,
                pos_fallback,
                neg_fallback,
                solver,
                context,
            )
        else:
            normalized = encode_signed_l1_band_norm_with_trace(
                vector,
                gamma,
                beta,
                half_low,
                half_high,
                pos_fallback,
                neg_fallback,
                trace["bandnorm"][context],
                solver,
                context,
            )
        record_solver_delta(assertion_profile, "norm", solver, before)
        result.append(normalized)
    return result


def _program_for(
    model_weights: Dict[str, Any], layer: int, head: int
) -> AttentionProgram | None:
    raw = model_weights.get("program_heads", {}).get(f"{layer}.{head}")
    return None if raw is None else AttentionProgram.from_dict(raw)


def _encode_head(
    input_tokens: List[int],
    node_outputs: Dict[str, List[List[ArithRef]]],
    layer: int,
    head: int,
    circuit_edges: Set[Tuple[str, str]],
    model_weights: Dict[str, Any],
    solver: Solver,
    ctx_prefix: str,
    trace: Dict[str, Any] | None,
    assertion_profile: Dict[str, Any] | None,
) -> List[List[ArithRef]]:
    d_model = model_weights["d_model"]
    n_heads = model_weights["n_heads"]
    head_dim = d_model // n_heads
    seq_len = len(input_tokens)
    node = f"attn_{layer}_h_{head}"
    residual = _residual_for(
        node, node_outputs, circuit_edges, model_weights, seq_len
    )
    normalized = _normalize(
        residual,
        model_weights,
        f"attn_{layer}_norm_gamma",
        f"attn_{layer}_norm_beta",
        solver,
        ctx_prefix,
        trace,
        assertion_profile,
    )
    start, stop = head * head_dim, (head + 1) * head_dim
    W_v = model_weights[f"attn_{layer}_W_v"][start:stop]
    b_v = model_weights[f"attn_{layer}_b_v"][start:stop]
    values = [
        [
            Sum(
                [
                    z3_real(W_v[coord][source]) * normalized[position][source]
                    for source in range(d_model)
                ]
            )
            + z3_real(b_v[coord])
            for coord in range(head_dim)
        ]
        for position in range(seq_len)
    ]

    program = _program_for(model_weights, layer, head)
    if program is not None:
        weights = program.rational_weights(input_tokens)
        return [
            [
                Sum(
                    [
                        z3_real(weights[query][key]) * values[key][coord]
                        for key in range(seq_len)
                    ]
                )
                for coord in range(head_dim)
            ]
            for query in range(seq_len)
        ]

    W_q = model_weights[f"attn_{layer}_W_q"][start:stop]
    W_k = model_weights[f"attn_{layer}_W_k"][start:stop]
    b_q = model_weights[f"attn_{layer}_b_q"][start:stop]
    b_k = model_weights[f"attn_{layer}_b_k"][start:stop]
    queries = []
    keys = []
    for position in range(seq_len):
        queries.append(
            [
                Sum(
                    [z3_real(W_q[c][s]) * normalized[position][s] for s in range(d_model)]
                )
                + z3_real(b_q[c])
                for c in range(head_dim)
            ]
        )
        keys.append(
            [
                Sum(
                    [z3_real(W_k[c][s]) * normalized[position][s] for s in range(d_model)]
                )
                + z3_real(b_k[c])
                for c in range(head_dim)
            ]
        )

    scale = head_dim**0.5
    output = []
    attention_before = len(solver.assertions())
    for query_position in range(seq_len):
        scores = [
            Sum(
                [queries[query_position][c] * keys[key_position][c] for c in range(head_dim)]
            )
            / z3_real(scale)
            for key_position in range(query_position + 1)
        ]
        support_key = f"{ctx_prefix}_p{query_position}"
        if trace is None:
            weights = encode_sparsemax(scores, solver, support_key)
        else:
            weights = encode_sparsemax_with_support(
                scores, trace["sparsemax"][support_key], solver, support_key
            )
        output.append(
            [
                Sum(
                    [
                        weights[key_position] * values[key_position][coord]
                        for key_position in range(query_position + 1)
                    ]
                )
                for coord in range(head_dim)
            ]
        )
    record_solver_delta(
        assertion_profile, "attention", solver, attention_before
    )
    return output


def encode_per_head_circuit_forward(
    input_tokens: List[int],
    circuit_edges: Set[Tuple[str, str]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    solver: Solver,
    ctx_prefix: str,
    trace: Dict[str, Any] | None = None,
    assertion_profile: Dict[str, Any] | None = None,
) -> Dict[int, ArithRef]:
    seq_len = len(input_tokens)
    d_model = model_weights["d_model"]
    n_layers = model_weights["n_layers"]
    n_heads = model_weights["n_heads"]
    head_dim = d_model // n_heads
    active = {"emb", "logits"}
    for source, target in circuit_edges:
        active.update((source, target))

    wte, wpe = model_weights["wte"], model_weights["wpe"]
    node_outputs: Dict[str, List[List[ArithRef]]] = {
        "emb": [
            [z3_real(wte[token][c]) + z3_real(wpe[position][c]) for c in range(d_model)]
            for position, token in enumerate(input_tokens)
        ]
    }

    for layer in range(n_layers):
        for head in range(n_heads):
            node = f"attn_{layer}_h_{head}"
            node_outputs[node] = (
                _encode_head(
                    input_tokens,
                    node_outputs,
                    layer,
                    head,
                    circuit_edges,
                    model_weights,
                    solver,
                    f"{ctx_prefix}_L{layer}_attn_h{head}",
                    trace,
                    assertion_profile,
                )
                if node in active
                else _zero(seq_len, head_dim)
            )

        mlp_node = f"mlp_{layer}"
        if mlp_node not in active:
            node_outputs[mlp_node] = _zero(seq_len, d_model)
            continue
        residual = _residual_for(
            mlp_node, node_outputs, circuit_edges, model_weights, seq_len
        )
        normalized = _normalize(
            residual,
            model_weights,
            f"mlp_{layer}_norm_gamma",
            f"mlp_{layer}_norm_beta",
            solver,
            f"{ctx_prefix}_L{layer}_mlp",
            trace,
            assertion_profile,
        )
        W_up = model_weights[f"mlp_{layer}_W_up"]
        b_up = model_weights[f"mlp_{layer}_b_up"]
        W_down = model_weights[f"mlp_{layer}_W_down"]
        b_down = model_weights[f"mlp_{layer}_b_down"]
        node_outputs[mlp_node] = []
        for position in range(seq_len):
            mlp_context = f"{ctx_prefix}_L{layer}_mlp_p{position}"
            before = len(solver.assertions())
            if trace is None:
                value = encode_mlp(
                    normalized[position], W_up, b_up, W_down, b_down
                )
            else:
                value = encode_mlp_with_trace(
                    normalized[position],
                    W_up,
                    b_up,
                    W_down,
                    b_down,
                    trace["leaky_relu"],
                    solver,
                    mlp_context,
                )
            record_solver_delta(assertion_profile, "mlp", solver, before)
            node_outputs[mlp_node].append(value)

    final = _residual_for(
        "logits", node_outputs, circuit_edges, model_weights, seq_len
    )[-1]
    norm_variant = model_weights.get("norm_variant", "layer_norm")
    if norm_variant == "none":
        normalized_final = final
    elif norm_variant == "signed_l1_band_norm":
        increment_norm_instances(assertion_profile)
        gamma = model_weights["final_norm_gamma"]
        beta = model_weights["final_norm_beta"]
        pos_fallback = [1.0 if index % 2 == 0 else 0.0 for index in range(d_model)]
        neg_fallback = [1.0 - value for value in pos_fallback]
        norm_context = f"{ctx_prefix}_logits_norm"
        arguments = (
            final,
            gamma,
            beta,
            model_weights["half_low"],
            model_weights["half_high"],
            pos_fallback,
            neg_fallback,
        )
        if trace is None:
            before = len(solver.assertions())
            normalized_final = encode_signed_l1_band_norm(
                *arguments, solver, norm_context
            )
        else:
            before = len(solver.assertions())
            normalized_final = encode_signed_l1_band_norm_with_trace(
                *arguments,
                trace["bandnorm"][norm_context],
                solver,
                norm_context,
            )
        record_solver_delta(assertion_profile, "norm", solver, before)
    else:
        raise ValueError(f"No exact SMT encoding for norm variant {norm_variant!r}")

    lm_head = model_weights["lm_head"]
    lm_bias = model_weights.get("lm_head_bias", [0.0] * len(lm_head))
    return {
        token: Sum(
            [z3_real(lm_head[token][coord]) * normalized_final[coord] for coord in range(d_model)]
        )
        + z3_real(lm_bias[token])
        for token in candidate_tokens
    }
