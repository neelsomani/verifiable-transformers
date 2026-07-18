"""Exact controlled forward pass for per-head residual-stream circuits."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import torch
from transformers import GPT2LMHeadModel

from scripts.circuits.graph import CircuitGraph, parse_head_node


Edge = Tuple[str, str]


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    dtype = logits.dtype
    z = logits.float()
    z = torch.where(torch.isfinite(z), z, torch.full_like(z, -1e4))
    z = z - z.max(dim=dim, keepdim=True).values
    sorted_z = torch.sort(z, dim=dim, descending=True).values
    cumsum = torch.cumsum(sorted_z, dim=dim)
    size = z.size(dim)
    ranks = torch.arange(1, size + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = size
    ranks = ranks.view(shape)
    support = 1 + ranks * sorted_z > cumsum
    support_size = support.long().sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(dim, support_size - 1) - 1.0) / support_size.to(z.dtype)
    return torch.clamp(z - tau, min=0.0).to(dtype)


def _attention_variant(attention) -> str:
    explicit = getattr(attention, "_verifiable_attention_variant", None)
    if explicit is not None:
        return explicit
    function = getattr(attention.forward, "__func__", attention.forward)
    if "sparsemax" in getattr(function, "__name__", ""):
        return "sparsemax"
    return "softmax"


def _compute_head(
    block,
    normalized: torch.Tensor,
    head: int,
    extended_attention_mask: Optional[torch.Tensor],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return one head's value mixture before ``W_O``."""
    attention = block.attn
    if hasattr(attention, "programs"):
        batch, seq_len, _ = normalized.shape
        start = head * attention.head_dim
        stop = start + attention.head_dim
        value = torch.nn.functional.linear(
            normalized,
            attention.value_proj.weight[start:stop],
            attention.value_proj.bias[start:stop],
        )
        if head in attention.programs:
            weights = attention.programs[head].weights(
                input_ids, dtype=value.dtype
            )
            if extended_attention_mask is not None:
                valid = (extended_attention_mask[:, 0, 0, :] == 0).to(value.dtype)
                weights = weights * valid.unsqueeze(1)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return torch.matmul(weights, value)

        neural_index = attention.neural_heads.index(head)
        neural_start = neural_index * attention.head_dim
        neural_stop = neural_start + attention.head_dim
        query = torch.nn.functional.linear(
            normalized,
            attention.query_proj.weight[neural_start:neural_stop],
            attention.query_proj.bias[neural_start:neural_stop],
        )
        key = torch.nn.functional.linear(
            normalized,
            attention.key_proj.weight[neural_start:neural_stop],
            attention.key_proj.bias[neural_start:neural_stop],
        )
        scores = torch.matmul(query, key.transpose(-1, -2))
        if attention.scale_attn_weights:
            scores = scores / torch.full(
                [], value.size(-1) ** 0.5, dtype=scores.dtype, device=scores.device
            )
        if attention.scale_attn_by_inverse_layer_idx:
            scores = scores / float(attention.layer_idx + 1)
        causal = attention.bias[:, :, :seq_len, :seq_len].squeeze(0).squeeze(0)
        mask_value = torch.full([], -1e4, dtype=scores.dtype, device=scores.device)
        scores = torch.where(causal, scores, mask_value)
        if extended_attention_mask is not None:
            scores = scores + extended_attention_mask[:, 0, 0, :].unsqueeze(1)
        if attention.attention_variant == "sparsemax":
            weights = _sparsemax(scores, dim=-1)
        else:
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        return torch.matmul(attention.attn_dropout(weights), value)

    # GPT-2's Conv1D stores weights as [input, output]. Project only the
    # requested head instead of materializing all heads' Q/K/V three times.
    start = head * attention.head_dim
    stop = start + attention.head_dim
    split = attention.split_size
    query = torch.matmul(normalized, attention.c_attn.weight[:, start:stop])
    query = query + attention.c_attn.bias[start:stop]
    key = torch.matmul(
        normalized, attention.c_attn.weight[:, split + start : split + stop]
    )
    key = key + attention.c_attn.bias[split + start : split + stop]
    value = torch.matmul(
        normalized, attention.c_attn.weight[:, 2 * split + start : 2 * split + stop]
    )
    value = value + attention.c_attn.bias[2 * split + start : 2 * split + stop]
    _, seq_len, _ = query.shape

    scores = torch.matmul(query, key.transpose(-1, -2))
    if attention.scale_attn_weights:
        scores = scores / torch.full(
            [], value.size(-1) ** 0.5, dtype=scores.dtype, device=scores.device
        )
    if getattr(attention, "scale_attn_by_inverse_layer_idx", False):
        scores = scores / float(attention.layer_idx + 1)

    causal = attention.bias[:, :, :seq_len, :seq_len].squeeze(0).squeeze(0)
    mask_value = torch.full([], -1e4, dtype=scores.dtype, device=scores.device)
    scores = torch.where(causal, scores, mask_value)
    if extended_attention_mask is not None:
        scores = scores + extended_attention_mask[:, 0, 0, :].unsqueeze(1)

    if _attention_variant(attention) == "sparsemax":
        weights = _sparsemax(scores, dim=-1)
    else:
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
    weights = attention.attn_dropout(weights)
    return torch.matmul(weights, value)


def _build_residual(
    model: GPT2LMHeadModel,
    node_outputs: Dict[str, torch.Tensor],
    child: str,
    edges_to_keep: Set[Edge],
    graph: CircuitGraph,
    template: torch.Tensor,
) -> torch.Tensor:
    residual = torch.zeros_like(template)
    heads_by_layer: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)

    for parent, _ in graph.incoming_edges[child]:
        if (parent, child) not in edges_to_keep:
            continue
        parsed = parse_head_node(parent)
        if parsed is None:
            residual = residual + node_outputs[parent]
        else:
            layer, head = parsed
            heads_by_layer[layer][head] = node_outputs[parent]

    for layer, selected in heads_by_layer.items():
        attention = model.transformer.h[layer].attn
        head_values = []
        for head in range(graph.n_heads):
            if head in selected:
                head_values.append(selected[head])
            else:
                shape = (*template.shape[:-1], attention.head_dim)
                head_values.append(template.new_zeros(shape))
        concatenated = torch.cat(head_values, dim=-1)
        projected = attention.c_proj(concatenated)
        projected = attention.resid_dropout(projected)
        residual = residual + projected

    return residual


def controlled_forward(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    edges_to_keep: Set[Edge],
    graph: CircuitGraph,
    attention_mask: Optional[torch.Tensor] = None,
    return_node_outputs: bool = False,
    return_final_resid: bool = False,
):
    """Run GPT-2 with zero-ablation on residual edges and pre-``W_O`` heads."""
    if not graph.per_head:
        raise ValueError("The shared controlled forward requires a per-head graph")
    if graph.n_heads != model.config.n_head:
        raise ValueError(
            f"graph has {graph.n_heads} heads, model has {model.config.n_head}"
        )

    device = input_ids.device
    batch, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
    embedding = model.transformer.drop(
        model.transformer.wte(input_ids) + model.transformer.wpe(positions)
    )
    node_outputs: Dict[str, torch.Tensor] = {"emb": embedding}
    active_nodes = {"emb", "logits"}
    for source, target in edges_to_keep:
        active_nodes.update((source, target))

    extended_mask = None
    if attention_mask is not None:
        extended_mask = (1.0 - attention_mask[:, None, None, :].to(embedding.dtype))
        extended_mask = extended_mask * torch.finfo(embedding.dtype).min

    for layer in range(graph.n_layers):
        block = model.transformer.h[layer]
        for head in range(graph.n_heads):
            node = f"attn_{layer}_h_{head}"
            if node not in active_nodes:
                node_outputs[node] = embedding.new_zeros(
                    (*embedding.shape[:-1], block.attn.head_dim)
                )
                continue
            residual = _build_residual(
                model, node_outputs, node, edges_to_keep, graph, embedding
            )
            normalized = block.ln_1(residual)
            node_outputs[node] = _compute_head(
                block, normalized, head, extended_mask, input_ids
            )

        mlp_node = f"mlp_{layer}"
        if mlp_node not in active_nodes:
            node_outputs[mlp_node] = torch.zeros_like(embedding)
            continue
        residual = _build_residual(
            model, node_outputs, mlp_node, edges_to_keep, graph, embedding
        )
        node_outputs[mlp_node] = block.mlp(block.ln_2(residual))

    final_residual = _build_residual(
        model, node_outputs, "logits", edges_to_keep, graph, embedding
    )
    logits = model.lm_head(model.transformer.ln_f(final_residual))

    if return_node_outputs and return_final_resid:
        return logits, node_outputs, final_residual
    if return_node_outputs:
        return logits, node_outputs
    if return_final_resid:
        return logits, final_residual
    return logits


def controlled_forward_block(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    edges_to_keep: Set[Edge],
    graph: CircuitGraph,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation for regression against legacy block graphs."""
    if graph.per_head:
        raise ValueError("controlled_forward_block requires a block graph")

    device = input_ids.device
    batch, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
    embedding = model.transformer.drop(
        model.transformer.wte(input_ids) + model.transformer.wpe(positions)
    )
    outputs: Dict[str, torch.Tensor] = {"emb": embedding}

    extended_mask = None
    if attention_mask is not None:
        extended_mask = (1.0 - attention_mask[:, None, None, :].to(embedding.dtype))
        extended_mask = extended_mask * torch.finfo(embedding.dtype).min

    def residual_for(child: str) -> torch.Tensor:
        residual = torch.zeros_like(embedding)
        for parent, _ in graph.incoming_edges[child]:
            if (parent, child) in edges_to_keep:
                residual = residual + outputs[parent]
        return residual

    for layer in range(graph.n_layers):
        block = model.transformer.h[layer]
        attention_node = f"attn_{layer}"
        attention_input = block.ln_1(residual_for(attention_node))
        outputs[attention_node] = block.attn(
            attention_input,
            attention_mask=extended_mask,
            head_mask=None,
            layer_past=None,
            use_cache=False,
            output_attentions=False,
        )[0]
        mlp_node = f"mlp_{layer}"
        outputs[mlp_node] = block.mlp(block.ln_2(residual_for(mlp_node)))

    return model.lm_head(model.transformer.ln_f(residual_for("logits")))
