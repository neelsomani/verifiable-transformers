"""Concrete branch tracing for certified SMT encodings."""

from typing import Any, Dict, List, Set, Tuple
import math
import numpy as np


def sparsemax_np(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = np.where(np.isfinite(z), z, -1e4)
    z = z - np.max(z)
    sorted_z = np.sort(z)[::-1]
    cumsum = np.cumsum(sorted_z)
    r = np.arange(1, len(z) + 1, dtype=np.float64)
    support = 1 + r * sorted_z > cumsum
    k = max(int(np.sum(support)), 1)
    tau = (cumsum[k - 1] - 1.0) / k
    return np.maximum(z - tau, 0.0)


def project_trace(y: np.ndarray, radius: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    mass = float(np.sum(y))
    if mass <= radius:
        return y, {"needed": False, "support": [int(i) for i in np.where(y > 0)[0]]}

    sorted_y = np.sort(y)[::-1]
    cumsum = np.cumsum(sorted_y)
    r = np.arange(1, len(y) + 1, dtype=np.float64)
    candidates = (cumsum - radius) / r
    mask = sorted_y > candidates
    rho = max(int(np.sum(mask)) - 1, 0)
    tau = candidates[rho]
    projected = np.maximum(y - tau, 0.0)
    support = [int(i) for i in np.where(y > tau)[0]]
    return projected, {"needed": True, "support": support}


def lift_trace(
    y: np.ndarray,
    target: float,
    fallback: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mass = float(np.sum(y))
    if mass >= target:
        return y, {"needed": False, "active": [int(i) for i in np.where(y > 0)[0]], "fallback": False}

    active = (y > 0).astype(np.float64)
    fallback_used = bool(np.sum(active) < 1e-8)
    if fallback_used:
        active = fallback.astype(np.float64)
    active_count = max(float(np.sum(active)), 1.0)
    delta = (target - mass) / active_count
    lifted = y + delta * active
    return lifted, {
        "needed": True,
        "active": [int(i) for i in np.where(active > 0)[0]],
        "fallback": fallback_used,
    }


def bandnorm_trace(
    x: np.ndarray,
    gamma: List[float],
    beta: List[float],
    half_low: float,
    half_high: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(x, dtype=np.float64)
    c = x - np.mean(x)
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in c]
    p = np.maximum(c, 0.0)
    n = np.maximum(-c, 0.0)

    p_projected, p_proj_trace = project_trace(p, half_high)
    n_projected, n_proj_trace = project_trace(n, half_high)

    d = len(x)
    pos_fallback = np.zeros(d, dtype=np.float64)
    pos_fallback[0::2] = 1.0
    neg_fallback = 1.0 - pos_fallback

    p_lifted, p_lift_trace = lift_trace(p_projected, half_low, pos_fallback)
    n_lifted, n_lift_trace = lift_trace(n_projected, half_low, neg_fallback)

    z = p_lifted - n_lifted
    z = z - np.mean(z)
    y = z * np.asarray(gamma, dtype=np.float64) + np.asarray(beta, dtype=np.float64)

    return y, {
        "signs": signs,
        "pos_projection": p_proj_trace,
        "neg_projection": n_proj_trace,
        "pos_lift": p_lift_trace,
        "neg_lift": n_lift_trace,
    }


def trace_circuit_forward(
    input_tokens: List[int],
    circuit_edges: Set[Tuple[str, str]],
    model_weights: Dict[str, Any],
    ctx_prefix: str,
) -> Dict[str, Any]:
    """Trace sparsemax supports and BandNorm branches for one concrete input."""
    d_model = model_weights["d_model"]
    n_layers = model_weights["n_layers"]
    n_heads = model_weights["n_heads"]
    head_dim = d_model // n_heads
    seq_len = len(input_tokens)

    trace: Dict[str, Any] = {"bandnorm": {}, "sparsemax": {}, "leaky_relu": {}}
    active_nodes = {"emb", "logits"}
    for node_from, node_to in circuit_edges:
        active_nodes.add(node_from)
        active_nodes.add(node_to)

    node_outputs: Dict[str, np.ndarray] = {}
    wte = np.asarray(model_weights["wte"], dtype=np.float64)
    wpe = np.asarray(model_weights["wpe"], dtype=np.float64)
    node_outputs["emb"] = np.stack([wte[tok] + wpe[pos] for pos, tok in enumerate(input_tokens)])

    def residual_for(parent_nodes: List[str], child: str) -> np.ndarray:
        residual = np.zeros((seq_len, d_model), dtype=np.float64)
        for parent in parent_nodes:
            if (parent, child) in circuit_edges:
                residual += node_outputs[parent]
        return residual

    for layer in range(n_layers):
        attn_node = f"attn_{layer}"
        mlp_node = f"mlp_{layer}"

        if attn_node in active_nodes:
            parents = ["emb"]
            for prev_layer in range(layer):
                parents.extend([f"attn_{prev_layer}", f"mlp_{prev_layer}"])
            residual = residual_for(parents, attn_node)

            normed = []
            for pos in range(seq_len):
                norm_ctx = f"{ctx_prefix}_L{layer}_attn_norm_p{pos}"
                out, tr = bandnorm_trace(
                    residual[pos],
                    model_weights[f"attn_{layer}_norm_gamma"],
                    model_weights[f"attn_{layer}_norm_beta"],
                    model_weights["half_low"],
                    model_weights["half_high"],
                )
                trace["bandnorm"][norm_ctx] = tr
                normed.append(out)
            normed = np.stack(normed)

            W_q = np.asarray(model_weights[f"attn_{layer}_W_q"], dtype=np.float64)
            W_k = np.asarray(model_weights[f"attn_{layer}_W_k"], dtype=np.float64)
            W_v = np.asarray(model_weights[f"attn_{layer}_W_v"], dtype=np.float64)
            b_q = np.asarray(model_weights[f"attn_{layer}_b_q"], dtype=np.float64)
            b_k = np.asarray(model_weights[f"attn_{layer}_b_k"], dtype=np.float64)
            b_v = np.asarray(model_weights[f"attn_{layer}_b_v"], dtype=np.float64)
            W_o = np.asarray(model_weights[f"attn_{layer}_W_o"], dtype=np.float64)
            b_o = np.asarray(model_weights[f"attn_{layer}_b_o"], dtype=np.float64)

            queries = normed @ W_q.T + b_q
            keys = normed @ W_k.T + b_k
            values = normed @ W_v.T + b_v

            attn_output = []
            for pos in range(seq_len):
                out = np.zeros(d_model, dtype=np.float64)
                for h in range(n_heads):
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    scores = np.array([
                        float(np.dot(queries[pos, start:end], keys[k_pos, start:end]) / math.sqrt(head_dim))
                        for k_pos in range(pos + 1)
                    ])
                    weights = sparsemax_np(scores)
                    sm_ctx = f"{ctx_prefix}_L{layer}_attn_p{pos}_h{h}"
                    trace["sparsemax"][sm_ctx] = [int(i) for i in np.where(weights > 0)[0]]
                    out[start:end] = weights @ values[:pos + 1, start:end]
                attn_output.append(W_o @ out + b_o)
            node_outputs[attn_node] = np.stack(attn_output)
        else:
            node_outputs[attn_node] = np.zeros((seq_len, d_model), dtype=np.float64)

        if mlp_node in active_nodes:
            parents = ["emb"]
            for prev_layer in range(layer):
                parents.extend([f"attn_{prev_layer}", f"mlp_{prev_layer}"])
            parents.append(attn_node)
            residual = residual_for(parents, mlp_node)

            normed = []
            for pos in range(seq_len):
                norm_ctx = f"{ctx_prefix}_L{layer}_mlp_norm_p{pos}"
                out, tr = bandnorm_trace(
                    residual[pos],
                    model_weights[f"mlp_{layer}_norm_gamma"],
                    model_weights[f"mlp_{layer}_norm_beta"],
                    model_weights["half_low"],
                    model_weights["half_high"],
                )
                trace["bandnorm"][norm_ctx] = tr
                normed.append(out)
            normed = np.stack(normed)

            W_up = np.asarray(model_weights[f"mlp_{layer}_W_up"], dtype=np.float64)
            b_up = np.asarray(model_weights[f"mlp_{layer}_b_up"], dtype=np.float64)
            W_down = np.asarray(model_weights[f"mlp_{layer}_W_down"], dtype=np.float64)
            b_down = np.asarray(model_weights[f"mlp_{layer}_b_down"], dtype=np.float64)
            hidden_pre = normed @ W_up.T + b_up
            for pos in range(seq_len):
                for h in range(hidden_pre.shape[-1]):
                    trace["leaky_relu"][f"{ctx_prefix}_L{layer}_mlp_p{pos}_hidden_{h}"] = bool(
                        hidden_pre[pos, h] >= 0
                    )
            hidden = np.where(hidden_pre >= 0, hidden_pre, 0.01 * hidden_pre)
            node_outputs[mlp_node] = hidden @ W_down.T + b_down
        else:
            node_outputs[mlp_node] = np.zeros((seq_len, d_model), dtype=np.float64)

    parents = ["emb"]
    for layer in range(n_layers):
        parents.extend([f"attn_{layer}", f"mlp_{layer}"])
    final_residual = np.zeros(d_model, dtype=np.float64)
    for parent in parents:
        if (parent, "logits") in circuit_edges:
            final_residual += node_outputs[parent][seq_len - 1]

    norm_ctx = f"{ctx_prefix}_logits_norm"
    _, tr = bandnorm_trace(
        final_residual,
        model_weights["final_norm_gamma"],
        model_weights["final_norm_beta"],
        model_weights["half_low"],
        model_weights["half_high"],
    )
    trace["bandnorm"][norm_ctx] = tr
    trace["final_residual"] = final_residual.tolist()
    return trace
