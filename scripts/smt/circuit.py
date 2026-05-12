"""SMT encoding of circuit forward pass with edge masking."""

from z3 import *
from typing import List, Dict, Set, Tuple, Any
from .encoders import (
    encode_leaky_relu,
    encode_signed_l1_band_norm,
    encode_sparsemax,
    encode_multihead_attention_sparsemax,
    encode_mlp,
    z3_real,
)


def zero_output(seq_len: int, d_model: int) -> List[List[ArithRef]]:
    """Create zero output for inactive nodes."""
    return [[RealVal(0) for _ in range(d_model)] for _ in range(seq_len)]


def encode_circuit_forward(
    input_tokens: List[int],
    circuit_edges: Set[Tuple[str, str]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    solver: Solver,
    ctx_prefix: str,
) -> Dict[int, ArithRef]:
    """Encode circuit forward pass with edge masking using SMT.

    Optimized to skip encoding inactive nodes (not on any path to logits).

    Args:
        input_tokens: Input token IDs
        circuit_edges: Set of retained edges (node_from, node_to)
        model_weights: Model weight parameters
        candidate_tokens: List of candidate output token IDs (to avoid full vocab)
        solver: Z3 solver to add constraints to
        ctx_prefix: Prefix for Z3 variables

    Returns:
        Dict mapping candidate token IDs to their logits
    """
    seq_len = len(input_tokens)
    d_model = model_weights["d_model"]
    n_layers = model_weights["n_layers"]
    n_heads = model_weights["n_heads"]

    # Compute active nodes from circuit edges
    active_nodes = {"emb", "logits"}
    for node_from, node_to in circuit_edges:
        active_nodes.add(node_from)
        active_nodes.add(node_to)

    # Node computation cache
    node_outputs = {}

    # Embedding: token + position
    wte = model_weights["wte"]
    wpe = model_weights["wpe"]

    emb_output = []
    for pos in range(seq_len):
        tok = input_tokens[pos]
        # Token embedding + position embedding
        emb_pos = [z3_real(wte[tok][j]) + z3_real(wpe[pos][j]) for j in range(d_model)]
        emb_output.append(emb_pos)

    node_outputs["emb"] = emb_output

    # Layer-by-layer forward pass (skip inactive nodes)
    for layer in range(n_layers):
        attn_node = f"attn_{layer}"
        mlp_node = f"mlp_{layer}"

        # Attention block
        if attn_node in active_nodes:
            attn_output = encode_attention_layer(
                node_outputs,
                layer,
                circuit_edges,
                model_weights,
                n_heads,
                solver,
                f"{ctx_prefix}_L{layer}_attn",
            )
            node_outputs[attn_node] = attn_output
        else:
            node_outputs[attn_node] = zero_output(seq_len, d_model)

        # MLP block
        if mlp_node in active_nodes:
            mlp_output = encode_mlp_layer(
                node_outputs,
                layer,
                circuit_edges,
                model_weights,
                solver,
                f"{ctx_prefix}_L{layer}_mlp",
            )
            node_outputs[mlp_node] = mlp_output
        else:
            node_outputs[mlp_node] = zero_output(seq_len, d_model)

    # Final logits (only for candidate tokens)
    logits = encode_logits_layer_candidates(
        node_outputs,
        n_layers,
        circuit_edges,
        model_weights,
        candidate_tokens,
        solver,
        f"{ctx_prefix}_logits",
    )

    return logits


def get_residual_input(
    node_outputs: Dict[str, Any],
    parent_nodes: List[str],
    child_node: str,
    circuit_edges: Set[Tuple[str, str]],
    seq_len: int,
    d_model: int,
) -> List[List[ArithRef]]:
    """Build residual stream input by summing parent outputs.

    Args:
        node_outputs: Cache of computed node outputs
        parent_nodes: List of potential parent nodes
        child_node: Current node name
        circuit_edges: Set of retained edges
        seq_len: Sequence length
        d_model: Model dimension

    Returns:
        Residual input [seq_len, d_model]
    """
    residual = [[RealVal(0) for _ in range(d_model)] for _ in range(seq_len)]

    for parent in parent_nodes:
        if (parent, child_node) in circuit_edges:
            parent_output = node_outputs[parent]
            for pos in range(seq_len):
                for j in range(d_model):
                    residual[pos][j] = residual[pos][j] + parent_output[pos][j]

    return residual


def encode_attention_layer(
    node_outputs: Dict[str, Any],
    layer: int,
    circuit_edges: Set[Tuple[str, str]],
    model_weights: Dict[str, Any],
    n_heads: int,
    solver: Solver,
    ctx_prefix: str,
) -> List[List[ArithRef]]:
    """Encode attention block with BandNorm and sparsemax.

    Args:
        node_outputs: Cached node outputs
        layer: Layer index
        circuit_edges: Retained edges
        model_weights: Model weights
        n_heads: Number of attention heads
        solver: Z3 solver
        ctx_prefix: Variable prefix

    Returns:
        Attention output [seq_len, d_model]
    """
    d_model = model_weights["d_model"]
    seq_len = len(node_outputs["emb"])

    # Get parent nodes (emb + all previous attn/mlp)
    parent_nodes = ["emb"]
    for prev_layer in range(layer):
        parent_nodes.extend([f"attn_{prev_layer}", f"mlp_{prev_layer}"])

    # Build residual input from parents
    residual = get_residual_input(
        node_outputs,
        parent_nodes,
        f"attn_{layer}",
        circuit_edges,
        seq_len,
        d_model,
    )

    # Pre-norm: BandNorm or LayerNorm
    norm_variant = model_weights.get("norm_variant", "layernorm")

    if norm_variant == "signed_l1_band_norm":
        # Get BandNorm parameters
        norm_gamma = model_weights[f"attn_{layer}_norm_gamma"]
        norm_beta = model_weights[f"attn_{layer}_norm_beta"]
        half_low = model_weights["half_low"]
        half_high = model_weights["half_high"]

        # Fallback masks (alternating pattern)
        pos_fallback = [1.0 if i % 2 == 0 else 0.0 for i in range(d_model)]
        neg_fallback = [0.0 if i % 2 == 0 else 1.0 for i in range(d_model)]

        normed = []
        for pos in range(seq_len):
            normed_pos = encode_signed_l1_band_norm(
                residual[pos],
                norm_gamma,
                norm_beta,
                half_low,
                half_high,
                pos_fallback,
                neg_fallback,
                solver,
                f"{ctx_prefix}_norm_p{pos}",
            )
            normed.append(normed_pos)
    else:
        # Simple pass-through for now (LayerNorm approximation)
        # For exact LayerNorm, would need variance computation
        norm_gamma = model_weights[f"attn_{layer}_norm_gamma"]
        norm_beta = model_weights[f"attn_{layer}_norm_beta"]

        normed = []
        for pos in range(seq_len):
            # Simplified: just affine transform (not exact LayerNorm)
            normed_pos = [residual[pos][i] * z3_real(norm_gamma[i]) + z3_real(norm_beta[i]) for i in range(d_model)]
            normed.append(normed_pos)

    # Attention: Q, K, V projections
    W_q = model_weights[f"attn_{layer}_W_q"]
    W_k = model_weights[f"attn_{layer}_W_k"]
    W_v = model_weights[f"attn_{layer}_W_v"]
    b_q = model_weights[f"attn_{layer}_b_q"]
    b_k = model_weights[f"attn_{layer}_b_k"]
    b_v = model_weights[f"attn_{layer}_b_v"]

    W_o = model_weights[f"attn_{layer}_W_o"]
    b_o = model_weights[f"attn_{layer}_b_o"]

    # Project to Q, K, V
    queries = []
    keys = []
    values = []
    for pos in range(seq_len):
        q = [Sum([z3_real(W_q[i][j]) * normed[pos][j] for j in range(d_model)]) + z3_real(b_q[i])
             for i in range(d_model)]
        k = [Sum([z3_real(W_k[i][j]) * normed[pos][j] for j in range(d_model)]) + z3_real(b_k[i])
             for i in range(d_model)]
        v = [Sum([z3_real(W_v[i][j]) * normed[pos][j] for j in range(d_model)]) + z3_real(b_v[i])
             for i in range(d_model)]
        queries.append(q)
        keys.append(k)
        values.append(v)

    # For each position, compute multi-head attention with sparsemax
    attn_output = []
    for pos in range(seq_len):
        # Causal mask: only attend to positions <= pos
        causal_keys = keys[:pos + 1]
        causal_values = values[:pos + 1]

        attn_pos = encode_multihead_attention_sparsemax(
            queries[pos],
            causal_keys,
            causal_values,
            n_heads,
            solver,
            f"{ctx_prefix}_p{pos}",
        )
        attn_output.append(attn_pos)

    # Output projection
    output = []
    for pos in range(seq_len):
        out_pos = [
            Sum([z3_real(W_o[i][j]) * attn_output[pos][j] for j in range(d_model)]) + z3_real(b_o[i])
            for i in range(d_model)
        ]
        output.append(out_pos)

    return output


def encode_mlp_layer(
    node_outputs: Dict[str, Any],
    layer: int,
    circuit_edges: Set[Tuple[str, str]],
    model_weights: Dict[str, Any],
    solver: Solver,
    ctx_prefix: str,
) -> List[List[ArithRef]]:
    """Encode MLP block with BandNorm and LeakyReLU.

    Args:
        node_outputs: Cached node outputs
        layer: Layer index
        circuit_edges: Retained edges
        model_weights: Model weights
        solver: Z3 solver
        ctx_prefix: Variable prefix

    Returns:
        MLP output [seq_len, d_model]
    """
    d_model = model_weights["d_model"]
    seq_len = len(node_outputs["emb"])

    # Get parent nodes (emb + all previous attn/mlp + current attn)
    parent_nodes = ["emb"]
    for prev_layer in range(layer):
        parent_nodes.extend([f"attn_{prev_layer}", f"mlp_{prev_layer}"])
    parent_nodes.append(f"attn_{layer}")

    # Build residual input
    residual = get_residual_input(
        node_outputs,
        parent_nodes,
        f"mlp_{layer}",
        circuit_edges,
        seq_len,
        d_model,
    )

    # Pre-norm: BandNorm or LayerNorm
    norm_variant = model_weights.get("norm_variant", "layernorm")

    if norm_variant == "signed_l1_band_norm":
        norm_gamma = model_weights[f"mlp_{layer}_norm_gamma"]
        norm_beta = model_weights[f"mlp_{layer}_norm_beta"]
        half_low = model_weights["half_low"]
        half_high = model_weights["half_high"]

        pos_fallback = [1.0 if i % 2 == 0 else 0.0 for i in range(d_model)]
        neg_fallback = [0.0 if i % 2 == 0 else 1.0 for i in range(d_model)]

        normed = []
        for pos in range(seq_len):
            normed_pos = encode_signed_l1_band_norm(
                residual[pos],
                norm_gamma,
                norm_beta,
                half_low,
                half_high,
                pos_fallback,
                neg_fallback,
                solver,
                f"{ctx_prefix}_norm_p{pos}",
            )
            normed.append(normed_pos)
    else:
        norm_gamma = model_weights[f"mlp_{layer}_norm_gamma"]
        norm_beta = model_weights[f"mlp_{layer}_norm_beta"]

        normed = []
        for pos in range(seq_len):
            normed_pos = [residual[pos][i] * z3_real(norm_gamma[i]) + z3_real(norm_beta[i]) for i in range(d_model)]
            normed.append(normed_pos)

    # MLP forward
    W_up = model_weights[f"mlp_{layer}_W_up"]
    b_up = model_weights[f"mlp_{layer}_b_up"]
    W_down = model_weights[f"mlp_{layer}_W_down"]
    b_down = model_weights[f"mlp_{layer}_b_down"]

    output = []
    for pos in range(seq_len):
        out_pos = encode_mlp(normed[pos], W_up, b_up, W_down, b_down)
        output.append(out_pos)

    return output


def encode_logits_layer_candidates(
    node_outputs: Dict[str, Any],
    n_layers: int,
    circuit_edges: Set[Tuple[str, str]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    solver: Solver,
    ctx_prefix: str,
) -> Dict[int, ArithRef]:
    """Encode final logits computation for candidate tokens only.

    Args:
        node_outputs: Cached node outputs
        n_layers: Number of transformer layers
        circuit_edges: Retained edges
        model_weights: Model weights
        candidate_tokens: List of candidate token IDs
        solver: Z3 solver
        ctx_prefix: Variable prefix

    Returns:
        Dict mapping token ID to logit value (only for candidates)
    """
    d_model = model_weights["d_model"]
    seq_len = len(node_outputs["emb"])

    # Get all parent nodes
    parent_nodes = ["emb"]
    for layer in range(n_layers):
        parent_nodes.extend([f"attn_{layer}", f"mlp_{layer}"])

    # Build residual input at last position only
    residual_last = [RealVal(0) for _ in range(d_model)]
    for parent in parent_nodes:
        if (parent, "logits") in circuit_edges:
            parent_output = node_outputs[parent]
            for j in range(d_model):
                residual_last[j] = residual_last[j] + parent_output[seq_len - 1][j]

    # Final norm
    norm_variant = model_weights.get("norm_variant", "layernorm")

    if norm_variant == "signed_l1_band_norm":
        norm_gamma = model_weights["final_norm_gamma"]
        norm_beta = model_weights["final_norm_beta"]
        half_low = model_weights["half_low"]
        half_high = model_weights["half_high"]

        pos_fallback = [1.0 if i % 2 == 0 else 0.0 for i in range(d_model)]
        neg_fallback = [0.0 if i % 2 == 0 else 1.0 for i in range(d_model)]

        normed = encode_signed_l1_band_norm(
            residual_last,
            norm_gamma,
            norm_beta,
            half_low,
            half_high,
            pos_fallback,
            neg_fallback,
            solver,
            f"{ctx_prefix}_norm",
        )
    else:
        norm_gamma = model_weights["final_norm_gamma"]
        norm_beta = model_weights["final_norm_beta"]

        normed = [residual_last[i] * z3_real(norm_gamma[i]) + z3_real(norm_beta[i]) for i in range(d_model)]

    # LM head projection - ONLY for candidate tokens
    lm_head = model_weights["lm_head"]
    logits = {}
    for tok in candidate_tokens:
        logits[tok] = Sum([z3_real(lm_head[tok][j]) * normed[j] for j in range(d_model)])

    return logits
