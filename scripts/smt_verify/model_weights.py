"""Extract model weights for SMT encoding."""

import torch
import json
import os
import sys
from typing import Dict, Any
from types import MethodType

# Add scripts directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import GPT2LMHeadModel, GPT2Config
from train_experiment import apply_model_variants
from .helpers import get_norm_params, get_bandnorm_params


def load_model_weights(model_path: str, model_info_path: str = None) -> Dict[str, Any]:
    """Load trained model and extract weights for SMT encoding.

    Uses the same loading logic as extract_circuit.py.

    Args:
        model_path: Path to model checkpoint directory
        model_info_path: Path to model_info.json (optional, auto-detected if None)

    Returns:
        Dictionary of weight matrices and parameters
    """
    # Load model_info.json
    if model_info_path is None:
        model_info_path = os.path.join(model_path, "model_info.json")
        if not os.path.exists(model_info_path):
            parent_dir = os.path.dirname(model_path)
            model_info_path = os.path.join(parent_dir, "model_info.json")

    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
        norm_variant = model_info.get("norm_variant", "layernorm")
        attn_variant = model_info.get("attn_variant", "softmax")
        activation_variant = model_info.get("activation_variant", "gelu")
        print(f"Model variants: norm={norm_variant}, attn={attn_variant}, act={activation_variant}")
    else:
        raise FileNotFoundError(f"model_info.json not found at {model_info_path}")

    # Load config
    config = GPT2Config.from_pretrained(model_path)

    # Apply activation variant before model creation
    if activation_variant == "leaky_relu":
        config.activation_function = "leaky_relu"
    elif activation_variant == "relu":
        config.activation_function = "relu"

    # Create model
    model = GPT2LMHeadModel(config)

    # Apply variants BEFORE loading weights
    apply_model_variants(
        model,
        norm_variant=norm_variant,
        attn_variant=attn_variant,
        activation_variant=activation_variant,
    )

    # Load weights
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_path, "model.safetensors")

    if os.path.exists(weights_path):
        if weights_path.endswith(".bin"):
            state_dict = torch.load(weights_path, map_location="cpu")
        else:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)

        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Model weights not found in {model_path}")

    model.eval()

    # Extract weights
    weights = {
        "d_model": config.n_embd,
        "n_layers": config.n_layer,
        "n_heads": config.n_head,
        "vocab_size": config.vocab_size,
        "d_ff": config.n_inner if config.n_inner else 4 * config.n_embd,
        "head_dim": config.n_embd // config.n_head,
        "norm_variant": norm_variant,
        "attn_variant": attn_variant,
        "activation_variant": activation_variant,
    }

    # Token embeddings
    weights["wte"] = model.transformer.wte.weight.detach().cpu().numpy().tolist()

    # Position embeddings
    weights["wpe"] = model.transformer.wpe.weight.detach().cpu().numpy().tolist()
    weights["max_position_embeddings"] = config.n_positions

    # BandNorm parameters (if applicable)
    if norm_variant == "signed_l1_band_norm":
        bandnorm_params = get_bandnorm_params(config.n_embd)
        weights.update(bandnorm_params)

    # Extract layer weights
    for layer_idx in range(config.n_layer):
        layer = model.transformer.h[layer_idx]

        # Attention norm
        attn_norm = layer.ln_1
        gamma, beta = get_norm_params(attn_norm)
        weights[f"attn_{layer_idx}_norm_gamma"] = gamma.detach().cpu().numpy().tolist()
        weights[f"attn_{layer_idx}_norm_beta"] = beta.detach().cpu().numpy().tolist()

        # Attention weights
        attn = layer.attn

        # QKV projection (combined in c_attn)
        c_attn_weight = attn.c_attn.weight.detach().cpu().numpy()  # [3*d_model, d_model]
        c_attn_bias = attn.c_attn.bias.detach().cpu().numpy()  # [3*d_model]

        d_model = config.n_embd

        # Split Q, K, V (GPT-2 stores as [d_model, 3*d_model] transposed)
        W_qkv = c_attn_weight.T  # [d_model, 3*d_model]
        W_q = W_qkv[:, :d_model].tolist()
        W_k = W_qkv[:, d_model:2*d_model].tolist()
        W_v = W_qkv[:, 2*d_model:].tolist()

        b_qkv = c_attn_bias
        b_q = b_qkv[:d_model].tolist()
        b_k = b_qkv[d_model:2*d_model].tolist()
        b_v = b_qkv[2*d_model:].tolist()

        weights[f"attn_{layer_idx}_W_q"] = W_q
        weights[f"attn_{layer_idx}_W_k"] = W_k
        weights[f"attn_{layer_idx}_W_v"] = W_v
        weights[f"attn_{layer_idx}_b_q"] = b_q
        weights[f"attn_{layer_idx}_b_k"] = b_k
        weights[f"attn_{layer_idx}_b_v"] = b_v

        # Attention output projection
        c_proj_weight = attn.c_proj.weight.detach().cpu().numpy()  # [d_model, d_model]
        c_proj_bias = attn.c_proj.bias.detach().cpu().numpy()  # [d_model]

        W_o = c_proj_weight.T.tolist()  # [d_model, d_model]
        b_o = c_proj_bias.tolist()

        weights[f"attn_{layer_idx}_W_o"] = W_o
        weights[f"attn_{layer_idx}_b_o"] = b_o

        # MLP norm
        mlp_norm = layer.ln_2
        gamma, beta = get_norm_params(mlp_norm)
        weights[f"mlp_{layer_idx}_norm_gamma"] = gamma.detach().cpu().numpy().tolist()
        weights[f"mlp_{layer_idx}_norm_beta"] = beta.detach().cpu().numpy().tolist()

        # MLP weights
        mlp = layer.mlp
        W_up = mlp.c_fc.weight.detach().cpu().numpy().T.tolist()  # [d_ff, d_model]
        b_up = mlp.c_fc.bias.detach().cpu().numpy().tolist()  # [d_ff]
        W_down = mlp.c_proj.weight.detach().cpu().numpy().T.tolist()  # [d_model, d_ff]
        b_down = mlp.c_proj.bias.detach().cpu().numpy().tolist()  # [d_model]

        weights[f"mlp_{layer_idx}_W_up"] = W_up
        weights[f"mlp_{layer_idx}_b_up"] = b_up
        weights[f"mlp_{layer_idx}_W_down"] = W_down
        weights[f"mlp_{layer_idx}_b_down"] = b_down

    # Final layer norm
    final_norm = model.transformer.ln_f
    gamma, beta = get_norm_params(final_norm)
    weights["final_norm_gamma"] = gamma.detach().cpu().numpy().tolist()
    weights["final_norm_beta"] = beta.detach().cpu().numpy().tolist()

    # LM head
    weights["lm_head"] = model.lm_head.weight.detach().cpu().numpy().tolist()

    return weights
