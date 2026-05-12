#!/usr/bin/env python3
"""
Extract model weights for SMT verification.

Extracts weights from a trained small verifiable Transformer checkpoint
and saves them in the format expected by scripts/smt/circuit.py.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.small.config import SmallVerifiableConfig


def load_small_config(checkpoint_path: str) -> SmallVerifiableConfig:
    """Load SmallVerifiableConfig for a checkpoint directory.

    HuggingFace checkpoints also contain a config.json, so prefer the parent
    small-model config and only fall back to checkpoint-local candidates.
    """
    candidates = [
        os.path.join(os.path.dirname(checkpoint_path), "config.json"),
        os.path.join(checkpoint_path, "small_config.json"),
        os.path.join(checkpoint_path, "config.json"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            return SmallVerifiableConfig.load(path)
        except TypeError:
            continue

    raise FileNotFoundError(
        f"Could not find SmallVerifiableConfig for checkpoint: {checkpoint_path}"
    )


def extract_weights_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Extract weights from checkpoint in SMT-compatible format.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dict with weight arrays in format expected by SMT encoder
    """
    from scripts.small.train import create_small_model

    # Load config
    config = load_small_config(checkpoint_path)

    # Create model
    model = create_small_model(config)

    # Load weights
    weights_path_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
    weights_path_safetensors = os.path.join(checkpoint_path, "model.safetensors")

    if os.path.exists(weights_path_bin):
        state_dict = torch.load(weights_path_bin, map_location="cpu")
        model.load_state_dict(state_dict)
    elif os.path.exists(weights_path_safetensors):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path_safetensors)
            model.load_state_dict(state_dict)
        except ImportError:
            raise ImportError("safetensors not installed")
    else:
        raise FileNotFoundError(f"No weights found in {checkpoint_path}")

    model.eval()

    weights = {
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "vocab_size": config.vocab_size,
        "d_ff": config.d_mlp,
        "head_dim": config.d_model // config.n_heads,
        "norm_variant": config.norm_variant,
        "attn_variant": config.attn_variant,
        "activation_variant": config.activation_variant,
        "half_low": (config.norm_l1_low_per_dim * config.d_model) / 2.0,
        "half_high": (config.norm_l1_high_per_dim * config.d_model) / 2.0,
    }

    # Token and position embeddings
    weights["wte"] = model.transformer.wte.weight.detach().cpu().numpy().tolist()
    weights["wpe"] = model.transformer.wpe.weight.detach().cpu().numpy().tolist()

    # Extract layer weights
    for i in range(config.n_layers):
        block = model.transformer.h[i]

        # Attention weights
        # GPT-2 uses c_attn which combines Q, K, V projections
        d_model = config.d_model
        c_attn_weight = block.attn.c_attn.weight.detach().cpu().numpy()  # [d_model, 3*d_model]
        c_attn_bias = block.attn.c_attn.bias.detach().cpu().numpy()      # [3*d_model]

        weights[f"attn_{i}_W_q"] = c_attn_weight[:, :d_model].T.tolist()
        weights[f"attn_{i}_W_k"] = c_attn_weight[:, d_model:2*d_model].T.tolist()
        weights[f"attn_{i}_W_v"] = c_attn_weight[:, 2*d_model:3*d_model].T.tolist()

        weights[f"attn_{i}_b_q"] = c_attn_bias[:d_model].tolist()
        weights[f"attn_{i}_b_k"] = c_attn_bias[d_model:2*d_model].tolist()
        weights[f"attn_{i}_b_v"] = c_attn_bias[2*d_model:3*d_model].tolist()

        # Attention output projection
        c_proj_weight = block.attn.c_proj.weight.detach().cpu().numpy().T  # [d_model, d_model]
        c_proj_bias = block.attn.c_proj.bias.detach().cpu().numpy()

        weights[f"attn_{i}_W_o"] = c_proj_weight.tolist()
        weights[f"attn_{i}_b_o"] = c_proj_bias.tolist()

        # MLP weights
        c_fc_weight = block.mlp.c_fc.weight.detach().cpu().numpy().T  # [d_mlp, d_model]
        c_fc_bias = block.mlp.c_fc.bias.detach().cpu().numpy()

        weights[f"mlp_{i}_W_up"] = c_fc_weight.tolist()
        weights[f"mlp_{i}_b_up"] = c_fc_bias.tolist()

        c_proj_mlp_weight = block.mlp.c_proj.weight.detach().cpu().numpy().T  # [d_model, d_mlp]
        c_proj_mlp_bias = block.mlp.c_proj.bias.detach().cpu().numpy()

        weights[f"mlp_{i}_W_down"] = c_proj_mlp_weight.tolist()
        weights[f"mlp_{i}_b_down"] = c_proj_mlp_bias.tolist()

        # Layer norm 1 (pre-attention)
        ln1_gamma = block.ln_1.gamma.detach().cpu().numpy() if hasattr(block.ln_1, 'gamma') else block.ln_1.weight.detach().cpu().numpy()
        ln1_beta = block.ln_1.beta.detach().cpu().numpy() if hasattr(block.ln_1, 'beta') else block.ln_1.bias.detach().cpu().numpy()

        weights[f"attn_{i}_norm_gamma"] = ln1_gamma.tolist()
        weights[f"attn_{i}_norm_beta"] = ln1_beta.tolist()

        # Layer norm 2 (pre-MLP)
        ln2_gamma = block.ln_2.gamma.detach().cpu().numpy() if hasattr(block.ln_2, 'gamma') else block.ln_2.weight.detach().cpu().numpy()
        ln2_beta = block.ln_2.beta.detach().cpu().numpy() if hasattr(block.ln_2, 'beta') else block.ln_2.bias.detach().cpu().numpy()

        weights[f"mlp_{i}_norm_gamma"] = ln2_gamma.tolist()
        weights[f"mlp_{i}_norm_beta"] = ln2_beta.tolist()

    # Final layer norm
    final_ln_gamma = model.transformer.ln_f.gamma.detach().cpu().numpy() if hasattr(model.transformer.ln_f, 'gamma') else model.transformer.ln_f.weight.detach().cpu().numpy()
    final_ln_beta = model.transformer.ln_f.beta.detach().cpu().numpy() if hasattr(model.transformer.ln_f, 'beta') else model.transformer.ln_f.bias.detach().cpu().numpy()

    weights["final_norm_gamma"] = final_ln_gamma.tolist()
    weights["final_norm_beta"] = final_ln_beta.tolist()

    # LM head (unembedding)
    weights["lm_head"] = model.lm_head.weight.detach().cpu().numpy().tolist()

    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Extract model weights for SMT verification"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for weights JSON",
    )
    args = parser.parse_args()

    print(f"Extracting weights from: {args.checkpoint}")

    # Extract weights
    weights = extract_weights_from_checkpoint(args.checkpoint)

    # Save to JSON
    print(f"Saving weights to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"Done! Extracted {len(weights)} weight arrays")
    print(f"Sample keys: {list(weights.keys())[:5]}")


if __name__ == "__main__":
    main()
