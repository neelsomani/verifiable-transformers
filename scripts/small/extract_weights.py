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
    from scripts.programs import ProgrammedAttention, load_programs
    from scripts.small.extract import load_model

    # Load config
    config = load_small_config(checkpoint_path)

    # The shared loader installs program heads before loading their state dict.
    model = load_model(checkpoint_path, config, torch.device("cpu"))

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
    programs_path = os.path.join(checkpoint_path, "programs.json")
    if os.path.exists(programs_path):
        programs = load_programs(programs_path)
        weights["program_heads"] = {
            f"{layer}.{head}": program.to_dict()
            for (layer, head), program in sorted(programs.items())
        }
    else:
        weights["program_heads"] = {}

    # Token and position embeddings
    weights["wte"] = model.transformer.wte.weight.detach().cpu().numpy().tolist()
    weights["wpe"] = model.transformer.wpe.weight.detach().cpu().numpy().tolist()

    # Extract layer weights
    for i in range(config.n_layers):
        block = model.transformer.h[i]

        # Attention weights
        d_model = config.d_model
        if isinstance(block.attn, ProgrammedAttention):
            # Program rows deliberately have no Q/K parameters. Zero placeholders
            # keep the dense JSON shape stable; the SMT encoder never reads them.
            W_q = torch.zeros(d_model, d_model)
            W_k = torch.zeros(d_model, d_model)
            b_q = torch.zeros(d_model)
            b_k = torch.zeros(d_model)
            for neural_index, head in enumerate(block.attn.neural_heads):
                source = slice(
                    neural_index * block.attn.head_dim,
                    (neural_index + 1) * block.attn.head_dim,
                )
                target = slice(
                    head * block.attn.head_dim,
                    (head + 1) * block.attn.head_dim,
                )
                W_q[target] = block.attn.query_proj.weight[source].detach().cpu()
                W_k[target] = block.attn.key_proj.weight[source].detach().cpu()
                b_q[target] = block.attn.query_proj.bias[source].detach().cpu()
                b_k[target] = block.attn.key_proj.bias[source].detach().cpu()
            W_v = block.attn.value_proj.weight.detach().cpu()
            b_v = block.attn.value_proj.bias.detach().cpu()
            weights[f"attn_{i}_W_q"] = W_q.numpy().tolist()
            weights[f"attn_{i}_W_k"] = W_k.numpy().tolist()
            weights[f"attn_{i}_W_v"] = W_v.numpy().tolist()
            weights[f"attn_{i}_b_q"] = b_q.numpy().tolist()
            weights[f"attn_{i}_b_k"] = b_k.numpy().tolist()
            weights[f"attn_{i}_b_v"] = b_v.numpy().tolist()
        else:
            # GPT-2 Conv1D stores the combined projection as [input, output].
            c_attn_weight = block.attn.c_attn.weight.detach().cpu().numpy()
            c_attn_bias = block.attn.c_attn.bias.detach().cpu().numpy()
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
        if isinstance(block.ln_1, torch.nn.Identity):
            ln1_gamma = torch.ones(config.d_model).numpy()
            ln1_beta = torch.zeros(config.d_model).numpy()
        else:
            ln1_gamma = block.ln_1.gamma.detach().cpu().numpy() if hasattr(block.ln_1, 'gamma') else block.ln_1.weight.detach().cpu().numpy()
            ln1_beta = block.ln_1.beta.detach().cpu().numpy() if hasattr(block.ln_1, 'beta') else block.ln_1.bias.detach().cpu().numpy()

        weights[f"attn_{i}_norm_gamma"] = ln1_gamma.tolist()
        weights[f"attn_{i}_norm_beta"] = ln1_beta.tolist()

        # Layer norm 2 (pre-MLP)
        if isinstance(block.ln_2, torch.nn.Identity):
            ln2_gamma = torch.ones(config.d_model).numpy()
            ln2_beta = torch.zeros(config.d_model).numpy()
        else:
            ln2_gamma = block.ln_2.gamma.detach().cpu().numpy() if hasattr(block.ln_2, 'gamma') else block.ln_2.weight.detach().cpu().numpy()
            ln2_beta = block.ln_2.beta.detach().cpu().numpy() if hasattr(block.ln_2, 'beta') else block.ln_2.bias.detach().cpu().numpy()

        weights[f"mlp_{i}_norm_gamma"] = ln2_gamma.tolist()
        weights[f"mlp_{i}_norm_beta"] = ln2_beta.tolist()

    # Final layer norm
    if isinstance(model.transformer.ln_f, torch.nn.Identity):
        final_ln_gamma = torch.ones(config.d_model).numpy()
        final_ln_beta = torch.zeros(config.d_model).numpy()
    else:
        final_ln_gamma = model.transformer.ln_f.gamma.detach().cpu().numpy() if hasattr(model.transformer.ln_f, 'gamma') else model.transformer.ln_f.weight.detach().cpu().numpy()
        final_ln_beta = model.transformer.ln_f.beta.detach().cpu().numpy() if hasattr(model.transformer.ln_f, 'beta') else model.transformer.ln_f.bias.detach().cpu().numpy()

    weights["final_norm_gamma"] = final_ln_gamma.tolist()
    weights["final_norm_beta"] = final_ln_beta.tolist()

    # LM head (unembedding)
    weights["lm_head"] = model.lm_head.weight.detach().cpu().numpy().tolist()
    weights["lm_head_bias"] = (
        model.lm_head.bias.detach().cpu().numpy().tolist()
        if model.lm_head.bias is not None
        else [0.0] * config.vocab_size
    )

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
