#!/usr/bin/env python3
"""
Interactive text generation with a trained verifiable transformer model.

Example usage:
    python scripts/generate_text.py --model_path artifacts/step2c-band-norm-sparsemax
"""

import argparse
import json
import os
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# Import custom norm/attention implementations from train_experiment
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_experiment import apply_model_variants


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Initial prompt (if not provided, enters interactive mode)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of generated sequence (includes prompt)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate (excluding prompt). Overrides max_length if specified.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (deterministic, picks most likely token). Disables sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random). Only used if --greedy is not set.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate",
    )
    args = parser.parse_args()

    # Load model info to get variant configurations
    # Try checkpoint dir first, then parent dir
    model_info_path = os.path.join(args.model_path, "model_info.json")
    if not os.path.exists(model_info_path):
        parent_dir = os.path.dirname(args.model_path)
        model_info_path = os.path.join(parent_dir, "model_info.json")

    if os.path.exists(model_info_path):
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        norm_variant = model_info.get("norm_variant", "layernorm")
        attn_variant = model_info.get("attn_variant", "softmax")
        activation_variant = model_info.get("activation_variant", "gelu")
        print(f"Model configuration (from {model_info_path}):")
        print(f"  Norm: {norm_variant}")
        print(f"  Attention: {attn_variant}")
        print(f"  Activation: {activation_variant}")
    else:
        print("Warning: model_info.json not found, assuming standard variants")
        norm_variant = "layernorm"
        attn_variant = "softmax"
        activation_variant = "gelu"

    print(f"\nLoading model from {args.model_path}...")

    # Load tokenizer (using standard GPT-2 tokenizer)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load config
    config = GPT2Config.from_pretrained(args.model_path)

    # Ensure activation variant is applied before model creation.
    # This matters because GPT2MLP reads config.activation_function in __init__.
    if activation_variant == "leaky_relu":
        config.activation_function = "leaky_relu"
    elif activation_variant == "relu":
        config.activation_function = "relu"
    # else keep whatever the checkpoint config says, usually gelu_new/gelu

    # Create model with correct architecture
    model = GPT2LMHeadModel(config)

    # Apply custom variants (norm, attention) BEFORE loading weights
    apply_model_variants(
        model,
        norm_variant=norm_variant,
        attn_variant=attn_variant,
        activation_variant=activation_variant
    )

    # Now load the trained weights
    weights_path = os.path.join(args.model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.model_path, "model.safetensors")

    if os.path.exists(weights_path):
        if weights_path.endswith(".bin"):
            state_dict = torch.load(weights_path, map_location="cpu")
        else:
            if not HAS_SAFETENSORS:
                raise ImportError("safetensors not installed. Install with: pip install safetensors")
            state_dict = load_safetensors(weights_path)

        # Handle tied weights (GPT-2 ties wte and lm_head)
        if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
            print("Note: lm_head.weight not in checkpoint, using tied weights from transformer.wte.weight")
            # Don't add it to state_dict, let model handle weight tying

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Could not find model weights in {args.model_path}")

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters\n")

    def generate(prompt_text):
        """Generate text from a prompt."""
        # Encode prompt
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        print(f"Prompt: {prompt_text}")
        print("-" * 80)

        # Generate
        # Determine length constraint
        gen_kwargs = {
            "attention_mask": attention_mask,
            "num_return_sequences": args.num_return_sequences,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Sampling vs greedy
        if args.greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_k"] = args.top_k
            gen_kwargs["top_p"] = args.top_p

        if args.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = args.max_new_tokens
        elif args.max_length is not None:
            gen_kwargs["max_length"] = args.max_length
        else:
            gen_kwargs["max_new_tokens"] = 50  # default

        with torch.no_grad():
            output_sequences = model.generate(input_ids, **gen_kwargs)

        # Decode and print results
        for i, sequence in enumerate(output_sequences):
            text = tokenizer.decode(sequence, skip_special_tokens=True)
            if args.num_return_sequences > 1:
                print(f"\n=== Generation {i+1} ===")
            print(text)
            print("-" * 80)

    # Run generation
    if args.prompt is not None:
        # Single prompt mode
        generate(args.prompt)
    else:
        # Interactive mode
        print("Interactive mode - enter prompts to generate text (Ctrl+C or 'quit' to exit)")
        print("=" * 80)
        while True:
            try:
                prompt_text = input("\nPrompt: ").strip()
                if not prompt_text or prompt_text.lower() in ["quit", "exit", "q"]:
                    print("Exiting...")
                    break
                generate(prompt_text)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == "__main__":
    main()
