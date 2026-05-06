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
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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
        default=100,
        help="Maximum length of generated sequence",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
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
    model_info_path = os.path.join(args.model_path, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        norm_variant = model_info.get("norm_variant", "layernorm")
        attn_variant = model_info.get("attn_variant", "softmax")
        activation_variant = model_info.get("activation_variant", "gelu")
        print(f"Model configuration:")
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

    # Load model
    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    # Apply custom variants
    apply_model_variants(
        model,
        norm_variant=norm_variant,
        attn_variant=attn_variant,
        activation_variant=activation_variant
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters\n")

    def generate(prompt_text):
        """Generate text from a prompt."""
        # Encode prompt
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        print(f"Prompt: {prompt_text}")
        print("-" * 80)

        # Generate
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

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
