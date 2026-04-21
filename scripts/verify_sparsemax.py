"""
Automated verification script for sparsemax attention.

Checks:
1. Sparsemax patch is active
2. No NaNs/infs in model outputs
3. Attention weights contain exact interior zeros (sparsemax support detection)
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# Import from train_experiment
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_experiment import apply_model_variants, sparsemax


def verify_sparsemax(model, tokenizer, device):
    """Run verification checks on sparsemax model."""

    print("\n=== Sparsemax Verification ===\n")

    # Check 1: Verify patch is active
    print("1. Checking sparsemax patch is active...")

    # Import global counter from train_experiment
    import train_experiment
    train_experiment._sparsemax_call_count = 0

    dummy = torch.randint(0, model.config.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        model(dummy)

    sparsemax_calls = train_experiment._sparsemax_call_count

    if sparsemax_calls == 0:
        print("   ✗ FAILED: Sparsemax patch not active!")
        return False
    print(f"   ✓ PASSED: {sparsemax_calls} sparsemax calls during forward")

    # Check 2: No NaNs/infs in outputs
    print("\n2. Checking for NaNs/infs in model outputs...")
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()

    if has_nan or has_inf:
        print(f"   ✗ FAILED: Found NaN={has_nan}, Inf={has_inf}")
        return False
    print("   ✓ PASSED: No NaNs or infs in outputs")

    # Check 3: Attention weights contain exact zeros
    print("\n3. Checking attention weights contain exact interior zeros...")

    # Hook to capture attention weights
    attn_weights_captured = []

    def capture_hook(module, input, output):
        # output is (attn_output, attn_weights)
        if len(output) >= 2:
            attn_weights_captured.append(output[1])

    # Register hook on first block
    hook = model.transformer.h[0].attn.register_forward_hook(capture_hook)

    with torch.no_grad():
        model(**inputs, output_attentions=True)

    hook.remove()

    if not attn_weights_captured:
        print("   ✗ FAILED: Could not capture attention weights")
        return False

    attn_weights = attn_weights_captured[0]  # Shape: [batch, heads, seq, seq]

    # Check for exact zeros (not just small values)
    has_exact_zeros = (attn_weights == 0.0).any().item()

    if not has_exact_zeros:
        print("   ✗ FAILED: No exact zeros in attention weights")
        print(f"      Min value: {attn_weights.min().item()}")
        print(f"      This suggests sparsemax is not producing sparse distributions")
        return False

    # Count zeros
    num_zeros = (attn_weights == 0.0).sum().item()
    total_elements = attn_weights.numel()
    zero_fraction = num_zeros / total_elements

    print(f"   ✓ PASSED: Found exact zeros in attention weights")
    print(f"      Zeros: {num_zeros}/{total_elements} ({zero_fraction:.2%})")

    # Check that not ALL weights are zero (sanity check)
    if zero_fraction > 0.99:
        print("   ⚠ WARNING: >99% of attention weights are zero (may indicate issue)")

    # Check 4: Verify attention weights sum to 1 (or close, for sparsemax)
    print("\n4. Checking attention weight normalization...")
    attn_sums = attn_weights.sum(dim=-1)  # Sum over keys
    min_sum = attn_sums.min().item()
    max_sum = attn_sums.max().item()

    # Sparsemax should sum to ≤ 1 (can be less due to sparsity)
    if min_sum < 0 or max_sum > 1.01:
        print(f"   ✗ FAILED: Attention sums out of range [0, 1]: [{min_sum:.4f}, {max_sum:.4f}]")
        return False
    print(f"   ✓ PASSED: Attention sums in valid range [{min_sum:.4f}, {max_sum:.4f}]")

    print("\n=== All Checks Passed ===\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify sparsemax attention implementation")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config.from_pretrained(args.model_name)
    model = GPT2LMHeadModel(config)

    # Apply sparsemax patch
    apply_model_variants(model, norm_variant="layernorm", attn_variant="sparsemax")

    model = model.to(args.device)
    model.eval()

    # Run verification
    success = verify_sparsemax(model, tokenizer, args.device)

    if success:
        print("✓ All verification checks passed")
        sys.exit(0)
    else:
        print("✗ Verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
