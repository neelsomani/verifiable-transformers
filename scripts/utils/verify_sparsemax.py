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

    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)

    # Check 1: Verify patch is active
    print("1. Checking sparsemax patch is active...")
    print("   (Note: This check verifies the AttentionInterface registration is working)")

    # Import global counter from train_experiment
    import train_experiment
    train_experiment._sparsemax_call_count = 0

    dummy = torch.randint(0, model.config.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        model(dummy)

    sparsemax_calls = train_experiment._sparsemax_call_count

    if sparsemax_calls == 0:
        print("   ✗ FAILED: Sparsemax patch not active!")
        print("   This likely means AttentionInterface registration isn't being invoked correctly.")
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

    # Check 3: Attention weights contain exact zeros in allowed causal region
    print("\n3. Checking attention weights contain exact interior zeros (excluding masked positions)...")

    # Hook to capture attention weights
    attn_weights_captured = []

    def capture_hook(module, input, output):
        # In transformers 4.49, GPT2Attention output is (attn_output, present, attn_weights)
        # when output_attentions=True, where present is a tuple (key, value)
        if len(output) >= 3:
            attn_weights_captured.append(output[2])  # attn_weights is third element
        elif len(output) >= 2 and isinstance(output[1], torch.Tensor):
            attn_weights_captured.append(output[1])  # fallback for other formats

    # Register hook on first block
    hook = model.transformer.h[0].attn.register_forward_hook(capture_hook)

    with torch.no_grad():
        model(**inputs, output_attentions=True)

    hook.remove()

    if not attn_weights_captured:
        print("   ✗ FAILED: Could not capture attention weights")
        return False

    attn_weights = attn_weights_captured[0]  # Shape: [batch, heads, seq, seq]
    batch_size, num_heads, seq_len, _ = attn_weights.shape

    # Create causal mask: position i can only attend to positions 0...i
    # causal_mask[i, j] = True if j <= i (allowed positions)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=attn_weights.device))

    # Extract only the allowed causal region (exclude future-masked positions)
    allowed_weights = attn_weights[:, :, causal_mask]

    # Check for exact zeros in allowed region (not just masked positions)
    has_exact_zeros = (allowed_weights == 0.0).any().item()

    if not has_exact_zeros:
        print("   ✗ FAILED: No exact zeros in allowed causal region")
        print(f"      Min value in allowed region: {allowed_weights.min().item()}")
        print(f"      This suggests sparsemax is not producing sparse distributions")
        return False

    # Count zeros in allowed region
    num_zeros = (allowed_weights == 0.0).sum().item()
    total_allowed = allowed_weights.numel()
    zero_fraction = num_zeros / total_allowed

    print(f"   ✓ PASSED: Found exact zeros in allowed causal region")
    print(f"      Zeros: {num_zeros}/{total_allowed} ({zero_fraction:.2%})")

    # Check that not ALL allowed weights are zero (sanity check)
    if zero_fraction > 0.99:
        print("   ⚠ WARNING: >99% of allowed attention weights are zero (may indicate issue)")

    # Check 3b: Verify future positions are exactly zero (causal masking)
    print("\n3b. Checking future positions are exactly zero (causal masking)...")
    future_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=attn_weights.device),
        diagonal=1,
    )
    future_weights = attn_weights[:, :, future_mask]

    if (future_weights != 0).any().item():
        print("   ✗ FAILED: nonzero attention on future positions")
        print(f"      Max future weight: {future_weights.max().item()}")
        print(f"      This suggests causal masking is not working correctly")
        return False
    print("   ✓ PASSED: future positions are exactly zero")

    # Check 4: Verify attention weights sum to ~1.0 (sparsemax is projection onto simplex)
    print("\n4. Checking attention weight normalization...")
    attn_sums = attn_weights.sum(dim=-1)  # Sum over keys (includes masked positions as zeros)

    # For causal attention, only sum over allowed positions per row
    causal_sums = []
    for i in range(seq_len):
        # Row i can attend to positions 0...i
        row_sum = attn_weights[:, :, i, :i+1].sum(dim=-1)  # [batch, heads]
        causal_sums.append(row_sum)
    causal_sums = torch.stack(causal_sums, dim=2)  # [batch, heads, seq]

    min_sum = causal_sums.min().item()
    max_sum = causal_sums.max().item()
    mean_sum = causal_sums.mean().item()

    # Sparsemax projects onto probability simplex, so sums should be ~1.0
    if not (0.99 <= min_sum and max_sum <= 1.01):
        print(f"   ✗ FAILED: Attention sums not close to 1.0: [{min_sum:.4f}, {max_sum:.4f}], mean={mean_sum:.4f}")
        print(f"      Sparsemax should produce normalized probability distributions")
        return False
    print(f"   ✓ PASSED: Attention sums close to 1.0: [{min_sum:.4f}, {max_sum:.4f}], mean={mean_sum:.4f}")

    # Check 5: Verify bf16 path explicitly (critical for real training)
    print("\n5. Checking bf16 numerical stability...")
    if device.type == "cuda":
        model_bf16 = model.to(torch.bfloat16)

        with torch.no_grad():
            outputs_bf16 = model_bf16(**inputs)
            logits_bf16 = outputs_bf16.logits

        has_nan = torch.isnan(logits_bf16).any().item()
        has_inf = torch.isinf(logits_bf16).any().item()

        if has_nan or has_inf:
            print(f"   ✗ FAILED: bf16 produced NaN={has_nan}, Inf={has_inf}")
            return False
        print("   ✓ PASSED: bf16 produces valid outputs")
    else:
        print("   ⊘ SKIPPED: bf16 test requires CUDA device")

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

    # Apply model variants (normalization + sparsemax monkey-patch)
    apply_model_variants(model, norm_variant="layernorm", attn_variant="sparsemax", activation_variant="gelu")

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
