#!/usr/bin/env python3
"""
Debug helper for induction task predictions.

Prints detailed diagnostics for each induction example:
- Prompt
- Correct token
- Full model top-5 candidate predictions
- Correct token rank and logit
- Best wrong token logit
- Margin (correct - best_wrong)

Usage:
    python scripts/utils/debug_induction.py \
        --model_path artifacts/step2c-band-norm-sparsemax/checkpoint-240000 \
        --n_examples 32
"""

import argparse
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import from parent circuits directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'circuits'))
from extract_circuit import (
    generate_induction_examples,
    get_single_token_id,
    select_last_real_logits,
    load_model_with_variants,
)


def debug_induction_predictions(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    examples,
    candidate_token_ids,
    device: str,
    n: int = 32
):
    """Print detailed diagnostics for induction predictions.

    Args:
        model: GPT2 model
        tokenizer: Tokenizer
        examples: List of induction examples
        candidate_token_ids: List of candidate token IDs
        device: Device
        n: Number of examples to debug
    """
    prompts = [ex.prompt for ex in examples[:n]]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    last_logits = select_last_real_logits(logits, attention_mask)
    candidate_ids = torch.tensor(candidate_token_ids, device=device)
    candidate_logits = last_logits[:, candidate_ids]

    n_correct = 0
    margins = []

    for i, ex in enumerate(examples[:n]):
        correct_id = get_single_token_id(tokenizer, ex.correct_token)
        if correct_id is None or correct_id not in candidate_token_ids:
            print(f"Skipping example {i}: correct token not in candidates")
            continue

        top_vals, top_idx = torch.topk(candidate_logits[i], k=min(5, len(candidate_token_ids)))

        print("=" * 80)
        print("PROMPT:", repr(ex.prompt))
        print("CORRECT:", repr(ex.correct_token), f"id={correct_id}")

        print("TOP CANDIDATES:")
        for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
            tok_id = candidate_token_ids[idx]
            tok_text = tokenizer.decode([tok_id])
            marker = "<-- CORRECT" if tok_id == correct_id else ""
            print(f"  {rank}. {tok_text!r:12s} id={tok_id:5d} logit={val:+.4f} {marker}")

        correct_pos = candidate_token_ids.index(correct_id)
        correct_logit = candidate_logits[i, correct_pos].item()

        # Find best wrong token
        other_logits = torch.cat([
            candidate_logits[i, :correct_pos],
            candidate_logits[i, correct_pos + 1:]
        ])
        best_wrong = other_logits.max().item()
        margin = correct_logit - best_wrong

        print(f"MARGIN (correct - best_wrong): {margin:+.4f}")

        # Check if correct
        if correct_logit >= candidate_logits[i].max().item() - 1e-6:
            n_correct += 1
            print("STATUS: CORRECT")
        else:
            print("STATUS: WRONG")

        margins.append(margin)
        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Examples analyzed: {len(margins)}")
    print(f"Correct predictions: {n_correct} / {len(margins)} ({100 * n_correct / len(margins):.1f}%)")
    print(f"Mean margin: {sum(margins) / len(margins):+.4f}")
    print(f"Median margin: {sorted(margins)[len(margins) // 2]:+.4f}")
    print(f"Min margin: {min(margins):+.4f}")
    print(f"Max margin: {max(margins):+.4f}")

    # Margin distribution
    positive = sum(1 for m in margins if m > 0)
    print(f"\nMargin > 0 (correct rank): {positive} / {len(margins)} ({100 * positive / len(margins):.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Debug induction predictions")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--n_examples", type=int, default=32,
                        help="Number of examples to debug (default: 32)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_path}...")
    model = load_model_with_variants(args.model_path, device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Generating {args.n_examples} induction examples...")
    examples = generate_induction_examples(args.n_examples)

    # Get candidate token IDs
    token_pool = [
        " red", " blue", " green", " cat", " dog", " tree",
        " car", " book", " city", " river", " sun", " moon",
        " star", " bird", " fish", " lake", " hill", " road",
    ]
    candidate_token_ids = [get_single_token_id(tokenizer, t) for t in token_pool]
    candidate_token_ids = [tid for tid in candidate_token_ids if tid is not None]

    print(f"Candidate tokens: {len(candidate_token_ids)}")
    print()

    # Run diagnostics
    debug_induction_predictions(
        model, tokenizer, examples, candidate_token_ids, device, args.n_examples
    )


if __name__ == "__main__":
    main()
