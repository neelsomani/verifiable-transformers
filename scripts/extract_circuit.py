#!/usr/bin/env python3
"""
Circuit extraction for verifiable transformers.

Behavior viability scanning - verify model exhibits target behaviors
before attempting circuit extraction.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Import model variant loading from generate_text
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_experiment import apply_model_variants


@dataclass
class BehaviorExample:
    """Single example for a behavior test."""
    prompt: str
    correct_token: str
    incorrect_token: str


@dataclass
class BehaviorMetrics:
    """Metrics for evaluating a single behavior."""
    n_examples_requested: int
    n_examples_used: int
    n_skipped: int
    binary_accuracy: float
    mean_logit_diff: float
    mean_correct_logprob: float
    mean_incorrect_logprob: float
    mean_rank_of_correct_token: float
    viability: str  # "none", "viable", or "strong"
    token_ids: Dict[str, int]  # Maps token text to token ID


def generate_quote_close_examples(n: int) -> List[BehaviorExample]:
    """Generate single vs double quote closing examples."""
    templates_single = [
        "x = 'hello world",
        "print('hello world",
        "message = 'foo bar",
        "return 'some text",
        "data.append('value",
        "name = 'alice",
        "s = 'test string",
        "key = 'item",
    ]

    templates_double = [
        'x = "hello world',
        'print("hello world',
        'message = "foo bar',
        'return "some text',
        'data.append("value',
        'name = "alice',
        's = "test string',
        'key = "item',
    ]

    examples = []
    for i in range(n // 2):
        template = templates_single[i % len(templates_single)]
        examples.append(BehaviorExample(
            prompt=template,
            correct_token="'",
            incorrect_token='"'
        ))
    for i in range(n // 2):
        template = templates_double[i % len(templates_double)]
        examples.append(BehaviorExample(
            prompt=template,
            correct_token='"',
            incorrect_token="'"
        ))
    return examples


def generate_bracket_type_examples(n: int) -> List[BehaviorExample]:
    """Generate examples distinguishing [] vs {}."""
    list_templates = [
        "x = [1, 2, 3",
        "items = [foo, bar",
        "return [a, b, c",
        "values.append([x, y",
        "data = [10, 20, 30",
        "arr = [True, False",
        "nums = [7, 8, 9",
        "lst = [x, y, z",
    ]

    dict_templates = [
        'x = {"a": 1, "b": 2',
        "items = {foo: bar",
        "return {'x': a, 'y': b",
        "mapping.update({key: value",
        'data = {"key": 10',
        "d = {a: 1, b: 2",
        'config = {"name": "test"',
        "obj = {x: y, z: w",
    ]

    examples = []
    for i in range(n // 2):
        template = list_templates[i % len(list_templates)]
        examples.append(BehaviorExample(
            prompt=template,
            correct_token=']',
            incorrect_token='}'
        ))
    for i in range(n // 2):
        template = dict_templates[i % len(dict_templates)]
        examples.append(BehaviorExample(
            prompt=template,
            correct_token='}',
            incorrect_token=']'
        ))
    return examples


def generate_list_depth_examples(n: int) -> List[BehaviorExample]:
    """Generate nested vs flat list examples (bracket counting)."""
    flat_templates = [
        "x = [1, 2, 3",
        "values = [a, b, c",
        "return [foo, bar",
        "data = [10, 20, 30",
        "items = [x, y, z",
        "arr = [True, False",
        "nums = [7, 8",
        "lst = [p, q, r",
    ]

    nested_templates = [
        "x = [[1, 2, 3",
        "values = [[a, b, c",
        "return [[foo, bar",
        "data = [[10, 20, 30",
        "items = [[x, y, z",
        "arr = [[True, False",
        "nums = [[7, 8",
        "lst = [[p, q, r",
    ]

    examples = []
    for i in range(n // 2):
        template = flat_templates[i % len(flat_templates)]
        examples.append(BehaviorExample(
            prompt=template,
            correct_token=']',
            incorrect_token=']]'
        ))
    for i in range(n // 2):
        template = nested_templates[i % len(nested_templates)]
        examples.append(BehaviorExample(
            prompt=template,
            correct_token=']]',
            incorrect_token=']'
        ))
    return examples


def generate_induction_examples(n: int) -> List[BehaviorExample]:
    """Generate induction examples: A B C ... A B -> predict C."""
    examples = []

    # Use common single-token words (with leading space for GPT-2)
    token_pool = [
        " red", " blue", " green", " cat", " dog", " tree",
        " car", " book", " city", " river", " sun", " moon",
        " star", " bird", " fish", " lake", " hill", " road",
    ]

    for i in range(n):
        a = token_pool[i % len(token_pool)]
        b = token_pool[(i + 1) % len(token_pool)]
        c = token_pool[(i + 2) % len(token_pool)]
        wrong = token_pool[(i + 3) % len(token_pool)]

        # Pattern: A B C ... A B -> predict C
        prompt = f"{a}{b}{c} foo bar baz{a}{b}"

        examples.append(BehaviorExample(
            prompt=prompt,
            correct_token=c,
            incorrect_token=wrong,
        ))

    return examples


BEHAVIOR_GENERATORS = {
    'quote_close': generate_quote_close_examples,
    'bracket_type': generate_bracket_type_examples,
    'list_depth': generate_list_depth_examples,
    'induction_ABCAB': generate_induction_examples,
}


def get_single_token_id(tokenizer: GPT2Tokenizer, token_text: str) -> int | None:
    """Get token ID if text encodes to exactly one token, else None."""
    ids = tokenizer.encode(token_text, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return ids[0]


def load_model_with_variants(model_path: str, device: str):
    """Load model with custom variants applied."""
    # Try to load model_info.json
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
        print("Warning: model_info.json not found, using standard variants")
        norm_variant = "layernorm"
        attn_variant = "softmax"
        activation_variant = "gelu"

    # Load config and create model
    config = GPT2Config.from_pretrained(model_path)

    # Ensure activation variant is applied before model creation.
    # This matters because GPT2MLP reads config.activation_function in __init__.
    if activation_variant == "leaky_relu":
        config.activation_function = "leaky_relu"
    elif activation_variant == "relu":
        config.activation_function = "relu"
    # else keep whatever the checkpoint config says, usually gelu_new/gelu

    model = GPT2LMHeadModel(config)

    # Apply custom variants (norm, attention) BEFORE loading weights
    apply_model_variants(
        model,
        norm_variant=norm_variant,
        attn_variant=attn_variant,
        activation_variant=activation_variant
    )

    # Load weights
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_path, "model.safetensors")

    if os.path.exists(weights_path):
        if weights_path.endswith(".bin"):
            state_dict = torch.load(weights_path, map_location="cpu")
        else:
            try:
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
            except ImportError:
                raise ImportError("safetensors not installed")

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Could not find model weights in {model_path}")

    model = model.to(device)
    model.eval()
    return model


def evaluate_behavior(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    examples: List[BehaviorExample],
    batch_size: int,
    device: str
) -> BehaviorMetrics:
    """Evaluate model performance on a behavior."""

    n_correct = 0
    logit_diffs = []
    correct_logprobs = []
    incorrect_logprobs = []
    correct_ranks = []
    n_skipped = 0
    token_id_map = {}  # Track token IDs for report

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]

            # Tokenize prompts
            prompts = [ex.prompt for ex in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Get logits at last REAL token position (not pad)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            lengths = attention_mask.sum(dim=1) - 1  # Last real token index
            batch_indices = torch.arange(input_ids.size(0), device=device)
            logits = outputs.logits[batch_indices, lengths, :]  # [batch, vocab]

            # Compute metrics for each example in batch
            for j, ex in enumerate(batch):
                # Get token IDs with single-token validation
                correct_id = get_single_token_id(tokenizer, ex.correct_token)
                incorrect_id = get_single_token_id(tokenizer, ex.incorrect_token)

                if correct_id is None or incorrect_id is None:
                    n_skipped += 1
                    continue

                # Track token IDs for reporting
                token_id_map[ex.correct_token] = correct_id
                token_id_map[ex.incorrect_token] = incorrect_id

                # Logits
                correct_logit = logits[j, correct_id].item()
                incorrect_logit = logits[j, incorrect_id].item()

                # Binary accuracy
                if correct_logit > incorrect_logit:
                    n_correct += 1

                # Logit difference
                logit_diffs.append(correct_logit - incorrect_logit)

                # Log probabilities
                log_probs = torch.log_softmax(logits[j], dim=-1)
                correct_logprobs.append(log_probs[correct_id].item())
                incorrect_logprobs.append(log_probs[incorrect_id].item())

                # Rank of correct token (optimized)
                correct_logit_tensor = logits[j, correct_id]
                rank = (logits[j] > correct_logit_tensor).sum().item() + 1
                correct_ranks.append(rank)

    # Compute aggregate metrics
    n_used = len(logit_diffs)
    if n_used == 0:
        # All examples were skipped
        return BehaviorMetrics(
            n_examples_requested=len(examples),
            n_examples_used=0,
            n_skipped=n_skipped,
            binary_accuracy=0.0,
            mean_logit_diff=0.0,
            mean_correct_logprob=0.0,
            mean_incorrect_logprob=0.0,
            mean_rank_of_correct_token=0.0,
            viability="none",
            token_ids=token_id_map
        )

    binary_accuracy = n_correct / n_used
    mean_logit_diff = sum(logit_diffs) / n_used
    mean_correct_logprob = sum(correct_logprobs) / n_used
    mean_incorrect_logprob = sum(incorrect_logprobs) / n_used
    mean_rank = sum(correct_ranks) / n_used

    # Determine viability
    if binary_accuracy >= 0.85 and mean_logit_diff >= 1.0:
        viability = "strong"
    elif binary_accuracy >= 0.70 and mean_logit_diff > 0.0:
        viability = "viable"
    else:
        viability = "none"

    return BehaviorMetrics(
        n_examples_requested=len(examples),
        n_examples_used=n_used,
        n_skipped=n_skipped,
        binary_accuracy=binary_accuracy,
        mean_logit_diff=mean_logit_diff,
        mean_correct_logprob=mean_correct_logprob,
        mean_incorrect_logprob=mean_incorrect_logprob,
        mean_rank_of_correct_token=mean_rank,
        viability=viability,
        token_ids=token_id_map
    )


def scan_behaviors(
    model_path: str,
    n_examples: int,
    batch_size: int,
    device: str
) -> Dict[str, BehaviorMetrics]:
    """Scan all behaviors and return metrics."""

    print(f"Loading model from {model_path}...")
    model = load_model_with_variants(model_path, device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\nScanning {len(BEHAVIOR_GENERATORS)} behaviors with {n_examples} examples each...")

    results = {}
    for behavior_name, generator in BEHAVIOR_GENERATORS.items():
        print(f"\nEvaluating {behavior_name}...")
        examples = generator(n_examples)
        metrics = evaluate_behavior(model, tokenizer, examples, batch_size, device)
        results[behavior_name] = metrics

        print(f"  Requested: {metrics.n_examples_requested}, Used: {metrics.n_examples_used}, Skipped: {metrics.n_skipped}")
        print(f"  Token IDs: {metrics.token_ids}")
        print(f"  Accuracy: {metrics.binary_accuracy:.3f}")
        print(f"  Logit diff: {metrics.mean_logit_diff:.3f}")
        print(f"  Viability: {metrics.viability}")

    return results


def write_behavior_scan_report(
    model_path: str,
    results: Dict[str, BehaviorMetrics],
    output_dir: str
):
    """Write JSON and text reports."""

    os.makedirs(output_dir, exist_ok=True)

    # Write JSON
    json_data = {
        "model_path": model_path,
        "results": {
            name: {
                "n_examples_requested": m.n_examples_requested,
                "n_examples_used": m.n_examples_used,
                "n_skipped": m.n_skipped,
                "token_ids": m.token_ids,
                "binary_accuracy": round(m.binary_accuracy, 4),
                "mean_logit_diff": round(m.mean_logit_diff, 4),
                "mean_correct_logprob": round(m.mean_correct_logprob, 4),
                "mean_incorrect_logprob": round(m.mean_incorrect_logprob, 4),
                "mean_rank_of_correct_token": round(m.mean_rank_of_correct_token, 2),
                "viability": m.viability
            }
            for name, m in results.items()
        }
    }

    json_path = os.path.join(output_dir, "behavior_scan.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nWrote JSON report to {json_path}")

    # Write text report
    txt_path = os.path.join(output_dir, "behavior_scan.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BEHAVIOR VIABILITY SCAN\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n\n")

        for name, m in results.items():
            f.write("-" * 80 + "\n")
            f.write(f"{name.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Examples requested:   {m.n_examples_requested}\n")
            f.write(f"Examples used:        {m.n_examples_used}\n")
            f.write(f"Examples skipped:     {m.n_skipped}\n")
            f.write(f"Token IDs:            {m.token_ids}\n")
            f.write(f"Binary accuracy:      {m.binary_accuracy:.4f}\n")
            f.write(f"Mean logit diff:      {m.mean_logit_diff:+.4f}\n")
            f.write(f"Mean correct logprob: {m.mean_correct_logprob:+.4f}\n")
            f.write(f"Mean incorrect logprob: {m.mean_incorrect_logprob:+.4f}\n")
            f.write(f"Mean rank (correct):  {m.mean_rank_of_correct_token:.2f}\n")
            f.write(f"Viability:            {m.viability.upper()}\n\n")

        # Summary
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        strong = [name for name, m in results.items() if m.viability == "strong"]
        viable = [name for name, m in results.items() if m.viability == "viable"]
        weak = [name for name, m in results.items() if m.viability == "none"]

        f.write(f"Strongly viable: {len(strong)}\n")
        for name in strong:
            f.write(f"  - {name}\n")

        f.write(f"\nViable: {len(viable)}\n")
        for name in viable:
            f.write(f"  - {name}\n")

        f.write(f"\nNot viable: {len(weak)}\n")
        for name in weak:
            f.write(f"  - {name}\n")

    print(f"Wrote text report to {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract circuits from verifiable transformers")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--scan_behaviors", action="store_true", help="Run behavior viability scan")
    parser.add_argument("--n_examples", type=int, default=256, help="Number of examples per behavior")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="artifacts/circuits/behavior_scan",
                        help="Output directory for reports")
    parser.add_argument("--force_extract", action="store_true",
                        help="Force circuit extraction even for non-viable behaviors")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.scan_behaviors:
        results = scan_behaviors(args.model_path, args.n_examples, args.batch_size, device)
        write_behavior_scan_report(args.model_path, results, args.output_dir)
    else:
        print("No action specified. Use --scan_behaviors to run behavior scan.")
        parser.print_help()


if __name__ == "__main__":
    main()
