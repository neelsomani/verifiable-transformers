#!/usr/bin/env python3
"""
Circuit extraction for verifiable transformers.

Behavior viability scanning - verify model exhibits target behaviors
before attempting circuit extraction.

ACDC-style circuit extraction - find minimal subgraph responsible for
specific behaviors by iterative edge pruning.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set, Optional

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


@dataclass
class CircuitGraph:
    """Computational graph for circuit extraction."""
    nodes: List[str]  # Topological order
    all_edges: Set[Tuple[str, str]]
    incoming_edges: Dict[str, List[Tuple[str, str]]]  # child -> [(parent, child), ...]
    n_layers: int


def generate_quote_close_examples(n: int) -> List[BehaviorExample]:
    """Generate single vs double quote closing examples."""
    # Templates for single and double quote examples
    single_templates = [
        "x = 'hello world",
        "print('hello world",
        "message = 'foo bar",
        "return 'some text",
        "data.append('value",
        "name = 'alice",
        "s = 'test string",
        "key = 'item",
    ]

    double_templates = [
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
        # Single quote example
        examples.append(BehaviorExample(
            prompt=single_templates[i % len(single_templates)],
            correct_token="'",
            incorrect_token='"'
        ))
    for i in range(n // 2):
        # Double quote example
        examples.append(BehaviorExample(
            prompt=double_templates[i % len(double_templates)],
            correct_token='"',
            incorrect_token="'"
        ))
    return examples


def generate_bracket_type_examples(n: int) -> List[BehaviorExample]:
    """Generate bracket vs brace examples."""
    # Templates for bracket and brace examples
    bracket_templates = [
        "x = [a, b, c",
        "items = [foo, bar",
        "return [x, y, z",
        "data = [one, two",
        "arr = [p, q, r",
        "vals = [red, blue",
        "tmp = [left, right",
        "out = [first, second",
    ]

    brace_templates = [
        "x = {a, b, c",
        "items = {foo, bar",
        "return {x, y, z",
        "data = {one, two",
        "arr = {p, q, r",
        "vals = {red, blue",
        "tmp = {left, right",
        "out = {first, second",
    ]

    examples = []
    for i in range(n // 2):
        # Bracket example
        examples.append(BehaviorExample(
            prompt=bracket_templates[i % len(bracket_templates)],
            correct_token=']',
            incorrect_token='}'
        ))
    for i in range(n // 2):
        # Brace example
        examples.append(BehaviorExample(
            prompt=brace_templates[i % len(brace_templates)],
            correct_token='}',
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

        # A B C ... A B -> predict C
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


# ============================================================================
# Circuit Extraction
# ============================================================================

def build_circuit_graph(n_layers: int) -> CircuitGraph:
    """Build computational graph for circuit extraction.

    Nodes: emb, attn_0, mlp_0, ..., attn_{n_layers-1}, mlp_{n_layers-1}, logits

    Edges represent residual stream dependencies:
    - attn_i reads from: emb, attn_0, mlp_0, ..., attn_{i-1}, mlp_{i-1}
    - mlp_i reads from: emb, attn_0, mlp_0, ..., attn_i
    - logits reads from: all nodes
    """
    nodes = ["emb"]
    for i in range(n_layers):
        nodes.append(f"attn_{i}")
        nodes.append(f"mlp_{i}")
    nodes.append("logits")

    all_edges = set()
    incoming_edges = {node: [] for node in nodes}

    # Build edges following residual stream flow
    for i in range(n_layers):
        attn_node = f"attn_{i}"
        mlp_node = f"mlp_{i}"

        # attn_i reads from all previous components
        parents_for_attn = ["emb"]
        for j in range(i):
            parents_for_attn.extend([f"attn_{j}", f"mlp_{j}"])

        for parent in parents_for_attn:
            edge = (parent, attn_node)
            all_edges.add(edge)
            incoming_edges[attn_node].append(edge)

        # mlp_i reads from all previous components + current attn
        parents_for_mlp = parents_for_attn + [attn_node]
        for parent in parents_for_mlp:
            edge = (parent, mlp_node)
            all_edges.add(edge)
            incoming_edges[mlp_node].append(edge)

    # logits reads from all components
    parents_for_logits = ["emb"]
    for i in range(n_layers):
        parents_for_logits.extend([f"attn_{i}", f"mlp_{i}"])

    for parent in parents_for_logits:
        edge = (parent, "logits")
        all_edges.add(edge)
        incoming_edges["logits"].append(edge)

    return CircuitGraph(
        nodes=nodes,
        all_edges=all_edges,
        incoming_edges=incoming_edges,
        n_layers=n_layers
    )


def controlled_forward(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    edges_to_keep: Set[Tuple[str, str]],
    graph: CircuitGraph,
    corrupt_cache: Optional[Dict[str, torch.Tensor]] = None,
    return_node_outputs: bool = False,
):
    """Controlled forward pass with edge ablation.

    Args:
        model: GPT2 model
        input_ids: [B, T]
        attention_mask: [B, T], 1 for real tokens, 0 for padding
        edges_to_keep: Set of (parent, child) edges to use clean activations
        graph: Circuit graph
        corrupt_cache: If provided, use corrupted activations for removed edges
        return_node_outputs: If True, return node activations

    Returns:
        logits: [B, T, vocab]
        node_outputs: Dict[str, Tensor] if return_node_outputs, else None
    """
    device = input_ids.device
    B, T = input_ids.shape

    node_outputs = {}

    # Compute embedding
    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    token_embeds = model.transformer.wte(input_ids)
    pos_embeds = model.transformer.wpe(position_ids)
    emb = model.transformer.drop(token_embeds + pos_embeds)
    node_outputs["emb"] = emb

    # Build extended attention mask for GPT2
    # GPT2 expects [B, 1, 1, T] with -inf for padding
    if attention_mask is not None:
        extended_mask = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(emb.dtype).min
    else:
        extended_mask = None

    # Process each layer
    for i in range(graph.n_layers):
        block = model.transformer.h[i]
        attn_node = f"attn_{i}"
        mlp_node = f"mlp_{i}"

        # Build residual input for attention
        attn_parents = [p for p, c in graph.incoming_edges[attn_node]]
        resid_for_attn = torch.zeros_like(emb)
        for parent in attn_parents:
            if (parent, attn_node) in edges_to_keep:
                resid_for_attn = resid_for_attn + node_outputs[parent]
            elif corrupt_cache is not None:
                resid_for_attn = resid_for_attn + corrupt_cache[parent]
            # else: ablate to zero (no contribution)

        # Compute attention output
        x_norm = block.ln_1(resid_for_attn)
        attn_out = block.attn(
            x_norm,
            attention_mask=extended_mask,
            head_mask=None,
            layer_past=None,
            use_cache=False,
            output_attentions=False,
        )[0]
        node_outputs[attn_node] = attn_out

        # Build residual input for MLP
        mlp_parents = [p for p, c in graph.incoming_edges[mlp_node]]
        resid_for_mlp = torch.zeros_like(emb)
        for parent in mlp_parents:
            if (parent, mlp_node) in edges_to_keep:
                resid_for_mlp = resid_for_mlp + node_outputs[parent]
            elif corrupt_cache is not None:
                resid_for_mlp = resid_for_mlp + corrupt_cache[parent]

        # Compute MLP output
        x_norm = block.ln_2(resid_for_mlp)
        mlp_out = block.mlp(x_norm)
        node_outputs[mlp_node] = mlp_out

    # Build final residual for logits
    logit_parents = [p for p, c in graph.incoming_edges["logits"]]
    final_resid = torch.zeros_like(emb)
    for parent in logit_parents:
        if (parent, "logits") in edges_to_keep:
            final_resid = final_resid + node_outputs[parent]
        elif corrupt_cache is not None:
            final_resid = final_resid + corrupt_cache[parent]

    # Final layer norm and logits
    hidden = model.transformer.ln_f(final_resid)
    logits = model.lm_head(hidden)

    if return_node_outputs:
        return logits, node_outputs
    return logits


def select_last_real_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select logits at last real token position for each example.

    Args:
        logits: [B, T, vocab]
        attention_mask: [B, T]

    Returns:
        last_logits: [B, vocab]
    """
    idx = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_idx, idx, :]




def verify_controlled_forward_matches_native(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graph: CircuitGraph,
    atol: float = 1e-2,
):
    """Verify controlled_forward reproduces native model forward.

    This catches implementation bugs before circuit extraction.
    """
    with torch.no_grad():
        native_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        controlled_logits = controlled_forward(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            edges_to_keep=graph.all_edges,
            graph=graph,
            corrupt_cache=None,
            return_node_outputs=False,
        )

    native_last = select_last_real_logits(native_logits, attention_mask)
    controlled_last = select_last_real_logits(controlled_logits, attention_mask)

    max_abs_diff = (native_last - controlled_last).abs().max().item()
    print(f"Controlled forward max abs diff vs native: {max_abs_diff:.6g}")

    if max_abs_diff > atol:
        raise RuntimeError(
            f"controlled_forward does not match native model forward. "
            f"max_abs_diff={max_abs_diff:.6g}, atol={atol}. "
            "Do not trust circuit extraction until this is fixed."
        )


def compute_task_metrics(
    logits: torch.Tensor,
    examples: List[BehaviorExample],
    tokenizer: GPT2Tokenizer,
    attention_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute task-specific metrics.

    Args:
        logits: [B, T, vocab]
        examples: List of examples
        tokenizer: Tokenizer
        attention_mask: [B, T]

    Returns:
        Dict with binary_accuracy, mean_logit_diff
    """
    n_correct = 0
    logit_diffs = []

    # Get logits at last real position
    last_logits = select_last_real_logits(logits, attention_mask)

    for i, ex in enumerate(examples):
        correct_id = get_single_token_id(tokenizer, ex.correct_token)
        incorrect_id = get_single_token_id(tokenizer, ex.incorrect_token)

        if correct_id is None or incorrect_id is None:
            continue

        # Get logits for this example
        ex_logits = last_logits[i]  # [vocab]

        correct_logit = ex_logits[correct_id].item()
        incorrect_logit = ex_logits[incorrect_id].item()

        if correct_logit > incorrect_logit:
            n_correct += 1

        logit_diffs.append(correct_logit - incorrect_logit)

    n_used = len(logit_diffs)
    if n_used == 0:
        return {
            "binary_accuracy": 0.0,
            "mean_logit_diff": 0.0,
            "n_examples": 0,
        }

    return {
        "binary_accuracy": n_correct / n_used,
        "mean_logit_diff": sum(logit_diffs) / n_used,
        "n_examples": n_used,
    }


def compute_target_kl(
    full_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Compute KL(full || candidate) at target position only.

    Args:
        full_logits: [B, T, vocab] - reference distribution (full model)
        candidate_logits: [B, T, vocab] - candidate distribution (ablated model)
        attention_mask: [B, T]

    Returns:
        Mean KL(full || candidate) at last real position
    """
    # Select logits at target position
    p_logits = select_last_real_logits(full_logits, attention_mask)
    q_logits = select_last_real_logits(candidate_logits, attention_mask)

    # Compute KL(full || candidate)
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = log_p.exp()

    # KL(P || Q) = sum(P * (log P - log Q))
    kl = (p * (log_p - log_q)).sum(dim=-1)  # [B]
    return kl.mean().item()


def find_circuit(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    examples: List[BehaviorExample],
    graph: CircuitGraph,
    threshold: float,
    metric: str,
    device: str,
    verbose: bool = True,
) -> Tuple[Set[Tuple[str, str]], List[Dict]]:
    """Find minimal circuit using ACDC algorithm with zero ablation.

    Args:
        model: GPT2 model
        tokenizer: Tokenizer
        examples: Behavior examples
        graph: Circuit graph
        threshold: Threshold for edge removal
        metric: Metric to use ("kl" or "logit_diff")
        device: Device
        verbose: Print progress

    Returns:
        edges_to_keep: Set of edges in circuit
        edge_log: List of edge decisions
    """
    # Prepare prompts
    prompts = [ex.prompt for ex in examples]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Compute full model baseline (with zero ablation for removed edges)
    if verbose:
        print("Computing full model baseline...")
    edges_to_keep = set(graph.all_edges)
    with torch.no_grad():
        full_logits = controlled_forward(
            model,
            input_ids,
            attention_mask,
            edges_to_keep=edges_to_keep,
            graph=graph,
            corrupt_cache=None,  # Zero ablation
            return_node_outputs=False,
        )

    current_score = 0.0  # KL from full model (always 0 for full model)
    edge_log = []

    # ACDC: iterate in reverse topological order
    nodes_reversed = list(reversed(graph.nodes))

    for child in nodes_reversed:
        if child == "emb":
            continue  # Skip embedding node

        incoming = graph.incoming_edges[child]
        if verbose:
            print(f"\nProcessing node: {child} ({len(incoming)} incoming edges)")

        for edge in incoming:
            if edge not in edges_to_keep:
                continue

            # Try removing this edge (use zero ablation)
            candidate_edges = edges_to_keep - {edge}

            with torch.no_grad():
                candidate_logits = controlled_forward(
                    model,
                    input_ids,
                    attention_mask,
                    edges_to_keep=candidate_edges,
                    graph=graph,
                    corrupt_cache=None,  # Zero ablation
                    return_node_outputs=False,
                )

            # Compute metric change (KL from full model)
            if metric == "kl":
                candidate_score = compute_target_kl(full_logits, candidate_logits, attention_mask)
            else:  # logit_diff
                metrics = compute_task_metrics(candidate_logits, examples, tokenizer, attention_mask)
                candidate_score = -metrics["mean_logit_diff"]  # Negative because we want to minimize

            delta = candidate_score - current_score

            if delta < threshold:
                # Remove edge
                edges_to_keep.remove(edge)
                current_score = candidate_score
                decision = "removed"
                if verbose:
                    print(f"  [-] {edge[0]:12s} -> {edge[1]:12s} (delta={delta:.6f})")
            else:
                decision = "kept"
                if verbose:
                    print(f"  [+] {edge[0]:12s} -> {edge[1]:12s} (delta={delta:.6f})")

            edge_log.append({
                "edge": list(edge),
                "delta": delta,
                "decision": decision,
            })

    return edges_to_keep, edge_log


def trim_circuit(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    examples: List[BehaviorExample],
    graph: CircuitGraph,
    edges_to_keep: Set[Tuple[str, str]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    full_logits: torch.Tensor,
    threshold: float,
    trim_rounds: int,
    device: str,
    verbose: bool = True,
) -> Set[Tuple[str, str]]:
    """Additional greedy trimming pass with zero ablation.

    Args:
        Similar to find_circuit

    Returns:
        Trimmed edge set
    """
    with torch.no_grad():
        current_logits = controlled_forward(
            model, input_ids, attention_mask, edges_to_keep, graph, None
        )
    current_score = compute_target_kl(full_logits, current_logits, attention_mask)

    for round_idx in range(trim_rounds):
        removed_this_round = 0

        for edge in list(edges_to_keep):
            candidate_edges = edges_to_keep - {edge}

            with torch.no_grad():
                candidate_logits = controlled_forward(
                    model, input_ids, attention_mask, candidate_edges, graph, None
                )

            candidate_score = compute_target_kl(full_logits, candidate_logits, attention_mask)
            delta = candidate_score - current_score

            if delta < threshold:
                edges_to_keep.remove(edge)
                current_score = candidate_score
                removed_this_round += 1
                if verbose:
                    print(f"  Trim round {round_idx}: removed {edge}")

        if verbose:
            print(f"Trim round {round_idx}: removed {removed_this_round} edges")

        if removed_this_round == 0:
            break

    return edges_to_keep


def cleanup_graph(
    edges: Set[Tuple[str, str]],
    graph: CircuitGraph,
) -> Set[Tuple[str, str]]:
    """Remove edges not on any path from emb to logits."""
    # Find nodes reachable from emb
    reachable_from_emb = {"emb"}
    changed = True
    while changed:
        changed = False
        for parent, child in edges:
            if parent in reachable_from_emb and child not in reachable_from_emb:
                reachable_from_emb.add(child)
                changed = True

    # Find nodes that can reach logits
    can_reach_logits = {"logits"}
    changed = True
    while changed:
        changed = False
        for parent, child in edges:
            if child in can_reach_logits and parent not in can_reach_logits:
                can_reach_logits.add(parent)
                changed = True

    # Keep only edges on paths from emb to logits
    valid_nodes = reachable_from_emb & can_reach_logits
    cleaned_edges = {(p, c) for p, c in edges if p in valid_nodes and c in valid_nodes}

    return cleaned_edges


def write_circuit_outputs(
    output_dir: str,
    model_path: str,
    task: str,
    edges: Set[Tuple[str, str]],
    edge_log: List[Dict],
    graph: CircuitGraph,
    full_metrics: Dict,
    circuit_metrics: Dict,
    ablated_metrics: Dict,
    inverse_metrics: Dict,
    threshold: float,
    trim_rounds: int,
    n_examples: int,
):
    """Write circuit JSON, DOT, and summary."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON output
    circuit_data = {
        "model_path": model_path,
        "task": task,
        "metric": "kl",
        "threshold": threshold,
        "trim_rounds": trim_rounds,
        "n_examples": n_examples,
        "nodes": graph.nodes,
        "edges": [list(e) for e in sorted(edges)],
        "num_edges": len(edges),
        "scores": {
            "full": full_metrics,
            "circuit": circuit_metrics,
            "ablated": ablated_metrics,
            "inverse_ablation": inverse_metrics,
        },
        "edge_log": edge_log,
    }

    json_path = os.path.join(output_dir, "circuit.json")
    with open(json_path, "w") as f:
        json.dump(circuit_data, f, indent=2)
    print(f"Wrote circuit JSON to {json_path}")

    # DOT output
    dot_path = os.path.join(output_dir, "circuit.dot")
    with open(dot_path, "w") as f:
        f.write("digraph circuit {\n")
        f.write("  rankdir=LR;\n")
        for parent, child in sorted(edges):
            f.write(f'  "{parent}" -> "{child}";\n')
        f.write("}\n")
    print(f"Wrote circuit DOT to {dot_path}")

    # Summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CIRCUIT EXTRACTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Task: {task}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Trim rounds: {trim_rounds}\n\n")

        f.write("-" * 80 + "\n")
        f.write("FULL MODEL\n")
        f.write("-" * 80 + "\n")
        for k, v in full_metrics.items():
            f.write(f"{k:30s}: {v:.6f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write(f"EXTRACTED CIRCUIT ({len(edges)} edges)\n")
        f.write("-" * 80 + "\n")
        for k, v in circuit_metrics.items():
            f.write(f"{k:30s}: {v:.6f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("INVERSE ABLATION\n")
        f.write("-" * 80 + "\n")
        for k, v in inverse_metrics.items():
            f.write(f"{k:30s}: {v:.6f}\n")

    print(f"Wrote summary to {summary_path}")


def extract_circuit_for_task(
    model_path: str,
    task: str,
    n_examples: int,
    threshold: float,
    trim_rounds: int,
    output_dir: str,
    device: str,
):
    """Extract circuit for a specific task."""
    print(f"\n{'=' * 80}")
    print(f"EXTRACTING CIRCUIT FOR: {task}")
    print(f"{'=' * 80}\n")

    # Load model
    print("Loading model...")
    model = load_model_with_variants(model_path, device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Get task examples
    if task not in BEHAVIOR_GENERATORS:
        raise ValueError(f"Unknown task: {task}. Available: {list(BEHAVIOR_GENERATORS.keys())}")

    generator = BEHAVIOR_GENERATORS[task]
    examples = generator(n_examples)
    print(f"Generated {len(examples)} examples for task: {task}")

    # Build graph
    n_layers = model.config.n_layer
    graph = build_circuit_graph(n_layers)
    print(f"Built graph with {len(graph.nodes)} nodes and {len(graph.all_edges)} edges")

    # Verify controlled forward matches native model
    print("\nVerifying controlled forward implementation...")
    prompts = [ex.prompt for ex in examples]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    verify_controlled_forward_matches_native(model, input_ids, attention_mask, graph, atol=1e-2)
    print("Controlled forward matches native model.\n")

    # Run ACDC with zero ablation
    print(f"Running ACDC with zero ablation (threshold={threshold})...")
    edges_to_keep, edge_log = find_circuit(
        model, tokenizer, examples, graph, threshold, "kl", device, verbose=True
    )

    print(f"\nACDC complete. Remaining edges: {len(edges_to_keep)} / {len(graph.all_edges)}")

    # Compute full model logits for comparison
    with torch.no_grad():
        full_logits = controlled_forward(
            model, input_ids, attention_mask, graph.all_edges, graph, None
        )

    # Trimming pass
    if trim_rounds > 0:
        print(f"\nRunning {trim_rounds} trimming rounds...")
        edges_to_keep = trim_circuit(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            graph=graph,
            edges_to_keep=edges_to_keep,
            input_ids=input_ids,
            attention_mask=attention_mask,
            full_logits=full_logits,
            threshold=threshold,
            trim_rounds=trim_rounds,
            device=device,
            verbose=True,
        )
        print(f"After trimming: {len(edges_to_keep)} edges")

    # Cleanup
    print("\nCleaning up graph...")
    edges_to_keep = cleanup_graph(edges_to_keep, graph)
    print(f"After cleanup: {len(edges_to_keep)} edges")

    # Compute final metrics
    print("\nComputing final metrics...")

    with torch.no_grad():
        circuit_logits = controlled_forward(
            model, input_ids, attention_mask, edges_to_keep, graph, None
        )
        ablated_edges = {("emb", "logits")} if ("emb", "logits") in graph.all_edges else set()
        ablated_logits = controlled_forward(
            model, input_ids, attention_mask, ablated_edges, graph, None
        )
        inverse_edges = graph.all_edges - edges_to_keep
        inverse_logits = controlled_forward(
            model, input_ids, attention_mask, inverse_edges, graph, None
        )

    full_metrics = compute_task_metrics(full_logits, examples, tokenizer, attention_mask)
    circuit_metrics = compute_task_metrics(circuit_logits, examples, tokenizer, attention_mask)
    circuit_metrics["kl_from_full"] = compute_target_kl(full_logits, circuit_logits, attention_mask)

    ablated_metrics = compute_task_metrics(ablated_logits, examples, tokenizer, attention_mask)
    inverse_metrics = compute_task_metrics(inverse_logits, examples, tokenizer, attention_mask)
    inverse_metrics["kl_from_full"] = compute_target_kl(full_logits, inverse_logits, attention_mask)

    # Write outputs
    print("\nWriting outputs...")
    write_circuit_outputs(
        output_dir, model_path, task, edges_to_keep, edge_log, graph,
        full_metrics, circuit_metrics, ablated_metrics, inverse_metrics,
        threshold, trim_rounds, n_examples
    )

    print(f"\nCircuit extraction complete!")
    print(f"Full model accuracy: {full_metrics['binary_accuracy']:.3f}")
    print(f"Circuit accuracy: {circuit_metrics['binary_accuracy']:.3f}")
    print(f"Circuit KL from full: {circuit_metrics['kl_from_full']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Extract circuits from verifiable transformers")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")

    # Mode selection
    parser.add_argument("--scan_behaviors", action="store_true", help="Run behavior viability scan")
    parser.add_argument("--extract_circuit", type=str, default=None,
                        help="Extract circuit for task (quote_close, bracket_type, induction_ABCAB)")

    # Common args
    parser.add_argument("--n_examples", type=int, default=256, help="Number of examples per behavior")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="artifacts/circuits",
                        help="Output directory for reports")

    # Circuit extraction args
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Threshold for edge removal (KL divergence)")
    parser.add_argument("--trim_rounds", type=int, default=0,
                        help="Number of trimming rounds after ACDC")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.scan_behaviors:
        results = scan_behaviors(args.model_path, args.n_examples, args.batch_size, device)
        scan_output_dir = os.path.join(args.output_dir, "behavior_scan")
        write_behavior_scan_report(args.model_path, results, scan_output_dir)

    elif args.extract_circuit:
        task = args.extract_circuit
        task_output_dir = os.path.join(args.output_dir, task)
        extract_circuit_for_task(
            args.model_path, task, args.n_examples, args.threshold,
            args.trim_rounds, task_output_dir, device
        )

    else:
        print("No action specified.")
        print("Use --scan_behaviors to run behavior viability scan")
        print("Use --extract_circuit <task> to extract circuit for a specific task")
        parser.print_help()


if __name__ == "__main__":
    main()
