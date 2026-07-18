#!/usr/bin/env python3
"""
Circuit extraction for small verifiable Transformer.

Extracts minimal subgraphs responsible for each task using ACDC-style
edge pruning with exhaustive evaluation domains.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

# Import small verifiable components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.small import (
    get_eval_dataset,
    vocab,
    VOCAB_SIZE,
)
from scripts.small.config import SmallVerifiableConfig
from scripts.circuits import CircuitGraph, controlled_forward


# The graph and controlled forward pass are shared with the GPT-2 extractor.
# Attention nodes are named ``attn_<layer>_h_<head>`` and store pre-W_O values.


def compute_candidate_kl(
    full_logits: torch.Tensor,
    circuit_logits: torch.Tensor,
    candidate_ids: List[int],
) -> float:
    """Compute KL(full || circuit) restricted to candidate tokens.

    Args:
        full_logits: [B, T, vocab] - reference distribution (full model)
        circuit_logits: [B, T, vocab] - candidate distribution (circuit)
        candidate_ids: List of candidate token IDs

    Returns:
        Mean KL divergence at final position over candidate tokens
    """
    # Select logits at final position
    p_logits_full = full_logits[:, -1, :]  # [B, vocab]
    q_logits_full = circuit_logits[:, -1, :]  # [B, vocab]

    # Extract candidate token logits
    candidate_ids_tensor = torch.tensor(candidate_ids, device=p_logits_full.device)
    p_logits = p_logits_full[:, candidate_ids_tensor]  # [B, |T|]
    q_logits = q_logits_full[:, candidate_ids_tensor]  # [B, |T|]

    # Compute KL(full || circuit) on restricted distribution
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = log_p.exp()

    # KL(P || Q) = sum(P * (log P - log Q))
    kl = (p * (log_p - log_q)).sum(dim=-1)  # [B]
    return kl.mean().item()


def compute_projected_agreement(
    full_logits: torch.Tensor,
    circuit_logits: torch.Tensor,
    candidate_ids: List[int],
) -> float:
    """Compute projected-decision agreement: Pr[d_T(C_E,x) = d_T(M,x)].

    Args:
        full_logits: [B, T, vocab] - reference distribution (full model)
        circuit_logits: [B, T, vocab] - candidate distribution (circuit)
        candidate_ids: List of candidate token IDs

    Returns:
        Fraction of examples where argmax over candidate tokens agrees
    """
    # Select logits at final position
    p_logits_full = full_logits[:, -1, :]  # [B, vocab]
    q_logits_full = circuit_logits[:, -1, :]  # [B, vocab]

    # Extract candidate token logits
    candidate_ids_tensor = torch.tensor(candidate_ids, device=p_logits_full.device)
    p_logits = p_logits_full[:, candidate_ids_tensor]  # [B, |T|]
    q_logits = q_logits_full[:, candidate_ids_tensor]  # [B, |T|]

    # Compute argmax over candidate tokens
    p_argmax = p_logits.argmax(dim=-1)  # [B]
    q_argmax = q_logits.argmax(dim=-1)  # [B]

    # Compute agreement
    agreement = (p_argmax == q_argmax).float().mean().item()
    return agreement


def evaluate_circuit(
    model: GPT2LMHeadModel,
    task_name: str,
    circuit_edges: Set[Tuple[str, str]],
    graph: CircuitGraph,
    device: torch.device = None,
) -> Dict:
    """
    Evaluate a circuit on the task's exhaustive domain.

    Args:
        model: The model to evaluate
        task_name: Task name ("quote_close", "bracket_type")
        circuit_edges: Set of edges to keep (all others are ablated)
        graph: Circuit graph
        device: Device to run on

    Returns:
        Dictionary with metrics:
        - candidate_accuracy: Accuracy on valid candidates
        - agreement: Fraction of examples matching full model
        - mean_candidate_kl: Mean KL divergence from full model on candidates
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    examples = get_eval_dataset(task_name)
    candidates = sorted(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task_name]))

    # Batch all examples
    input_ids_list = [example["input_ids"] for example in examples]
    targets = [example["target"] for example in examples]

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)  # [N, T]

    with torch.no_grad():
        # Full model logits
        full_logits = controlled_forward(model, input_ids, graph.get_edges(), graph)

        # Circuit logits
        circuit_logits = controlled_forward(model, input_ids, circuit_edges, graph)

    # Compute candidate accuracy
    final_logits = circuit_logits[:, -1, :]  # [N, vocab]
    candidate_logits = final_logits[:, candidates]  # [N, |candidates|]
    predicted = candidate_logits.argmax(dim=-1)  # [N]
    predicted_tokens = [candidates[idx] for idx in predicted.cpu().tolist()]

    correct = sum(pred == target for pred, target in zip(predicted_tokens, targets))
    accuracy = correct / len(targets)

    # Compute agreement with full model
    agreement = compute_projected_agreement(full_logits, circuit_logits, candidates)

    # Compute KL divergence
    mean_kl = compute_candidate_kl(full_logits, circuit_logits, candidates)

    return {
        "candidate_accuracy": accuracy,
        "agreement": agreement,
        "mean_candidate_kl": mean_kl,
    }


# ============================================================================
# Circuit Extraction
# ============================================================================


def verify_controlled_forward(
    model: GPT2LMHeadModel,
    task_name: str,
    graph: CircuitGraph,
    device: torch.device,
    atol: float = 1e-4,
):
    """Verify controlled_forward reproduces native model forward.

    This catches implementation bugs before circuit extraction.
    """
    examples = get_eval_dataset(task_name)
    input_ids_list = [example["input_ids"] for example in examples]
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)

    with torch.no_grad():
        # Native forward
        native_outputs = model(input_ids=input_ids)
        native_logits = native_outputs.logits

        # Controlled forward with all edges
        controlled_logits = controlled_forward(
            model, input_ids, graph.all_edges, graph
        )

    # Compare final position logits
    native_final = native_logits[:, -1, :]
    controlled_final = controlled_logits[:, -1, :]

    max_abs_diff = (native_final - controlled_final).abs().max().item()
    print(f"Controlled forward max abs diff vs native: {max_abs_diff:.6g}")

    if max_abs_diff > atol:
        raise RuntimeError(
            f"controlled_forward does not match native model forward. "
            f"max_abs_diff={max_abs_diff:.6g}, atol={atol}. "
            "Do not trust circuit extraction until this is fixed."
        )


def find_circuit(
    model: GPT2LMHeadModel,
    task_name: str,
    graph: CircuitGraph,
    threshold: float,
    min_agreement: float,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[Set[Tuple[str, str]], List[Dict]]:
    """Find minimal circuit using ACDC algorithm with zero ablation.

    Args:
        model: GPT2 model
        task_name: Task name
        graph: Circuit graph
        threshold: Threshold for edge removal (KL divergence)
        min_agreement: Minimum projected-agreement required
        device: Device
        verbose: Print progress

    Returns:
        edges_to_keep: Set of edges in circuit
        edge_log: List of edge decisions
    """
    model.eval()

    examples = get_eval_dataset(task_name)
    candidates = sorted(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task_name]))

    # Batch all examples
    input_ids_list = [example["input_ids"] for example in examples]
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)  # [N, T]

    # Verify controlled forward matches native
    if verbose:
        print("Verifying controlled forward implementation...")
    verify_controlled_forward(model, task_name, graph, device, atol=1e-4)
    if verbose:
        print("Controlled forward matches native model.\n")

    # Compute full model baseline
    if verbose:
        print("Computing full model baseline...")
    edges_to_keep = set(graph.all_edges)

    with torch.no_grad():
        full_logits = controlled_forward(model, input_ids, edges_to_keep, graph)

    current_kl = 0.0  # KL from full model (always 0 for full model)
    edge_log = []

    # ACDC: iterate in reverse topological order
    nodes_reversed = list(reversed(graph.nodes))

    for child in nodes_reversed:
        if child == "emb":
            continue  # Skip embedding node

        incoming = sorted(graph.incoming_edges[child])
        if verbose:
            print(f"\nProcessing node: {child} ({len(incoming)} incoming edges)")

        for edge in incoming:
            if edge not in edges_to_keep:
                continue

            # Try removing this edge
            candidate_edges = edges_to_keep - {edge}

            with torch.no_grad():
                candidate_logits = controlled_forward(
                    model, input_ids, candidate_edges, graph
                )

            # Compute projected-agreement
            agreement = compute_projected_agreement(full_logits, candidate_logits, candidates)

            # Compute KL change
            candidate_kl = compute_candidate_kl(full_logits, candidate_logits, candidates)
            delta = candidate_kl - current_kl

            # Apply projected-agreement guard and threshold
            if agreement >= min_agreement and delta < threshold:
                # Remove edge
                edges_to_keep.remove(edge)
                current_kl = candidate_kl
                decision = "removed"
                if verbose:
                    print(f"  [-] {edge[0]:12s} -> {edge[1]:12s} (delta={delta:.6f}, agreement={agreement:.3f})")
            else:
                decision = "kept"
                reason = ""
                if agreement < min_agreement:
                    reason = f" [agreement={agreement:.3f} < {min_agreement}]"
                if verbose:
                    print(f"  [+] {edge[0]:12s} -> {edge[1]:12s} (delta={delta:.6f}, agreement={agreement:.3f}){reason}")

            edge_log.append({
                "edge": list(edge),
                "delta": delta,
                "agreement": agreement,
                "decision": decision,
            })

    return edges_to_keep, edge_log


def cleanup_graph(
    edges: Set[Tuple[str, str]],
    graph: CircuitGraph,
) -> Set[Tuple[str, str]]:
    """Remove only subgraphs that cannot influence logits structurally.

    Do not require reachability from ``emb``: GPT attention and MLP modules have
    biases, so a node with a zero residual input can still emit a nonzero value.
    Every remaining removal is checked behaviorally by ``projected_trim``.
    """
    can_reach_logits = {"logits"}
    changed = True
    while changed:
        changed = False
        for parent, child in edges:
            if child in can_reach_logits and parent not in can_reach_logits:
                can_reach_logits.add(parent)
                changed = True

    cleaned_edges = {
        (parent, child)
        for parent, child in edges
        if parent in can_reach_logits and child in can_reach_logits
    }

    return cleaned_edges


def projected_trim(
    model: GPT2LMHeadModel,
    task_name: str,
    edges: Set[Tuple[str, str]],
    graph: CircuitGraph,
    min_agreement: float,
    device: torch.device,
) -> Tuple[Set[Tuple[str, str]], List[Dict]]:
    """Remove every edge unnecessary for the projected decision domain.

    ACDC ranks removals with candidate KL, while the formal edge-necessity
    property is decision-based. This deterministic fixed-point pass aligns the
    emitted circuit with that acceptance metric without changing search ranking.
    """
    examples = get_eval_dataset(task_name)
    candidates = sorted(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task_name]))
    input_ids = torch.tensor(
        [example["input_ids"] for example in examples],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        full_logits = controlled_forward(model, input_ids, graph.get_edges(), graph)

    trimmed = cleanup_graph(set(edges), graph)
    log = []
    changed = True
    while changed:
        changed = False
        for edge in sorted(trimmed):
            candidate = cleanup_graph(trimmed - {edge}, graph)
            with torch.no_grad():
                logits = controlled_forward(model, input_ids, candidate, graph)
            agreement = compute_projected_agreement(
                full_logits, logits, candidates
            )
            removed = agreement >= min_agreement
            log.append(
                {
                    "edge": list(edge),
                    "agreement": agreement,
                    "decision": "removed" if removed else "kept",
                    "stage": "projected_trim",
                }
            )
            if removed:
                trimmed = candidate
                changed = True
                break
    return trimmed, log


def extract_circuit_threshold(
    model: GPT2LMHeadModel,
    task_name: str,
    config: SmallVerifiableConfig,
    threshold: float,
    min_agreement: float = 1.0,
    device: torch.device = None,
) -> Tuple[CircuitGraph, Dict, Set[Tuple[str, str]], List[Dict]]:
    """
    Extract circuit using ACDC with zero ablation.

    Args:
        model: Model to extract from
        task_name: Task name
        config: Model config
        threshold: KL divergence threshold for edge removal
        min_agreement: Minimum agreement with full model
        device: Device to run on

    Returns:
        (circuit_graph, metrics, edges_to_keep, edge_log)
    """
    if device is None:
        device = next(model.parameters()).device

    # Create full graph
    graph = CircuitGraph(
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        per_head=True,
    )

    print(f"Starting circuit extraction for {task_name}")
    print(f"  Full graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"  Threshold: {threshold}")
    print(f"  Min agreement: {min_agreement}")

    # Run ACDC
    edges_to_keep, edge_log = find_circuit(
        model, task_name, graph, threshold, min_agreement, device, verbose=True
    )

    print(f"\nACDC complete. Remaining edges: {len(edges_to_keep)} / {len(graph.edges)}")

    # Cleanup
    print("\nCleaning up graph...")
    edges_to_keep = cleanup_graph(edges_to_keep, graph)
    print(f"After cleanup: {len(edges_to_keep)} edges")

    print("\nTrimming edges unnecessary for projected agreement...")
    edges_to_keep, trim_log = projected_trim(
        model,
        task_name,
        edges_to_keep,
        graph,
        min_agreement,
        device,
    )
    edge_log.extend(trim_log)
    print(f"After projected trim: {len(edges_to_keep)} edges")

    # Evaluate final circuit
    metrics = evaluate_circuit(model, task_name, edges_to_keep, graph, device=device)

    print(f"\nExtracted circuit for {task_name}:")
    print(f"  Edges: {len(edges_to_keep)} / {len(graph.edges)}")
    print(f"  Candidate accuracy: {metrics['candidate_accuracy']:.4f}")
    print(f"  Agreement with full model: {metrics['agreement']:.4f}")
    print(f"  Mean candidate KL: {metrics['mean_candidate_kl']:.6f}")

    return graph, metrics, edges_to_keep, edge_log


def threshold_sweep(
    model: GPT2LMHeadModel,
    task_name: str,
    config: SmallVerifiableConfig,
    thresholds: List[float],
    min_agreement: float = 1.0,
    device: torch.device = None,
) -> List[Dict]:
    """
    Sweep over thresholds and extract circuits.

    Returns:
        List of results, one per threshold
    """
    results = []

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Threshold: {threshold}")
        print(f"{'='*60}")

        graph, metrics, edges_to_keep, edge_log = extract_circuit_threshold(
            model, task_name, config, threshold, min_agreement, device
        )

        # Create circuit dict with extracted edges
        circuit_dict = {
            "task": task_name,
            "n_layers": graph.n_layers,
            "n_heads": graph.n_heads,
            "granularity": "head",
            "nodes": graph.nodes,
            "edges": [{"source": s, "target": t} for s, t in sorted(edges_to_keep)],
            "num_edges": len(edges_to_keep),
            "ablation": "zero",
            "metric": "candidate_kl",
            "min_agreement": min_agreement,
            "threshold": threshold,
        }

        result = {
            "threshold": threshold,
            "n_edges": len(edges_to_keep),
            "metrics": metrics,
            "circuit": circuit_dict,
            "edge_log": edge_log,
        }
        results.append(result)

    return results


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract circuit from small verifiable Transformer"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["quote_close", "bracket_type"],
        help="Task to extract circuit for",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="candidate_kl",
        choices=["candidate_kl"],
        help="Metric to use for circuit selection (default: candidate_kl)",
    )
    parser.add_argument(
        "--min_agreement",
        type=float,
        default=1.0,
        help="Minimum agreement with full model (default: 1.0 = perfect)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="zero",
        choices=["zero"],
        help="Ablation type (only 'zero' supported for now)",
    )
    parser.add_argument(
        "--threshold_sweep",
        action="store_true",
        help="Perform threshold sweep",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
        help="Thresholds to sweep over",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for circuit",
    )
    return parser.parse_args()


def load_model(model_path: str, config: SmallVerifiableConfig, device: torch.device) -> GPT2LMHeadModel:
    """Load model with support for both pytorch_model.bin and model.safetensors."""
    from scripts.small.train import create_small_model
    from scripts.programs import install_program_heads, load_programs

    model = create_small_model(config)

    programs_path = os.path.join(model_path, "programs.json")
    if os.path.exists(programs_path):
        programs = load_programs(programs_path)
        install_program_heads(
            model, programs, attention_variant=config.attn_variant
        )
        print(f"Installed {len(programs)} program heads from {programs_path}")

    # Try pytorch_model.bin first, then model.safetensors
    weights_path_bin = os.path.join(model_path, "pytorch_model.bin")
    weights_path_safetensors = os.path.join(model_path, "model.safetensors")

    if os.path.exists(weights_path_bin):
        print(f"Loading weights from {weights_path_bin}")
        checkpoint = torch.load(weights_path_bin, map_location="cpu")
        model.load_state_dict(checkpoint)
    elif os.path.exists(weights_path_safetensors):
        print(f"Loading weights from {weights_path_safetensors}")
        try:
            from safetensors.torch import load_file
            checkpoint = load_file(weights_path_safetensors)
            model.load_state_dict(checkpoint)
        except ImportError:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )
    else:
        raise FileNotFoundError(
            f"Could not find model weights in {model_path}. "
            f"Expected either pytorch_model.bin or model.safetensors"
        )

    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
    config = SmallVerifiableConfig.load(config_path)

    print("Circuit Extraction for Small Verifiable Transformer")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Metric: {args.metric}")
    print(f"Min agreement: {args.min_agreement}")
    print(f"Ablation: {args.ablation}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, config, device)

    print(f"Model loaded on {device}")

    # Extract circuit
    if args.threshold_sweep:
        print("\nPerforming threshold sweep...")
        results = threshold_sweep(
            model,
            args.task,
            config,
            args.thresholds,
            args.min_agreement,
            device,
        )

        # Find best circuit (smallest with perfect agreement)
        best_result = None
        for result in results:
            if result["metrics"]["agreement"] >= args.min_agreement:
                if best_result is None or result["n_edges"] < best_result["n_edges"]:
                    best_result = result

        if best_result is None:
            print("\nWARNING: No circuit found with sufficient agreement!")
            # Fall back to circuit with highest agreement
            best_result = max(results, key=lambda r: r["metrics"]["agreement"])

        # Save sweep results
        sweep_path = os.path.join(args.output_dir, "threshold_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nThreshold sweep saved to: {sweep_path}")

        # Save best circuit
        circuit_path = os.path.join(args.output_dir, "circuit.json")
        with open(circuit_path, "w") as f:
            json.dump(best_result["circuit"], f, indent=2)
        print(f"Best circuit saved to: {circuit_path}")

        # Save edge log for best circuit
        edge_log_path = os.path.join(args.output_dir, "edge_log.json")
        with open(edge_log_path, "w") as f:
            json.dump(best_result["edge_log"], f, indent=2)
        print(f"Edge log saved to: {edge_log_path}")

        # Save DOT file for visualization
        dot_path = os.path.join(args.output_dir, "circuit.dot")
        with open(dot_path, "w") as f:
            f.write("digraph circuit {\n")
            f.write("  rankdir=LR;\n")
            for edge_dict in best_result["circuit"]["edges"]:
                source = edge_dict["source"]
                target = edge_dict["target"]
                f.write(f'  "{source}" -> "{target}";\n')
            f.write("}\n")
        print(f"DOT file saved to: {dot_path}")

        # Save summary
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Circuit Extraction Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Task: {args.task}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"\n")
            f.write(f"Best Circuit:\n")
            f.write(f"  Threshold: {best_result['threshold']}\n")
            f.write(f"  Nodes: {len(best_result['circuit']['nodes'])}\n")
            f.write(f"  Edges: {best_result['n_edges']}\n")
            f.write(f"  Candidate Accuracy: {best_result['metrics']['candidate_accuracy']:.4f}\n")
            f.write(f"  Agreement with Full Model: {best_result['metrics']['agreement']:.4f}\n")
            f.write(f"  Mean Candidate KL: {best_result['metrics']['mean_candidate_kl']:.6f}\n")
        print(f"Summary saved to: {summary_path}")

    else:
        # Single threshold extraction
        threshold = args.thresholds[0]
        graph, metrics, edges_to_keep, edge_log = extract_circuit_threshold(
            model, args.task, config, threshold, args.min_agreement, device
        )

        # Create circuit dict
        circuit_dict = {
            "task": args.task,
            "n_layers": graph.n_layers,
            "nodes": graph.nodes,
            "edges": [{"source": s, "target": t} for s, t in sorted(edges_to_keep)],
            "num_edges": len(edges_to_keep),
            "ablation": args.ablation,
            "metric": args.metric,
            "min_agreement": args.min_agreement,
            "threshold": threshold,
        }

        # Save circuit
        circuit_path = os.path.join(args.output_dir, "circuit.json")
        with open(circuit_path, "w") as f:
            json.dump(circuit_dict, f, indent=2)
        print(f"\nCircuit saved to: {circuit_path}")

        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")

        # Save edge log
        edge_log_path = os.path.join(args.output_dir, "edge_log.json")
        with open(edge_log_path, "w") as f:
            json.dump(edge_log, f, indent=2)
        print(f"Edge log saved to: {edge_log_path}")

        # Save DOT file for visualization
        dot_path = os.path.join(args.output_dir, "circuit.dot")
        with open(dot_path, "w") as f:
            f.write("digraph circuit {\n")
            f.write("  rankdir=LR;\n")
            for source, target in sorted(edges_to_keep):
                f.write(f'  "{source}" -> "{target}";\n')
            f.write("}\n")
        print(f"DOT file saved to: {dot_path}")

    print("\n" + "=" * 60)
    print("Circuit extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
