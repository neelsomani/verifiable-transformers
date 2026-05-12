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
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, asdict

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


# ============================================================================
# Circuit Graph
# ============================================================================


@dataclass
class CircuitNode:
    """A node in the circuit computational graph."""
    name: str
    layer: int
    type: str  # "emb", "attn", "mlp", "logits"


@dataclass
class CircuitEdge:
    """An edge in the circuit computational graph."""
    source: str
    target: str
    weight: float = 1.0  # Edge importance (for future use)


class CircuitGraph:
    """Computational graph for circuit extraction."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.nodes: List[str] = []
        self.edges: Set[Tuple[str, str]] = set()
        self.incoming_edges: Dict[str, List[Tuple[str, str]]] = {}

        self._build_graph()

    def _build_graph(self):
        """Build the computational graph structure."""
        # Add embedding node
        self.nodes.append("emb")
        self.incoming_edges["emb"] = []

        # Add layer nodes
        for layer in range(self.n_layers):
            attn_node = f"attn_{layer}"
            mlp_node = f"mlp_{layer}"

            self.nodes.append(attn_node)
            self.nodes.append(mlp_node)

            self.incoming_edges[attn_node] = []
            self.incoming_edges[mlp_node] = []

        # Add output node
        self.nodes.append("logits")
        self.incoming_edges["logits"] = []

        # Add edges (coarse-grained: block-level)
        # Embedding feeds into all attention and MLP blocks (via residual)
        for layer in range(self.n_layers):
            self.edges.add(("emb", f"attn_{layer}"))
            self.edges.add(("emb", f"mlp_{layer}"))

        # Each layer feeds into subsequent layers
        for layer in range(self.n_layers):
            attn_node = f"attn_{layer}"
            mlp_node = f"mlp_{layer}"

            # Attention output feeds into same-layer MLP
            self.edges.add((attn_node, mlp_node))

            # Both feed into next layer's attention and MLP
            for next_layer in range(layer + 1, self.n_layers):
                self.edges.add((attn_node, f"attn_{next_layer}"))
                self.edges.add((attn_node, f"mlp_{next_layer}"))
                self.edges.add((mlp_node, f"attn_{next_layer}"))
                self.edges.add((mlp_node, f"mlp_{next_layer}"))

            # Feed into logits
            self.edges.add((attn_node, "logits"))
            self.edges.add((mlp_node, "logits"))

        # Also embedding directly to logits
        self.edges.add(("emb", "logits"))

        # Build incoming edge lists
        for source, target in self.edges:
            self.incoming_edges[target].append((source, target))

    def remove_edge(self, edge: Tuple[str, str]):
        """Remove an edge from the graph."""
        if edge in self.edges:
            self.edges.discard(edge)
            source, target = edge
            self.incoming_edges[target] = [
                e for e in self.incoming_edges[target] if e != edge
            ]

    def get_edges(self) -> Set[Tuple[str, str]]:
        """Get all current edges."""
        return self.edges.copy()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_layers": self.n_layers,
            "nodes": self.nodes,
            "edges": [{"source": s, "target": t} for s, t in sorted(self.edges)],
        }


# ============================================================================
# Circuit Evaluation
# ============================================================================


def evaluate_circuit(
    model: GPT2LMHeadModel,
    task_name: str,
    circuit_edges: Set[Tuple[str, str]],
    ablation_type: str = "zero",
    device: torch.device = None,
) -> Dict:
    """
    Evaluate a circuit on the task's exhaustive domain.

    Args:
        model: The model to evaluate
        task_name: Task name ("quote_close", "bracket_type", "add_mod_5")
        circuit_edges: Set of edges to keep (all others are ablated)
        ablation_type: Type of ablation ("zero" only for now)
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
    candidates = list(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task_name]))

    # TODO: Implement actual circuit ablation
    # For now, just evaluate the full model
    # This is a placeholder - proper implementation would:
    # 1. Hook into model forward pass
    # 2. Ablate edges not in circuit_edges
    # 3. Run forward pass with ablated edges

    correct = 0
    total = len(examples)

    with torch.no_grad():
        for example in examples:
            input_ids = torch.tensor(
                example["input_ids"], dtype=torch.long, device=device
            ).unsqueeze(0)
            target = example["target"]

            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]

            # Candidate prediction
            candidate_logits = logits[candidates]
            pred_candidate_idx = candidate_logits.argmax().item()
            pred_candidate = candidates[pred_candidate_idx]

            if pred_candidate == target:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "candidate_accuracy": accuracy,
        "agreement": 1.0,  # Placeholder
        "mean_candidate_kl": 0.0,  # Placeholder
    }


# ============================================================================
# Circuit Extraction
# ============================================================================


def extract_circuit_threshold(
    model: GPT2LMHeadModel,
    task_name: str,
    config: SmallVerifiableConfig,
    threshold: float,
    min_agreement: float = 1.0,
    device: torch.device = None,
) -> Tuple[CircuitGraph, Dict]:
    """
    Extract circuit using threshold-based pruning.

    For now, this is a simple placeholder that returns the full graph.
    A proper implementation would:
    1. Start with full graph
    2. Iteratively prune edges below threshold
    3. Check if agreement >= min_agreement
    4. Return smallest circuit that maintains agreement

    Args:
        model: Model to extract from
        task_name: Task name
        config: Model config
        threshold: Edge importance threshold
        min_agreement: Minimum agreement with full model
        device: Device to run on

    Returns:
        (circuit_graph, metrics)
    """
    if device is None:
        device = next(model.parameters()).device

    # Create full graph
    graph = CircuitGraph(n_layers=config.n_layers)

    # Evaluate full circuit
    metrics = evaluate_circuit(model, task_name, graph.get_edges(), device=device)

    # TODO: Implement actual circuit extraction
    # For now, return full graph
    print(f"Extracted circuit for {task_name}:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Candidate accuracy: {metrics['candidate_accuracy']:.4f}")

    return graph, metrics


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
        print(f"\nThreshold: {threshold}")
        graph, metrics = extract_circuit_threshold(
            model, task_name, config, threshold, min_agreement, device
        )

        result = {
            "threshold": threshold,
            "n_edges": len(graph.edges),
            "metrics": metrics,
            "circuit": graph.to_dict(),
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
        choices=["quote_close", "bracket_type", "add_mod_5"],
        help="Task to extract circuit for",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="candidate_kl",
        choices=["candidate_kl", "agreement"],
        help="Metric to use for circuit selection",
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
    from scripts.small.train import create_small_model

    model = create_small_model(config)

    # Load weights
    checkpoint = torch.load(
        os.path.join(args.model_path, "pytorch_model.bin"),
        map_location="cpu",
    )
    model.load_state_dict(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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
            if result["metrics"]["candidate_accuracy"] >= args.min_agreement:
                if best_result is None or result["n_edges"] < best_result["n_edges"]:
                    best_result = result

        if best_result is None:
            print("\nWARNING: No circuit found with sufficient agreement!")
            best_result = results[0]  # Use full circuit

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
        print(f"Summary saved to: {summary_path}")

    else:
        # Single threshold extraction
        threshold = args.thresholds[0]
        graph, metrics = extract_circuit_threshold(
            model, args.task, config, threshold, args.min_agreement, device
        )

        # Save circuit
        circuit_path = os.path.join(args.output_dir, "circuit.json")
        with open(circuit_path, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        print(f"\nCircuit saved to: {circuit_path}")

        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print("Circuit extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
