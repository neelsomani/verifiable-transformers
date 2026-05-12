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
        self.all_edges = self.edges.copy()  # Full edge set for reference

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


def controlled_forward(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    edges_to_keep: Set[Tuple[str, str]],
    graph: CircuitGraph,
    return_node_outputs: bool = False,
):
    """Controlled forward pass with edge ablation (zero ablation only).

    Args:
        model: GPT2 model
        input_ids: [B, T]
        edges_to_keep: Set of (parent, child) edges to use clean activations
        graph: Circuit graph
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
            # else: zero ablation (deleted edges contribute 0)

        # Compute attention output
        x_norm = block.ln_1(resid_for_attn)
        attn_out = block.attn(
            x_norm,
            attention_mask=None,
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
            # else: zero ablation (deleted edges contribute 0)

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
        # else: zero ablation (deleted edges contribute 0)

    # Final layer norm and logits
    hidden = model.transformer.ln_f(final_resid)
    logits = model.lm_head(hidden)

    if return_node_outputs:
        return logits, node_outputs
    return logits


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
    atol: float = 1e-5,
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
    verify_controlled_forward(model, task_name, graph, device, atol=1e-5)
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
    graph = CircuitGraph(n_layers=config.n_layers)

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

    model = create_small_model(config)

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
