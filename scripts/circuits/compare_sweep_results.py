#!/usr/bin/env python3
"""
Compare circuit extraction results across threshold sweep.

Helps select the best threshold based on:
- Accuracy preservation
- Edge count (smaller is better)
- Candidate KL (lower is better)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_sweep_results(sweep_dir: str, task: str) -> List[Dict[str, Any]]:
    """Load all circuit results for a task from sweep directory."""
    results = []
    sweep_path = Path(sweep_dir)

    # Find all threshold directories for this task
    for thresh_dir in sorted(sweep_path.glob(f"{task}_t*")):
        circuit_json = thresh_dir / "circuit.json"
        if not circuit_json.exists():
            continue

        with open(circuit_json) as f:
            data = json.load(f)

        # Extract threshold from directory name
        thresh_str = thresh_dir.name.split("_t")[-1]
        threshold = float(thresh_str)

        results.append({
            "threshold": threshold,
            "num_edges": data["num_edges"],
            "metric": data.get("metric", "kl"),
            "full_candidate_accuracy": data["scores"]["full"].get("candidate_accuracy", data["scores"]["full"]["binary_accuracy"]),
            "circuit_candidate_accuracy": data["scores"]["circuit"].get("candidate_accuracy", data["scores"]["circuit"]["binary_accuracy"]),
            "circuit_candidate_kl": data["scores"]["circuit"].get("candidate_kl_from_full", 0.0),
            "projected_agreement": data["scores"]["circuit"].get("projected_agreement_with_full", 0.0),
            "full_margin": data["scores"]["full"].get("mean_margin", data["scores"]["full"]["mean_logit_diff"]),
            "circuit_margin": data["scores"]["circuit"].get("mean_margin", data["scores"]["circuit"]["mean_logit_diff"]),
            "path": str(thresh_dir),
        })

    return sorted(results, key=lambda x: x["threshold"])


def print_comparison_table(results: List[Dict[str, Any]], task: str):
    """Print formatted comparison table."""
    print(f"\n{'=' * 100}")
    print(f"THRESHOLD SWEEP RESULTS: {task}")
    print(f"{'=' * 100}\n")

    # Header
    print(f"{'Threshold':>10} {'Edges':>6} {'Full Acc':>9} {'Circuit Acc':>12} "
          f"{'Agreement':>10} {'Cand KL':>9} {'Full Margin':>12} {'Circ Margin':>12}")
    print("-" * 100)

    # Rows
    for r in results:
        print(f"{r['threshold']:>10.4f} {r['num_edges']:>6d} {r['full_candidate_accuracy']:>9.3f} "
              f"{r['circuit_candidate_accuracy']:>12.3f} {r['projected_agreement']:>10.3f} "
              f"{r['circuit_candidate_kl']:>9.5f} {r['full_margin']:>12.3f} {r['circuit_margin']:>12.3f}")

    print()


def recommend_threshold(results: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    """Recommend best threshold based on task-specific criteria."""
    if task in ["quote_close", "bracket_type"]:
        # For quote/bracket: require perfect projected agreement, minimize edges
        perfect = [r for r in results if r["projected_agreement"] >= 0.999]
        if perfect:
            best = min(perfect, key=lambda x: x["num_edges"])
            return best
        else:
            print("WARNING: No threshold achieved perfect projected agreement!")
            return min(results, key=lambda x: (1.0 - x["projected_agreement"], x["num_edges"]))

    elif task == "induction_ABCAB":
        # For induction: require projected agreement >= 0.95 and accuracy >= full model, minimize edges
        full_acc = results[0]["full_candidate_accuracy"] if results else 0.85
        target_acc = max(full_acc, 0.85)
        viable = [r for r in results if r["projected_agreement"] >= 0.95 and r["circuit_candidate_accuracy"] >= target_acc]

        if viable:
            best = min(viable, key=lambda x: x["num_edges"])
            return best
        else:
            print(f"WARNING: No threshold achieved target criteria (agreement >= 0.95, accuracy >= {target_acc:.3f})!")
            return min(results, key=lambda x: (1.0 - x["projected_agreement"], target_acc - x["circuit_candidate_accuracy"], x["num_edges"]))

    else:
        # Default: minimize edges while preserving projected agreement and accuracy
        full_acc = results[0]["full_candidate_accuracy"] if results else 1.0
        viable = [r for r in results if r["projected_agreement"] >= 0.99 and r["circuit_candidate_accuracy"] >= full_acc - 0.01]
        if viable:
            return min(viable, key=lambda x: x["num_edges"])
        else:
            return results[0] if results else None


def main():
    parser = argparse.ArgumentParser(description="Compare threshold sweep results")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Directory containing sweep results")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (quote_close, bracket_type, induction_ABCAB)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional: save comparison to JSON")

    args = parser.parse_args()

    # Load results
    results = load_sweep_results(args.sweep_dir, args.task)

    if not results:
        print(f"ERROR: No results found for task '{args.task}' in {args.sweep_dir}")
        return

    # Print table
    print_comparison_table(results, args.task)

    # Recommend best threshold
    best = recommend_threshold(results, args.task)

    if best:
        print(f"{'=' * 100}")
        print("RECOMMENDED THRESHOLD")
        print(f"{'=' * 100}\n")
        print(f"Threshold:           {best['threshold']:.4f}")
        print(f"Edges:               {best['num_edges']} / 325 ({100 * best['num_edges'] / 325:.1f}%)")
        print(f"Projected Agreement: {best['projected_agreement']:.4f}")
        print(f"Circuit Cand Acc:    {best['circuit_candidate_accuracy']:.4f}")
        print(f"Full Cand Acc:       {best['full_candidate_accuracy']:.4f}")
        print(f"Candidate KL:        {best['circuit_candidate_kl']:.6f}")
        print(f"Circuit Path:        {best['path']}")
        print()

        # Selection criteria
        if args.task in ["quote_close", "bracket_type"]:
            print("Selection: Perfect projected agreement (1.000), smallest edge count")
        elif args.task == "induction_ABCAB":
            print(f"Selection: Agreement >= 0.95, Accuracy >= {max(best['full_candidate_accuracy'], 0.85):.3f}, smallest edge count")
        print()

    # Save to JSON if requested
    if args.output_json:
        output_data = {
            "task": args.task,
            "sweep_dir": args.sweep_dir,
            "results": results,
            "recommended": best,
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved comparison to: {args.output_json}\n")


if __name__ == "__main__":
    main()
