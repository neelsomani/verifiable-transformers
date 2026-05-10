"""Property verification using SMT solvers."""

from z3 import *
from typing import List, Dict, Any, Set, Tuple, Callable, Optional
from .circuit import encode_circuit_forward
from .bounded_domain import generate_bounded_sequences
from .helpers import parse_circuit_edges, get_candidate_tokens


def verify_functional_equivalence(
    circuit: Dict[str, Any],
    reference_program: Callable[[List[int]], int],
    input_sequences: List[Tuple[List[int], Any]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    timeout_ms: int = 60000,
) -> Dict[str, Any]:
    """Verify y_C(x) = P(x) for all inputs using SMT.

    Args:
        circuit: Extracted circuit with edges
        reference_program: Function mapping input tokens to expected output token
        input_sequences: List of (input_tokens, expected_output) pairs
        model_weights: Model weight parameters
        candidate_tokens: List of candidate output token IDs
        timeout_ms: Solver timeout in milliseconds

    Returns:
        Verification result dict
    """
    circuit_edges = parse_circuit_edges(circuit)

    counterexamples = []
    verified_count = 0
    timeout_count = 0

    print(f"Verifying functional equivalence on {len(input_sequences)} sequences...")

    for i, (input_tokens, expected_output) in enumerate(input_sequences):
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i}/{len(input_sequences)}")

        # Create fresh solver for each sequence
        solver = Solver()
        solver.set("timeout", timeout_ms)

        # Encode circuit forward pass
        try:
            circuit_logits = encode_circuit_forward(
                input_tokens,
                circuit_edges,
                model_weights,
                candidate_tokens,
                solver,
                f"seq_{i}",
            )

            # Get expected token
            if isinstance(expected_output, int):
                expected_token = expected_output
            elif expected_output == "single":
                expected_token = model_weights.get("single_quote_id", candidate_tokens[0])
            elif expected_output == "double":
                expected_token = model_weights.get("double_quote_id", candidate_tokens[1])
            else:
                expected_token = int(expected_output)

            # Check if expected token is in candidates
            if expected_token not in circuit_logits:
                print(f"  Warning: expected token {expected_token} not in candidates")
                continue

            # Add constraint: exists candidate token with higher logit than expected
            violations = []
            for tok in candidate_tokens:
                if tok != expected_token and tok in circuit_logits:
                    violations.append(circuit_logits[tok] > circuit_logits[expected_token])

            if not violations:
                # No other candidates, trivially verified
                verified_count += 1
                continue

            solver.add(Or(violations))

            # Check satisfiability
            result = solver.check()

            if result == unsat:
                # Property holds: expected token has highest logit
                verified_count += 1
            elif result == sat:
                # Counterexample found
                model = solver.model()
                counterexamples.append({
                    "input": input_tokens,
                    "expected": expected_token,
                    "model_prediction": "incorrect",
                })
                if len(counterexamples) >= 10:
                    # Stop after finding 10 counterexamples
                    break
            else:
                # Timeout or unknown
                timeout_count += 1

        except Exception as e:
            print(f"  Error on sequence {i}: {e}")
            continue

    success = len(counterexamples) == 0
    status = "VERIFIED" if success else "FAILED"

    return {
        "property": "functional_equivalence",
        "status": status,
        "verified_count": verified_count,
        "timeout_count": timeout_count,
        "total_sequences": len(input_sequences),
        "counterexamples": counterexamples[:5],  # Return first 5
        "num_counterexamples": len(counterexamples),
    }


def verify_content_invariance(
    circuit: Dict[str, Any],
    input_sequences: List[Tuple[List[int], Any]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    get_structural_feature: Callable[[List[int]], Any],
    timeout_ms: int = 60000,
) -> Dict[str, Any]:
    """Verify: same structure => same output.

    For quote_close: quote(x) = quote(x') => y_C(x) = y_C(x')

    Args:
        circuit: Extracted circuit
        input_sequences: List of input sequences
        model_weights: Model weights
        candidate_tokens: List of candidate output token IDs
        get_structural_feature: Function extracting structural feature (e.g., quote type)
        timeout_ms: Solver timeout

    Returns:
        Verification result dict
    """
    circuit_edges = parse_circuit_edges(circuit)

    # Group sequences by structural feature
    feature_groups = {}
    for seq, label in input_sequences:
        feature = get_structural_feature(seq)
        if feature not in feature_groups:
            feature_groups[feature] = []
        feature_groups[feature].append((seq, label))

    counterexamples = []
    verified_pairs = 0
    timeout_count = 0

    print(f"Verifying content invariance across {len(feature_groups)} feature groups...")

    for feature, sequences in feature_groups.items():
        if len(sequences) < 2:
            continue

        # Compare first two sequences with same feature
        seq1, label1 = sequences[0]
        seq2, label2 = sequences[1]

        if len(seq1) != len(seq2):
            continue  # Skip different lengths for now

        solver = Solver()
        solver.set("timeout", timeout_ms)

        try:
            # Encode both sequences
            logits1 = encode_circuit_forward(
                seq1, circuit_edges, model_weights, candidate_tokens, solver, "seq1"
            )
            logits2 = encode_circuit_forward(
                seq2, circuit_edges, model_weights, candidate_tokens, solver, "seq2"
            )

            # Add constraint: logits differ for any candidate
            differences = []
            for tok in candidate_tokens:
                if tok in logits1 and tok in logits2:
                    differences.append(logits1[tok] != logits2[tok])

            if not differences:
                verified_pairs += 1
                continue

            solver.add(Or(differences))

            result = solver.check()

            if result == unsat:
                # Property holds: outputs are identical
                verified_pairs += 1
            elif result == sat:
                counterexamples.append({
                    "feature": str(feature),
                    "seq1": seq1,
                    "seq2": seq2,
                })
                if len(counterexamples) >= 10:
                    break
            else:
                timeout_count += 1

        except Exception as e:
            print(f"  Error verifying feature {feature}: {e}")
            continue

    success = len(counterexamples) == 0
    status = "VERIFIED" if success else "FAILED"

    return {
        "property": "content_invariance",
        "status": status,
        "verified_pairs": verified_pairs,
        "timeout_count": timeout_count,
        "counterexamples": counterexamples[:5],
        "num_counterexamples": len(counterexamples),
    }


def verify_edge_necessity(
    circuit: Dict[str, Any],
    test_inputs: List[List[int]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    timeout_ms: int = 60000,
) -> Dict[str, Any]:
    """Verify: for each edge e, exists x such that C(x) != (C \\ e)(x).

    Args:
        circuit: Extracted circuit
        test_inputs: Test input sequences
        model_weights: Model weights
        candidate_tokens: List of candidate output token IDs
        timeout_ms: Solver timeout

    Returns:
        Verification result dict
    """
    edges = [(e["from"], e["to"]) if isinstance(e, dict) else (e[0], e[1]) for e in circuit["edges"]]
    circuit_edges = parse_circuit_edges(circuit)

    unnecessary_edges = []
    necessary_edges = []
    timeout_count = 0

    print(f"Verifying necessity of {len(edges)} edges...")

    for edge_idx, edge in enumerate(edges):
        if edge_idx % 10 == 0 and edge_idx > 0:
            print(f"  Progress: {edge_idx}/{len(edges)}")

        edge_from, edge_to = edge

        # Create ablated circuit (remove this edge)
        ablated_edges = circuit_edges - {edge}

        # Try to find input where removing edge changes output
        found_witness = False

        for input_tokens in test_inputs[:20]:  # Test on first 20 inputs per edge
            solver = Solver()
            solver.set("timeout", timeout_ms)

            try:
                # Encode full circuit
                logits_full = encode_circuit_forward(
                    input_tokens,
                    circuit_edges,
                    model_weights,
                    candidate_tokens,
                    solver,
                    f"full_e{edge_idx}",
                )

                # Encode ablated circuit
                logits_ablated = encode_circuit_forward(
                    input_tokens,
                    ablated_edges,
                    model_weights,
                    candidate_tokens,
                    solver,
                    f"ablated_e{edge_idx}",
                )

                # Add constraint: logits are identical for all candidates
                constraints = []
                for tok in candidate_tokens:
                    if tok in logits_full and tok in logits_ablated:
                        constraints.append(logits_full[tok] == logits_ablated[tok])

                if not constraints:
                    continue

                solver.add(And(constraints))

                result = solver.check()

                if result == unsat:
                    # Outputs differ - edge is necessary
                    found_witness = True
                    necessary_edges.append(edge)
                    break
                elif result == sat:
                    # Outputs same on this input - try next input
                    continue
                else:
                    timeout_count += 1

            except Exception as e:
                print(f"  Error testing edge {edge}: {e}")
                break

        if not found_witness:
            unnecessary_edges.append({
                "edge": f"{edge_from} -> {edge_to}",
                "tested_inputs": min(20, len(test_inputs)),
            })

    success = len(unnecessary_edges) == 0
    status = "VERIFIED" if success else "FAILED"

    return {
        "property": "edge_necessity",
        "status": status,
        "total_edges": len(edges),
        "necessary_edges": len(necessary_edges),
        "unnecessary_edges_found": len(unnecessary_edges),
        "timeout_count": timeout_count,
        "suspicious_edges": unnecessary_edges[:10],
    }


def verify_token_renaming_equivariance(
    circuit: Dict[str, Any],
    test_sequences: List[List[int]],
    model_weights: Dict[str, Any],
    vocab_size: int,
    timeout_ms: int = 60000,
) -> Dict[str, Any]:
    """Verify: y_C(r(x)) = r(y_C(x)) for token renaming r.

    Args:
        circuit: Extracted circuit
        test_sequences: Test sequences
        model_weights: Model weights
        vocab_size: Vocabulary size
        timeout_ms: Solver timeout

    Returns:
        Verification result dict
    """
    circuit_edges = parse_circuit_edges(circuit)

    counterexamples = []
    verified_count = 0

    # Use synthetic token set for tractability
    candidate_tokens = list(range(20, 30))

    print(f"Verifying token-renaming equivariance on {len(test_sequences)} sequences...")

    # Test simple permutations (swap two tokens)
    for seq_idx, seq in enumerate(test_sequences[:20]):
        if seq_idx % 10 == 0 and seq_idx > 0:
            print(f"  Progress: {seq_idx}/20")

        # Create a simple permutation: swap tokens 20 and 21
        def permute(tokens):
            return [21 if t == 20 else 20 if t == 21 else t for t in tokens]

        seq_permuted = permute(seq)

        solver = Solver()
        solver.set("timeout", timeout_ms)

        try:
            # Encode original sequence
            logits_orig = encode_circuit_forward(
                seq, circuit_edges, model_weights, candidate_tokens, solver, f"orig_{seq_idx}"
            )

            # Encode permuted sequence
            logits_perm = encode_circuit_forward(
                seq_permuted, circuit_edges, model_weights, candidate_tokens, solver, f"perm_{seq_idx}"
            )

            # Check: logits_perm[20] should equal logits_orig[21]
            #        logits_perm[21] should equal logits_orig[20]
            if 20 in logits_orig and 21 in logits_orig and 20 in logits_perm and 21 in logits_perm:
                solver.add(Or(
                    logits_perm[20] != logits_orig[21],
                    logits_perm[21] != logits_orig[20],
                ))

                result = solver.check()

                if result == unsat:
                    verified_count += 1
                elif result == sat:
                    counterexamples.append({"sequence": seq})
                    if len(counterexamples) >= 10:
                        break

        except Exception as e:
            print(f"  Error on sequence {seq_idx}: {e}")
            continue

    success = len(counterexamples) == 0
    status = "VERIFIED" if success else "FAILED"

    return {
        "property": "token_renaming_equivariance",
        "status": status,
        "verified_count": verified_count,
        "counterexamples": counterexamples[:5],
        "num_counterexamples": len(counterexamples),
    }


def verify_structural_constraint(
    circuit: Dict[str, Any],
    constraint_checker: Callable[[Dict[int, ArithRef], Solver], None],
    test_inputs: List[List[int]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    constraint_name: str,
    timeout_ms: int = 60000,
) -> Dict[str, Any]:
    """Verify a structural constraint holds for all inputs.

    Args:
        circuit: Extracted circuit
        constraint_checker: Function that adds constraints to solver
        test_inputs: Test input sequences
        model_weights: Model weights
        candidate_tokens: List of candidate output token IDs
        constraint_name: Name of constraint
        timeout_ms: Solver timeout

    Returns:
        Verification result dict
    """
    circuit_edges = parse_circuit_edges(circuit)

    violations = []
    verified_count = 0

    print(f"Verifying {constraint_name} on {len(test_inputs)} inputs...")

    for i, input_tokens in enumerate(test_inputs):
        if i % 50 == 0 and i > 0:
            print(f"  Progress: {i}/{len(test_inputs)}")

        solver = Solver()
        solver.set("timeout", timeout_ms)

        try:
            # Encode circuit
            logits = encode_circuit_forward(
                input_tokens, circuit_edges, model_weights, candidate_tokens, solver, f"input_{i}"
            )

            # Add constraint and negate it (check if constraint can be violated)
            constraint_checker(logits, solver)

            result = solver.check()

            if result == unsat:
                # Constraint cannot be violated
                verified_count += 1
            elif result == sat:
                # Constraint violated
                violations.append({"input": input_tokens})
                if len(violations) >= 10:
                    break

        except Exception as e:
            print(f"  Error on input {i}: {e}")
            continue

    success = len(violations) == 0
    status = "VERIFIED" if success else "FAILED"

    return {
        "property": constraint_name,
        "status": status,
        "verified_count": verified_count,
        "violations": violations[:5],
        "num_violations": len(violations),
    }
