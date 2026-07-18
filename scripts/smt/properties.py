"""Property verification using SMT solvers."""

from z3 import *
import time
from typing import List, Dict, Any, Set, Tuple, Callable, Optional
from .circuit import encode_circuit_forward
from .domain import generate_bounded_sequences
from .utils import parse_circuit_edges, get_candidate_tokens
from .encoders import (
    encode_signed_l1_band_norm,
    encode_signed_l1_band_norm_with_trace,
    signed_l1_band_norm_guard_conditions,
    z3_real,
)
from .trace import trace_circuit_forward
from .attribution import (
    add_profiled_assertion,
    assertion_total,
    increment_norm_instances,
    new_assertion_profile,
    record_solver_delta,
)


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
    error_count = 0
    assertion_profile = new_assertion_profile()
    encoding_seconds = 0.0
    solve_seconds = 0.0

    print(f"Verifying functional equivalence on {len(input_sequences)} sequences...")

    for i, (input_tokens, _) in enumerate(input_sequences):
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i}/{len(input_sequences)}")

        # Create fresh solver for each sequence
        solver = Solver()
        solver.set("timeout", timeout_ms)

        # Encode circuit forward pass
        try:
            print(f"  seq {i}: encoding...", flush=True)
            t0 = time.time()

            circuit_logits = encode_circuit_forward(
                input_tokens,
                circuit_edges,
                model_weights,
                candidate_tokens,
                solver,
                f"seq_{i}",
                trace=trace_circuit_forward(input_tokens, circuit_edges, model_weights, f"seq_{i}"),
                assertion_profile=assertion_profile,
            )

            encoding_seconds += time.time() - t0
            print(
                f"  seq {i}: encoded in {time.time() - t0:.2f}s, "
                f"assertions={len(solver.assertions())}",
                flush=True,
            )

            # Get expected token from reference program
            expected_token = reference_program(input_tokens)

            # Check if expected token is in candidates
            if expected_token not in circuit_logits:
                print(f"  Warning: expected token {expected_token} not in candidates")
                continue

            # Add constraint: exists candidate token tied with or above expected.
            # UNSAT then proves the expected token strictly beats every other
            # candidate, so projected decisions are not silently verified on ties.
            violations = []
            for tok in candidate_tokens:
                if tok != expected_token and tok in circuit_logits:
                    violations.append(circuit_logits[tok] >= circuit_logits[expected_token])

            print(f"  seq {i}: validating base constraints...", flush=True)
            solve_started = time.time()
            base_result = solver.check()
            solve_seconds += time.time() - solve_started
            print(f"  seq {i}: base result={base_result}", flush=True)

            if base_result == unsat:
                print(f"  ERROR: trace certificate invalid on sequence {i}", flush=True)
                error_count += 1
                continue
            if base_result != sat:
                print(f"  UNKNOWN: base trace constraints timed out on sequence {i}", flush=True)
                timeout_count += 1
                continue

            if not violations:
                # No other candidates, trivially verified after certificate validation.
                verified_count += 1
                continue

            solver.push()
            add_profiled_assertion(
                assertion_profile, "decision", solver, Or(violations)
            )

            # Check satisfiability
            print(f"  seq {i}: checking violation...", flush=True)
            t1 = time.time()
            result = solver.check()
            solve_seconds += time.time() - t1
            print(f"  seq {i}: violation result={result} in {time.time() - t1:.2f}s", flush=True)
            solver.pop()

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
            error_count += 1
            continue

    if len(counterexamples) > 0:
        status = "FAILED"
    elif timeout_count > 0 or error_count > 0:
        status = "UNKNOWN"
    else:
        status = "VERIFIED"

    return {
        "property": "projected_functional_equivalence",
        "status": status,
        "verified_count": verified_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "total_sequences": len(input_sequences),
        "counterexamples": counterexamples[:5],  # Return first 5
        "num_counterexamples": len(counterexamples),
        "assertion_count": assertion_total(assertion_profile),
        "assertion_attribution": assertion_profile,
        "encoding_seconds": encoding_seconds,
        "solve_seconds": solve_seconds,
    }


def _verify_content_invariance_via_anchors(
    circuit: Dict[str, Any],
    input_sequences: List[Tuple[List[int], Any]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    get_structural_feature: Callable[[List[int]], Any],
    timeout_ms: int,
) -> Dict[str, Any]:
    """Prove each group against one anchor; transitivity covers every pair."""
    circuit_edges = parse_circuit_edges(circuit)
    groups: Dict[Any, List[List[int]]] = {}
    for sequence, _ in input_sequences:
        groups.setdefault(get_structural_feature(sequence), []).append(sequence)

    verified_comparisons = 0
    total_pairs = sum(len(group) * (len(group) - 1) // 2 for group in groups.values())
    assertion_profile = new_assertion_profile()
    solve_seconds = 0.0
    timeout_count = 0
    error_count = 0
    counterexamples = []
    for feature, sequences in groups.items():
        if len(sequences) < 2:
            continue
        anchor = sequences[0]
        for index, sequence in enumerate(sequences[1:], start=1):
            solver = Solver()
            solver.set("timeout", timeout_ms)
            try:
                anchor_prefix = f"feature_{feature}_anchor_{index}"
                sequence_prefix = f"feature_{feature}_seq_{index}"
                anchor_logits = encode_circuit_forward(
                    anchor,
                    circuit_edges,
                    model_weights,
                    candidate_tokens,
                    solver,
                    anchor_prefix,
                    trace=trace_circuit_forward(
                        anchor, circuit_edges, model_weights, anchor_prefix
                    ),
                    assertion_profile=assertion_profile,
                )
                sequence_logits = encode_circuit_forward(
                    sequence,
                    circuit_edges,
                    model_weights,
                    candidate_tokens,
                    solver,
                    sequence_prefix,
                    trace=trace_circuit_forward(
                        sequence, circuit_edges, model_weights, sequence_prefix
                    ),
                    assertion_profile=assertion_profile,
                )
                ordering_differs = []
                for left in candidate_tokens:
                    for right in candidate_tokens:
                        if left == right:
                            continue
                        ordering_differs.extend(
                            [
                                And(
                                    anchor_logits[left] > anchor_logits[right],
                                    sequence_logits[left] <= sequence_logits[right],
                                ),
                                And(
                                    anchor_logits[left] <= anchor_logits[right],
                                    sequence_logits[left] > sequence_logits[right],
                                ),
                            ]
                        )
                started = time.time()
                base = solver.check()
                solve_seconds += time.time() - started
                if base != sat:
                    if base == unknown:
                        timeout_count += 1
                    else:
                        error_count += 1
                    continue
                solver.push()
                add_profiled_assertion(
                    assertion_profile, "decision", solver, Or(ordering_differs)
                )
                started = time.time()
                result = solver.check()
                solve_seconds += time.time() - started
                solver.pop()
                if result == unsat:
                    verified_comparisons += 1
                elif result == sat:
                    counterexamples.append(
                        {"feature": str(feature), "anchor": anchor, "sequence": sequence}
                    )
                else:
                    timeout_count += 1
            except Exception:
                error_count += 1

    if counterexamples:
        status = "FAILED"
    elif timeout_count or error_count:
        status = "UNKNOWN"
    else:
        status = "VERIFIED"
    return {
        "property": "projected_content_invariance",
        "status": status,
        "proof_strategy": "one_anchor_per_feature_plus_transitivity",
        "verified_anchor_comparisons": verified_comparisons,
        "verified_pairs_by_transitivity": total_pairs if status == "VERIFIED" else 0,
        "assertion_count": assertion_total(assertion_profile),
        "assertion_attribution": assertion_profile,
        "solve_seconds": solve_seconds,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "counterexamples": counterexamples[:5],
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
    """Verify: same structure => same decision (argmax over candidate tokens).

    For quote_close: quote(x) = quote(x') => argmax_T(C(x)) = argmax_T(C(x'))

    NOTE: This implementation samples one pair per feature group for tractability.
    For full-domain verification, all pairs within each group should be checked.

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
    if circuit.get("granularity") == "head":
        return _verify_content_invariance_via_anchors(
            circuit,
            input_sequences,
            model_weights,
            candidate_tokens,
            get_structural_feature,
            timeout_ms,
        )

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
    error_count = 0
    assertion_profile = new_assertion_profile()
    solve_seconds = 0.0

    total_pairs = sum(
        len(sequences) * (len(sequences) - 1) // 2
        for sequences in feature_groups.values()
        if len(sequences) >= 2
    )
    checked_pairs = 0
    progress_interval = 25

    print(
        f"Verifying content invariance across {len(feature_groups)} feature groups "
        f"({total_pairs} pairs)..."
    )

    for feature, sequences in feature_groups.items():
        if len(sequences) < 2:
            continue

        feature_total_pairs = len(sequences) * (len(sequences) - 1) // 2
        feature_checked_pairs = 0
        feature_t0 = time.time()
        print(
            f"  feature {feature}: checking {feature_total_pairs} pairs "
            f"from {len(sequences)} sequences...",
            flush=True,
        )

        # Check all pairs within this feature group
        for a in range(len(sequences)):
            for b in range(a + 1, len(sequences)):
                feature_checked_pairs += 1
                checked_pairs += 1

                if feature_checked_pairs == 1 or feature_checked_pairs % progress_interval == 0:
                    print(
                        f"  feature {feature}: pair {feature_checked_pairs}/{feature_total_pairs} "
                        f"(total {checked_pairs}/{total_pairs})...",
                        flush=True,
                    )

                seq1, _ = sequences[a]
                seq2, _ = sequences[b]

                if len(seq1) != len(seq2):
                    continue  # Skip different lengths

                solver = Solver()
                solver.set("timeout", timeout_ms)

                try:
                    # Encode both sequences
                    logits1 = encode_circuit_forward(
                        seq1,
                        circuit_edges,
                        model_weights,
                        candidate_tokens,
                        solver,
                        f"feat{feature}_a{a}",
                        trace=trace_circuit_forward(seq1, circuit_edges, model_weights, f"feat{feature}_a{a}"),
                    )
                    logits2 = encode_circuit_forward(
                        seq2,
                        circuit_edges,
                        model_weights,
                        candidate_tokens,
                        solver,
                        f"feat{feature}_b{b}",
                        trace=trace_circuit_forward(seq2, circuit_edges, model_weights, f"feat{feature}_b{b}"),
                    )

                    # Add constraint: decisions (argmax) differ
                    # Check if there exist tok1, tok2 where ordering differs
                    ordering_violations = []
                    for tok1 in candidate_tokens:
                        if tok1 not in logits1 or tok1 not in logits2:
                            continue
                        for tok2 in candidate_tokens:
                            if tok2 != tok1 and tok2 in logits1 and tok2 in logits2:
                                # Ordering differs: tok1 > tok2 in seq1 but tok1 <= tok2 in seq2
                                ordering_violations.append(
                                    And(logits1[tok1] > logits1[tok2], logits2[tok1] <= logits2[tok2])
                                )
                                # Or: tok1 <= tok2 in seq1 but tok1 > tok2 in seq2
                                ordering_violations.append(
                                    And(logits1[tok1] <= logits1[tok2], logits2[tok1] > logits2[tok2])
                                )

                    base_result = solver.check()
                    if base_result == unsat:
                        print(f"  ERROR: trace certificate invalid for feature {feature}, pair ({a}, {b})")
                        error_count += 1
                        continue
                    if base_result != sat:
                        timeout_count += 1
                        continue

                    if not ordering_violations:
                        verified_pairs += 1
                        continue

                    solver.push()
                    solver.add(Or(ordering_violations))

                    result = solver.check()
                    solver.pop()

                    if result == unsat:
                        # Property holds: decisions are identical
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
                    print(f"  Error verifying feature {feature}, pair ({a}, {b}): {e}")
                    error_count += 1
                    continue

            # Break outer loop if we have enough counterexamples
            if len(counterexamples) >= 10:
                break

        print(
            f"  feature {feature}: checked {feature_checked_pairs}/{feature_total_pairs} pairs "
            f"in {time.time() - feature_t0:.2f}s",
            flush=True,
        )

        # Break feature loop if we have enough counterexamples
        if len(counterexamples) >= 10:
            break

    if len(counterexamples) > 0:
        status = "FAILED"
    elif timeout_count > 0 or error_count > 0:
        status = "UNKNOWN"
    else:
        status = "VERIFIED"

    return {
        "property": "projected_content_invariance",
        "status": status,
        "verified_pairs": verified_pairs,
        "timeout_count": timeout_count,
        "error_count": error_count,
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
    circuit_edges = parse_circuit_edges(circuit)
    edges = sorted(circuit_edges)

    unnecessary_edges = []
    necessary_edges = []
    unresolved_edges = []
    timeout_count = 0
    error_count = 0
    assertion_profile = new_assertion_profile()
    solve_seconds = 0.0

    print(f"Verifying necessity of {len(edges)} edges...")

    for edge_idx, edge in enumerate(edges):
        if edge_idx % 10 == 0 and edge_idx > 0:
            print(f"  Progress: {edge_idx}/{len(edges)}")

        edge_from, edge_to = edge

        # Create ablated circuit (remove this edge)
        ablated_edges = circuit_edges - {edge}

        # Try to find input where removing edge changes output
        found_witness = False
        had_timeout_or_error = False

        for input_tokens in test_inputs:
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
                    trace=trace_circuit_forward(input_tokens, circuit_edges, model_weights, f"full_e{edge_idx}"),
                    assertion_profile=assertion_profile,
                )

                # Encode ablated circuit
                logits_ablated = encode_circuit_forward(
                    input_tokens,
                    ablated_edges,
                    model_weights,
                    candidate_tokens,
                    solver,
                    f"ablated_e{edge_idx}",
                    trace=trace_circuit_forward(input_tokens, ablated_edges, model_weights, f"ablated_e{edge_idx}"),
                    assertion_profile=assertion_profile,
                )

                # Add constraint: decisions (argmax) differ
                # Check if ordering differs between full and ablated circuits
                ordering_violations = []
                for tok1 in candidate_tokens:
                    if tok1 not in logits_full or tok1 not in logits_ablated:
                        continue
                    for tok2 in candidate_tokens:
                        if tok2 != tok1 and tok2 in logits_full and tok2 in logits_ablated:
                            # Ordering differs: tok1 > tok2 in full but tok1 <= tok2 in ablated
                            ordering_violations.append(
                                And(logits_full[tok1] > logits_full[tok2],
                                    logits_ablated[tok1] <= logits_ablated[tok2])
                            )
                            # Or: tok1 <= tok2 in full but tok1 > tok2 in ablated
                            ordering_violations.append(
                                And(logits_full[tok1] <= logits_full[tok2],
                                    logits_ablated[tok1] > logits_ablated[tok2])
                            )

                solve_started = time.time()
                base_result = solver.check()
                solve_seconds += time.time() - solve_started
                if base_result == unsat:
                    print(f"  ERROR: full/ablated trace certificate invalid for edge {edge}")
                    error_count += 1
                    had_timeout_or_error = True
                    break
                if base_result != sat:
                    timeout_count += 1
                    had_timeout_or_error = True
                    continue

                if not ordering_violations:
                    continue

                solver.push()
                add_profiled_assertion(
                    assertion_profile, "decision", solver, Or(ordering_violations)
                )

                solve_started = time.time()
                result = solver.check()
                solve_seconds += time.time() - solve_started
                solver.pop()

                if result == sat:
                    # Decisions differ - edge is necessary
                    found_witness = True
                    necessary_edges.append(edge)
                    break
                elif result == unsat:
                    # Outputs same on this input - try next input
                    continue
                else:
                    timeout_count += 1
                    had_timeout_or_error = True

            except Exception as e:
                print(f"  Error testing edge {edge}: {e}")
                error_count += 1
                had_timeout_or_error = True
                break

        # Categorize edge based on results
        if found_witness:
            pass  # Already added to necessary_edges
        elif had_timeout_or_error:
            unresolved_edges.append({
                "edge": f"{edge_from} -> {edge_to}",
                "tested_inputs": len(test_inputs),
            })
        else:
            unnecessary_edges.append({
                "edge": f"{edge_from} -> {edge_to}",
                "tested_inputs": len(test_inputs),
            })

    if len(unnecessary_edges) > 0:
        status = "FAILED"
    elif len(unresolved_edges) > 0 or timeout_count > 0 or error_count > 0:
        status = "UNKNOWN"
    else:
        status = "VERIFIED"

    return {
        "property": "projected_edge_necessity",
        "status": status,
        "total_edges": len(edges),
        "necessary_edges": len(necessary_edges),
        "unnecessary_edges_found": len(unnecessary_edges),
        "unresolved_edges_found": len(unresolved_edges),
        "timeout_count": timeout_count,
        "error_count": error_count,
        "unnecessary_edges": unnecessary_edges[:10],
        "unresolved_edges": unresolved_edges[:10],
        "assertion_count": assertion_total(assertion_profile),
        "assertion_attribution": assertion_profile,
        "solve_seconds": solve_seconds,
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
    timeout_count = 0
    error_count = 0
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
                seq,
                circuit_edges,
                model_weights,
                candidate_tokens,
                solver,
                f"orig_{seq_idx}",
                trace=trace_circuit_forward(seq, circuit_edges, model_weights, f"orig_{seq_idx}"),
            )

            # Encode permuted sequence
            logits_perm = encode_circuit_forward(
                seq_permuted,
                circuit_edges,
                model_weights,
                candidate_tokens,
                solver,
                f"perm_{seq_idx}",
                trace=trace_circuit_forward(seq_permuted, circuit_edges, model_weights, f"perm_{seq_idx}"),
            )

            # Check: logits_perm[20] should equal logits_orig[21]
            #        logits_perm[21] should equal logits_orig[20]
            if 20 in logits_orig and 21 in logits_orig and 20 in logits_perm and 21 in logits_perm:
                base_result = solver.check()
                if base_result == unsat:
                    print(f"  ERROR: trace certificate invalid on sequence {seq_idx}")
                    error_count += 1
                    continue
                if base_result != sat:
                    timeout_count += 1
                    continue

                solver.push()
                solver.add(Or(
                    logits_perm[20] != logits_orig[21],
                    logits_perm[21] != logits_orig[20],
                ))

                result = solver.check()
                solver.pop()

                if result == unsat:
                    verified_count += 1
                elif result == sat:
                    counterexamples.append({"sequence": seq})
                    if len(counterexamples) >= 10:
                        break
                else:
                    timeout_count += 1

        except Exception as e:
            print(f"  Error on sequence {seq_idx}: {e}")
            error_count += 1
            continue

    if len(counterexamples) > 0:
        status = "FAILED"
    elif timeout_count > 0 or error_count > 0:
        status = "UNKNOWN"
    else:
        status = "VERIFIED"

    return {
        "property": "token_renaming_equivariance",
        "status": status,
        "verified_count": verified_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
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
    timeout_count = 0
    error_count = 0
    assertion_profile = new_assertion_profile()
    solve_seconds = 0.0

    print(f"Verifying {constraint_name} on {len(test_inputs)} inputs...")

    for i, input_tokens in enumerate(test_inputs):
        if i % 50 == 0 and i > 0:
            print(f"  Progress: {i}/{len(test_inputs)}")

        solver = Solver()
        solver.set("timeout", timeout_ms)

        try:
            # Encode circuit
            logits = encode_circuit_forward(
                input_tokens,
                circuit_edges,
                model_weights,
                candidate_tokens,
                solver,
                f"input_{i}",
                trace=trace_circuit_forward(input_tokens, circuit_edges, model_weights, f"input_{i}"),
            )

            base_result = solver.check()
            if base_result == unsat:
                print(f"  ERROR: trace certificate invalid on input {i}")
                error_count += 1
                continue
            if base_result != sat:
                timeout_count += 1
                continue

            # Add constraint and negate it (check if constraint can be violated)
            solver.push()
            constraint_checker(logits, solver)

            result = solver.check()
            solver.pop()

            if result == unsat:
                # Constraint cannot be violated
                verified_count += 1
            elif result == sat:
                # Constraint violated
                violations.append({"input": input_tokens})
                if len(violations) >= 10:
                    break
            else:
                timeout_count += 1

        except Exception as e:
            print(f"  Error on input {i}: {e}")
            error_count += 1
            continue

    if len(violations) > 0:
        status = "FAILED"
    elif timeout_count > 0 or error_count > 0:
        status = "UNKNOWN"
    else:
        status = "VERIFIED"

    return {
        "property": constraint_name,
        "status": status,
        "verified_count": verified_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "violations": violations[:5],
        "num_violations": len(violations),
    }


def verify_continuous_robustness(
    circuit: Dict[str, Any],
    test_inputs: List[List[int]],
    model_weights: Dict[str, Any],
    candidate_tokens: List[int],
    epsilon: float = 0.01,
    timeout_ms: int = 60000,
) -> Dict[str, Any]:
    """Verify continuous robustness to perturbations of final residual.

    Property: ∀x, ∀η: ||η||_∞ ≤ ε ⇒ g_T(r_E(x)+η) = g_T(r_E(x))

    where r_E(x) is the final residual before ln_f, and
    g_T(r) = argmax_{t∈T}(W_U ln_f(r))_t

    Args:
        circuit: Extracted circuit
        test_inputs: Test input sequences
        model_weights: Model weights
        candidate_tokens: List of candidate output token IDs
        epsilon: Perturbation radius (L-infinity norm)
        timeout_ms: Solver timeout

    Returns:
        Verification result dict
    """
    circuit_edges = parse_circuit_edges(circuit)

    branch_unstable = []
    decision_violations = []
    verified_count = 0
    timeout_count = 0
    error_count = 0
    assertion_profile = new_assertion_profile()
    solve_seconds = 0.0

    d_model = model_weights["d_model"]
    print(f"Verifying continuous robustness on {len(test_inputs)} inputs...")
    print(f"Perturbation radius: ε = {epsilon}\n")

    for i, input_tokens in enumerate(test_inputs):
        if i % 10 == 0 and i > 0:
            print(f"  Progress: {i}/{len(test_inputs)}")

        try:
            trace = trace_circuit_forward(input_tokens, circuit_edges, model_weights, f"rob_{i}")
            final_residual = [z3_real(v) for v in trace["final_residual"]]
            lm_head = model_weights["lm_head"]
            lm_bias = model_weights.get("lm_head_bias", [0.0] * len(lm_head))

            if model_weights.get("norm_variant") == "none":
                # With ln_f folded away, g_T is a single affine map followed by
                # candidate argmax. There is no branch certificate to validate
                # and a linear solver decides the entire epsilon box directly.
                decision_solver = Solver()
                decision_solver.set("timeout", timeout_ms)
                eta = [Real(f"rob_{i}_eta_{j}") for j in range(d_model)]
                for j in range(d_model):
                    add_profiled_assertion(
                        assertion_profile,
                        "decision",
                        decision_solver,
                        eta[j] >= -z3_real(epsilon),
                        eta[j] <= z3_real(epsilon),
                    )
                perturbed = [final_residual[j] + eta[j] for j in range(d_model)]
                logits_original = {
                    tok: Sum(
                        [z3_real(lm_head[tok][j]) * final_residual[j] for j in range(d_model)]
                    )
                    + z3_real(lm_bias[tok])
                    for tok in candidate_tokens
                }
                logits_perturbed = {
                    tok: Sum(
                        [z3_real(lm_head[tok][j]) * perturbed[j] for j in range(d_model)]
                    )
                    + z3_real(lm_bias[tok])
                    for tok in candidate_tokens
                }
                decision_flips = []
                for tok1 in candidate_tokens:
                    for tok2 in candidate_tokens:
                        if tok1 == tok2:
                            continue
                        decision_flips.extend(
                            [
                                And(
                                    logits_original[tok1] > logits_original[tok2],
                                    logits_perturbed[tok1] <= logits_perturbed[tok2],
                                ),
                                And(
                                    logits_original[tok1] <= logits_original[tok2],
                                    logits_perturbed[tok1] > logits_perturbed[tok2],
                                ),
                            ]
                        )
                add_profiled_assertion(
                    assertion_profile,
                    "decision",
                    decision_solver,
                    Or(decision_flips),
                )
                solve_started = time.time()
                decision_result = decision_solver.check()
                solve_seconds += time.time() - solve_started
                if decision_result == unsat:
                    verified_count += 1
                elif decision_result == sat:
                    decision_violations.append(
                        {"input": input_tokens, "decision_flip_found": True}
                    )
                    if len(decision_violations) >= 10:
                        break
                else:
                    timeout_count += 1
                continue

            final_trace = trace["bandnorm"][f"rob_{i}_logits_norm"]
            norm_gamma = model_weights["final_norm_gamma"]
            norm_beta = model_weights["final_norm_beta"]
            half_low = model_weights["half_low"]
            half_high = model_weights["half_high"]
            pos_fallback = [1.0 if k % 2 == 0 else 0.0 for k in range(d_model)]
            neg_fallback = [0.0 if k % 2 == 0 else 1.0 for k in range(d_model)]

            # Check 1: final BandNorm branch is stable throughout the epsilon ball.
            stability_solver = Solver()
            stability_solver.set("timeout", timeout_ms)
            eta = [Real(f"rob_{i}_stable_eta_{j}") for j in range(d_model)]
            for j in range(d_model):
                add_profiled_assertion(
                    assertion_profile,
                    "decision",
                    stability_solver,
                    eta[j] >= -z3_real(epsilon),
                    eta[j] <= z3_real(epsilon),
                )
            perturbed_residual = [final_residual[j] + eta[j] for j in range(d_model)]
            guards = signed_l1_band_norm_guard_conditions(
                perturbed_residual,
                half_low,
                half_high,
                final_trace,
            )
            increment_norm_instances(assertion_profile)
            add_profiled_assertion(
                assertion_profile,
                "norm",
                stability_solver,
                Or([Not(g) for g in guards]),
            )
            solve_started = time.time()
            stability_result = stability_solver.check()
            solve_seconds += time.time() - solve_started

            if stability_result == sat:
                branch_unstable.append({
                    "input": input_tokens,
                    "branch_instability_found": True,
                })
                continue
            if stability_result != unsat:
                timeout_count += 1
                continue

            # Check 2: under the stable certified branch, no candidate decision flips.
            decision_solver = Solver()
            decision_solver.set("timeout", timeout_ms)
            eta = [Real(f"rob_{i}_decision_eta_{j}") for j in range(d_model)]
            for j in range(d_model):
                add_profiled_assertion(
                    assertion_profile,
                    "decision",
                    decision_solver,
                    eta[j] >= -z3_real(epsilon),
                    eta[j] <= z3_real(epsilon),
                )
            perturbed_residual = [final_residual[j] + eta[j] for j in range(d_model)]

            increment_norm_instances(assertion_profile)
            norm_before = len(decision_solver.assertions())
            normed_original = encode_signed_l1_band_norm_with_trace(
                final_residual,
                norm_gamma,
                norm_beta,
                half_low,
                half_high,
                pos_fallback,
                neg_fallback,
                final_trace,
                decision_solver,
                f"rob_{i}_norm_orig",
            )
            record_solver_delta(
                assertion_profile, "norm", decision_solver, norm_before
            )
            increment_norm_instances(assertion_profile)
            norm_before = len(decision_solver.assertions())
            normed_perturbed = encode_signed_l1_band_norm_with_trace(
                perturbed_residual,
                norm_gamma,
                norm_beta,
                half_low,
                half_high,
                pos_fallback,
                neg_fallback,
                final_trace,
                decision_solver,
                f"rob_{i}_norm_pert",
            )
            record_solver_delta(
                assertion_profile, "norm", decision_solver, norm_before
            )

            logits_original = {
                tok: Sum([z3_real(lm_head[tok][j]) * normed_original[j] for j in range(d_model)])
                for tok in candidate_tokens
            }
            logits_perturbed = {
                tok: Sum([z3_real(lm_head[tok][j]) * normed_perturbed[j] for j in range(d_model)])
                for tok in candidate_tokens
            }

            decision_flips = []
            for tok1 in candidate_tokens:
                for tok2 in candidate_tokens:
                    if tok1 != tok2:
                        decision_flips.append(
                            And(logits_original[tok1] > logits_original[tok2],
                                logits_perturbed[tok1] <= logits_perturbed[tok2])
                        )
                        decision_flips.append(
                            And(logits_original[tok1] <= logits_original[tok2],
                                logits_perturbed[tok1] > logits_perturbed[tok2])
                        )

            if not decision_flips:
                verified_count += 1
                continue

            solve_started = time.time()
            base_result = decision_solver.check()
            solve_seconds += time.time() - solve_started
            if base_result == unsat:
                print(f"  ERROR: robustness trace certificate invalid on input {i}")
                error_count += 1
                continue
            if base_result != sat:
                timeout_count += 1
                continue

            decision_solver.push()
            add_profiled_assertion(
                assertion_profile,
                "decision",
                decision_solver,
                Or(decision_flips),
            )
            solve_started = time.time()
            decision_result = decision_solver.check()
            solve_seconds += time.time() - solve_started
            decision_solver.pop()

            if decision_result == unsat:
                verified_count += 1
            elif decision_result == sat:
                decision_violations.append({
                    "input": input_tokens,
                    "decision_flip_found": True,
                })
                if len(decision_violations) >= 10:
                    break
            else:
                timeout_count += 1

        except Exception as e:
            print(f"  Error on input {i}: {e}")
            error_count += 1
            continue

    if len(decision_violations) > 0:
        status = "FAILED"
    elif len(branch_unstable) > 0:
        status = "UNKNOWN_BRANCH_UNSTABLE"
    elif timeout_count > 0 or error_count > 0:
        status = "UNKNOWN"
    else:
        status = "VERIFIED"

    return {
        "property": "projected_continuous_robustness",
        "status": status,
        "verified_count": verified_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "certified_epsilon": epsilon,
        "violations": decision_violations[:5],
        "num_violations": len(decision_violations),
        "decision_violations": decision_violations[:5],
        "num_decision_violations": len(decision_violations),
        "branch_unstable": branch_unstable[:5],
        "num_branch_unstable": len(branch_unstable),
        "branch_certificates_required": model_weights.get("norm_variant") != "none",
        "assertion_count": assertion_total(assertion_profile),
        "assertion_attribution": assertion_profile,
        "solve_seconds": solve_seconds,
    }
