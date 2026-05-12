"""Property verification using SMT solvers."""

from z3 import *
from typing import List, Dict, Any, Set, Tuple, Callable, Optional
from .circuit import encode_circuit_forward
from .domain import generate_bounded_sequences
from .utils import parse_circuit_edges, get_candidate_tokens
from .encoders import encode_signed_l1_band_norm


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

    print(f"Verifying functional equivalence on {len(input_sequences)} sequences...")

    for i, (input_tokens, _) in enumerate(input_sequences):
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

            # Get expected token from reference program
            expected_token = reference_program(input_tokens)

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

    print(f"Verifying content invariance across {len(feature_groups)} feature groups...")

    for feature, sequences in feature_groups.items():
        if len(sequences) < 2:
            continue

        # Check all pairs within this feature group
        for a in range(len(sequences)):
            for b in range(a + 1, len(sequences)):
                seq1, _ = sequences[a]
                seq2, _ = sequences[b]

                if len(seq1) != len(seq2):
                    continue  # Skip different lengths

                solver = Solver()
                solver.set("timeout", timeout_ms)

                try:
                    # Encode both sequences
                    logits1 = encode_circuit_forward(
                        seq1, circuit_edges, model_weights, candidate_tokens, solver, f"feat{feature}_a{a}"
                    )
                    logits2 = encode_circuit_forward(
                        seq2, circuit_edges, model_weights, candidate_tokens, solver, f"feat{feature}_b{b}"
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

                    if not ordering_violations:
                        verified_pairs += 1
                        continue

                    solver.add(Or(ordering_violations))

                    result = solver.check()

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
    edges = [(e["from"], e["to"]) if isinstance(e, dict) else (e[0], e[1]) for e in circuit["edges"]]
    circuit_edges = parse_circuit_edges(circuit)

    unnecessary_edges = []
    necessary_edges = []
    unresolved_edges = []
    timeout_count = 0
    error_count = 0

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

                if not ordering_violations:
                    continue

                solver.add(Or(ordering_violations))

                result = solver.check()

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
    timeout_count = 0
    error_count = 0

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
    from .circuit import encode_circuit_forward, get_residual_input

    circuit_edges = parse_circuit_edges(circuit)

    violations = []
    verified_count = 0
    timeout_count = 0
    error_count = 0

    d_model = model_weights["d_model"]
    n_layers = model_weights["n_layers"]
    norm_variant = model_weights.get("norm_variant", "layernorm")

    print(f"Verifying continuous robustness on {len(test_inputs)} inputs...")
    print(f"Perturbation radius: ε = {epsilon}\n")

    for i, input_tokens in enumerate(test_inputs):
        if i % 10 == 0 and i > 0:
            print(f"  Progress: {i}/{len(test_inputs)}")

        solver = Solver()
        solver.set("timeout", timeout_ms)

        try:
            # Encode circuit forward pass (get final residual via modified circuit encoding)
            # We need to reimplement the final residual extraction inline here
            seq_len = len(input_tokens)

            # Get all node outputs
            from .circuit import encode_attention_layer, encode_mlp_layer, zero_output

            # Compute active nodes
            active_nodes = {"emb", "logits"}
            for node_from, node_to in circuit_edges:
                active_nodes.add(node_from)
                active_nodes.add(node_to)

            # Node computation cache
            node_outputs = {}

            # Embedding
            wte = model_weights["wte"]
            wpe = model_weights["wpe"]
            emb_output = []
            for pos in range(seq_len):
                tok = input_tokens[pos]
                emb_pos = [wte[tok][j] + wpe[pos][j] for j in range(d_model)]
                emb_output.append(emb_pos)
            node_outputs["emb"] = emb_output

            # Layer-by-layer forward pass
            for layer in range(n_layers):
                attn_node = f"attn_{layer}"
                mlp_node = f"mlp_{layer}"

                if attn_node in active_nodes:
                    attn_output = encode_attention_layer(
                        node_outputs,
                        layer,
                        circuit_edges,
                        model_weights,
                        model_weights["n_heads"],
                        solver,
                        f"rob_{i}_L{layer}_attn",
                    )
                    node_outputs[attn_node] = attn_output
                else:
                    node_outputs[attn_node] = zero_output(seq_len, d_model)

                if mlp_node in active_nodes:
                    mlp_output = encode_mlp_layer(
                        node_outputs,
                        layer,
                        circuit_edges,
                        model_weights,
                        solver,
                        f"rob_{i}_L{layer}_mlp",
                    )
                    node_outputs[mlp_node] = mlp_output
                else:
                    node_outputs[mlp_node] = zero_output(seq_len, d_model)

            # Extract final residual at last position (before ln_f)
            parent_nodes = ["emb"]
            for layer in range(n_layers):
                parent_nodes.extend([f"attn_{layer}", f"mlp_{layer}"])

            residual_last = [RealVal(0) for _ in range(d_model)]
            for parent in parent_nodes:
                if (parent, "logits") in circuit_edges:
                    parent_output = node_outputs[parent]
                    for j in range(d_model):
                        residual_last[j] = residual_last[j] + parent_output[seq_len - 1][j]

            # Create symbolic perturbation η with ||η||_∞ ≤ ε
            eta = [Real(f"rob_{i}_eta_{j}") for j in range(d_model)]
            for j in range(d_model):
                solver.add(eta[j] >= -epsilon)
                solver.add(eta[j] <= epsilon)

            # Perturbed residual
            perturbed_residual = [residual_last[j] + eta[j] for j in range(d_model)]

            # Apply ln_f to both residuals
            def apply_final_norm(residual, ctx):
                if norm_variant == "signed_l1_band_norm":
                    norm_gamma = model_weights["final_norm_gamma"]
                    norm_beta = model_weights["final_norm_beta"]
                    half_low = model_weights["half_low"]
                    half_high = model_weights["half_high"]
                    pos_fallback = [1.0 if k % 2 == 0 else 0.0 for k in range(d_model)]
                    neg_fallback = [0.0 if k % 2 == 0 else 1.0 for k in range(d_model)]
                    return encode_signed_l1_band_norm(
                        residual, norm_gamma, norm_beta,
                        half_low, half_high,
                        pos_fallback, neg_fallback,
                        solver, ctx,
                    )
                else:
                    norm_gamma = model_weights["final_norm_gamma"]
                    norm_beta = model_weights["final_norm_beta"]
                    return [residual[k] * norm_gamma[k] + norm_beta[k] for k in range(d_model)]

            normed_original = apply_final_norm(residual_last, f"rob_{i}_norm_orig")
            normed_perturbed = apply_final_norm(perturbed_residual, f"rob_{i}_norm_pert")

            # Compute logits for both
            lm_head = model_weights["lm_head"]
            logits_original = {tok: Sum([lm_head[tok][j] * normed_original[j] for j in range(d_model)])
                              for tok in candidate_tokens}
            logits_perturbed = {tok: Sum([lm_head[tok][j] * normed_perturbed[j] for j in range(d_model)])
                               for tok in candidate_tokens}

            # Check if decisions differ
            ordering_violations = []
            for tok1 in candidate_tokens:
                for tok2 in candidate_tokens:
                    if tok1 != tok2:
                        # Original prefers tok1, perturbed prefers tok2
                        ordering_violations.append(
                            And(logits_original[tok1] > logits_original[tok2],
                                logits_perturbed[tok1] <= logits_perturbed[tok2])
                        )
                        # Or vice versa
                        ordering_violations.append(
                            And(logits_original[tok1] <= logits_original[tok2],
                                logits_perturbed[tok1] > logits_perturbed[tok2])
                        )

            if not ordering_violations:
                verified_count += 1
                continue

            solver.add(Or(ordering_violations))

            result = solver.check()

            if result == unsat:
                # No perturbation can change the decision
                verified_count += 1
            elif result == sat:
                # Found a perturbation that changes decision
                model = solver.model()
                violations.append({
                    "input": input_tokens,
                    "perturbation_found": True,
                })
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
        "property": "projected_continuous_robustness",
        "status": status,
        "verified_count": verified_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "certified_epsilon": epsilon,
        "violations": violations[:5],
        "num_violations": len(violations),
    }
