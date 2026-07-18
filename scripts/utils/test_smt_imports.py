#!/usr/bin/env python3
"""Test that SMT verification module imports work correctly.

NOTE: This is for development/debugging of the SMT encoding logic itself.
Actual circuit verification requires real trained model weights; see
scripts/small/verify.py or scripts/gpt2/verify.py.
"""

import sys
import os

# Add repository root to path when this file is executed directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def test_imports():
    """Test all SMT module imports."""
    print("Testing SMT verification module imports...")

    try:
        from scripts.smt import (
            encode_leaky_relu,
            encode_signed_l1_band_norm,
            encode_sparsemax,
            encode_multihead_attention_sparsemax,
            encode_mlp,
            encode_nonnegative_l1_projection,
            encode_additive_lift,
            encode_circuit_forward,
            verify_functional_equivalence,
            verify_content_invariance,
            verify_edge_necessity,
            verify_token_renaming_equivariance,
            verify_structural_constraint,
            generate_bounded_sequences,
            generate_quote_close_sequences,
            generate_bracket_type_sequences,
            generate_induction_sequences,
            enumerate_small_domain,
        )
        print("✓ All SMT module imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_sequence_generation():
    """Test sequence generation."""
    print("\nTesting sequence generation...")

    try:
        from scripts.smt import (
            generate_quote_close_sequences,
            generate_induction_sequences,
        )

        # Test quote close sequences
        special_tokens = {
            "single_quote": 10,
            "double_quote": 11,
            "content_tokens": [12, 13],
        }
        sequences = generate_quote_close_sequences(max_length=5, special_tokens=special_tokens)
        print(f"✓ Generated {len(sequences)} quote close sequences")

        # Test induction sequences
        vocab = set(range(20, 30))
        sequences = generate_induction_sequences(max_length=6, vocab=vocab)
        print(f"✓ Generated {len(sequences)} induction sequences")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_z3_encoders():
    """Test basic Z3 encoders."""
    print("\nTesting Z3 encoders...")

    try:
        from z3 import Real, Solver, sat
        from scripts.smt import encode_leaky_relu, encode_sparsemax

        # Test LeakyReLU
        x = Real('x')
        y = encode_leaky_relu(x, alpha=0.01)
        print("✓ LeakyReLU encoder works")

        # Test sparsemax
        solver = Solver()
        logits = [Real(f'z_{i}') for i in range(3)]
        output = encode_sparsemax(logits, solver, "test")
        print("✓ Sparsemax encoder works")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("SMT Verification Module Tests")
    print("=" * 80)

    results = []
    results.append(("Module imports", test_imports()))
    results.append(("Sequence generation", test_sequence_generation()))
    results.append(("Z3 encoders", test_z3_encoders()))

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
