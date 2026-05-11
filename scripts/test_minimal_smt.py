#!/usr/bin/env python3
"""
Minimal SMT encoding test to diagnose BandNorm complexity issues.

Tests the absolute simplest case: single token, to see if BandNorm
SMT encoding is feasible at any scale.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from z3 import *
from scripts.smt_verify import encode_circuit_forward
from scripts.smt_verify.model_weights import load_model_weights
from scripts.smt_verify.helpers import parse_circuit_edges
import json
import time

# Load circuit
circuit_path = 'artifacts/circuits_sweep/quote_close_t0.02/circuit.json'
print(f"Loading circuit from {circuit_path}...")
with open(circuit_path) as f:
    circuit = json.load(f)

print(f"Circuit has {circuit['num_edges']} edges")
circuit_edges = parse_circuit_edges(circuit)

# Load weights
model_path = 'artifacts/step2c-band-norm-sparsemax/checkpoint-240000'
print(f"Loading weights from {model_path}...")
weights = load_model_weights(model_path)
print(f"Loaded model: {weights['n_layers']} layers, {weights['d_model']} dims")

# Simplest possible test: single token
input_tokens = [6]
candidate_tokens = [6, 1]

print(f"\nEncoding circuit for input {input_tokens}...")
print(f"Candidate tokens: {candidate_tokens}")

solver = Solver()
solver.set("timeout", 10000)

import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Encoding timed out after 30s")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

start = time.time()
print("  Building SMT formula (30s timeout)...", flush=True)
try:
    logits = encode_circuit_forward(
        input_tokens,
        circuit_edges,
        weights,
        candidate_tokens,
        solver,
        "test"
    )
    signal.alarm(0)  # Cancel alarm
    encode_time = time.time() - start
    num_constraints = len(solver.assertions())
    print(f"  Formula built in {encode_time:.1f}s", flush=True)

    print(f"✓ Encoding succeeded in {encode_time:.1f}s")
    print(f"  Generated {num_constraints} constraints")

    print("\nSolving...")
    start = time.time()
    result = solver.check()
    solve_time = time.time() - start

    print(f"✓ Solver result: {result} in {solve_time:.1f}s")

    if result == sat:
        model = solver.model()
        print(f"\nLogits:")
        for tok in candidate_tokens:
            if tok in logits:
                val = model.eval(logits[tok], model_completion=True)
                if hasattr(val, "as_fraction"):
                    logit_val = float(val.as_fraction())
                else:
                    logit_val = float(val.as_decimal(20).rstrip("?"))
                print(f"  Token {tok}: {logit_val:.4f}")

except TimeoutError as e:
    encode_time = time.time() - start
    print(f"\n✗ TIMEOUT after {encode_time:.1f}s during encoding")
    print("BandNorm SMT encoding is too complex - encoding never completes")
    print("The conditional logic in BandNorm creates an intractable constraint system")
    sys.exit(1)

except KeyboardInterrupt:
    encode_time = time.time() - start
    print(f"\n✗ INTERRUPTED after {encode_time:.1f}s during encoding")
    print("BandNorm SMT encoding appears too complex for practical use")
    sys.exit(1)

except Exception as e:
    encode_time = time.time() - start
    print(f"✗ FAILED after {encode_time:.1f}s: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ SUCCESS: BandNorm SMT encoding is feasible for length-1 inputs")
