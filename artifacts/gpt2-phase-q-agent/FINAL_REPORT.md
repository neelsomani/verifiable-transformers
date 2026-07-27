# Phase Q terminal report

Date: 2026-07-27

Status: **complete**

Phase Q produced a formally verified GPT-2-derived quote-closing circuit over
the frozen, hash-pinned 1,280-prompt domain D. The final circuit contains one
symbolic attention head, no neural-attention bilinear terms, and no
normalization. Functional equivalence, content invariance, edge necessity, and
continuous robustness all pass over D.

The claim is bounded to D and the registered two-token quote projection. It is
not a held-out-generalization or unrestricted-language claim.

## 1. Healing-based route

The preregistered core-aware healing attempts did not produce a causally pinned
GPT-2-scale mechanism. One run failed destructively. After correcting a
fourfold gradient-accumulation mis-scaling and adding the omitted full-forward
task loss, an adaptive run recovered exact full and circuit-only behavior
within the perplexity budget, but the installed programs remained jointly
bypassable in the full model.

That negative result is preserved. It establishes the scale limit of
healing-based mechanism migration in this experiment; it is not the final
Phase Q result.

## 2. Pre-healing causal audit

The missing pre-intervention lesion baselines were measured before any further
optimization:

- jointly ablating original heads `7.11` and `9.0` in the untouched model
  leaves 1,256/1,280 prompts correct;
- the programs-installed zero-step model has the same 1,256/1,280 joint-lesion
  result;
- ablating the whole selected circuit leaves 710/1,280 prompts correct; and
- all 23 zero-step full-model errors lie within the 24 prompts not covered by
  the native redundant route.

The 98.1%-coverage backup therefore predates healing. Healing did not create
the redundancy; it closed the final 24 cases.

## 3. Constrained calibration

The registered follow-up freezes and hash-verifies every non-program parameter.
Only program-local contributions can train, and every trainable contribution
vanishes under the corresponding program lesion. A bypass cannot be learned
without violating the parameter and lesion-identity gates.

All three registered rungs reached exact 1,280/1,280 full and circuit-only
agreement under the locked OpenWebText perplexity budget:

| Rung | Trainable parameters | OWT perplexity | Budget |
|---|---|---:|---:|
| 1 | two scalar output gains | 25.6691 | 28.6176 |
| 2 | per-channel diagonal gains | 25.7157 | 28.6176 |
| 3 | program-local W_V/W_O | 25.5707 | 28.6176 |

Rung 3 is the model of record. Its joint-head lesion remains exactly
1,256/1,280 and its registered whole-circuit lesion remains exactly 710/1,280,
matching the frozen baseline.

The original individual-head-necessity gate stopped the first continuation
because `9.0` is individually redundant even inside the circuit, while `7.11`
is load-bearing. Under a documented amendment, that criterion was replaced for
the constrained design by:

1. the leakage-identity battery;
2. joint necessity of the program-head set;
3. whole-circuit necessity; and
4. circuit-internal edge necessity as a formal verification property.

The amendment does not relax task exactness, the perplexity budget, lesion
identity, or any formal property. It removes a requirement that calibration
manufacture non-redundancy absent from the original model.

Audit and calibration commits: `daf939e`, `ee35774`.

## 4. Re-extracted circuit

Exact re-extraction on D selected three edges:

```text
emb → mlp_0 → attn_7_h_11 → logits
```

Head `7.11` is a frozen restricted-DSL program using manifest-scoped
token-identity scanning over the four registered quote-opener token IDs. Head
`9.0` remains a frozen program in the full checkpoint but lies outside the
selected circuit, consistent with the lesion asymmetry.

The selected circuit has:

- one program attention head;
- zero neural attention heads;
- zero QK/value-aggregation bilinear terms; and
- zero normalization branches.

Its mechanism shape qualitatively matches the string-closing organization
reported by Gao et al. for weight-sparse models—early-MLP quote features
followed by one copy head—here found independently in a densely trained model.
This is a mechanism-shape comparison, not a circuit-stability claim.

## 5. Exact verification

The FP32 encoder agrees with PyTorch to maximum candidate-logit error
1.11 × 10⁻⁸. The registered constant-folded verifier certifies every used
LeakyReLU sign in exact rational arithmetic, contracts each concrete forward
to exact rational candidate logits, and reduces every property to decidable
linear real arithmetic. Folded and independent monolithic evaluators agree by
literal rational equality on all eight preserved synthetic anchors.

All four registered properties pass:

| Property | Result |
|---|---|
| Functional equivalence | 1,280/1,280 |
| Content invariance | 1,280/1,280 |
| Edge necessity | all 3 edges; 640 decision-changing witnesses per edge |
| Continuous robustness | 1,280/1,280 at ε = 0.01 |

The minimum exact certified robustness radius is approximately 0.01514986.
The proof records attribute 5,911,040 assertions and contain 62 unique exact
MLP sign signatures. No *unknown* result is reachable.

One legacy length-3 sequence, `[10, 1, 10]`, is a genuine counterexample
outside D. It is reproduced exactly by both evaluators and preserved as a
constructive exhibit of the declared domain boundary.

Verification commits: `bc7cfaf`, `0374707`, `7f237a1`.

## 6. Bracket localization boundary

The held-out-generalization track ended at protocol v4. The full norm-free
model was exact on both fresh 512-prompt gates. The exactly-generalizing
bracket circuit retained 340 edges and all 144 attention heads, while the
sparse quote circuit missed one fresh prompt. There is no protocol v5.

Bracket type is therefore reported as a localization boundary rather than a
verified GPT-2-scale circuit. It does not weaken the bounded quote result.

## 7. Model and evidence

The FP32 model-of-record weights are intentionally outside Git:

```text
artifacts/gpt2-phase-q-readside-calibration/fp32_export/model.safetensors
```

SHA-256:

```text
bcec649b087b984a986760a18aedad2477bd118460b33067e9186cd633f8e65d
```

The complete hash and claim root is:

```text
artifacts/gpt2-phase-q-agent/evidence_manifest.json
```

Principal verification artifacts:

- `artifacts/gpt2-phase-q-readside-calibration/FOLDED_VERIFICATION_REPORT.md`
- `artifacts/gpt2-phase-q-readside-calibration/folded_verification/summary.json`
- `artifacts/gpt2-phase-q-readside-calibration/folded_verification/per_input.jsonl`
- `artifacts/gpt2-phase-q-readside-calibration/folded_verification/anchor_equality.json`
- `artifacts/gpt2-phase-q-readside-calibration/encoder_sanity.json`
- `artifacts/gpt2-phase-q-readside-calibration/reextraction/selection.json`
- `artifacts/gpt2-phase-q-readside-calibration/rung3_causal_migration.json`
- `artifacts/gpt2-phase-q-readside-calibration/rung3_program_local.json`
- `artifacts/gpt2-phase-q-readside-calibration/amended_causal_replay.json`

The earlier healing failures, causal audit, criterion amendments, and legacy
synthetic-domain counterexample remain preserved alongside the final result.
