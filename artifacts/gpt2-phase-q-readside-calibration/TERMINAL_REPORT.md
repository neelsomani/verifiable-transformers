# Historical preflight stop (superseded by the completed continuation)

This preserved report records the first constrained causal-integration
preflight stop. A registered paired-baseline amendment resolved the numerical
reproducibility issue without changing epsilon, and the continuation ultimately
completed the causal gates and formal verification described below.

The initial constrained causal-integration follow-up stopped at the mandatory
deterministic FP32 preflight. No search, optimization, calibration step, OWT
evaluation, migration sweep, extraction, or SMT verification was run.

## Causal evidence

The implemented rungs 1–2 read-side path decomposes a programmed head's
bias-free `W_O` contribution at a specific retained reader. A scalar or
768-channel diagonal transform is applied there, without changing the head
residual write seen by other readers. Focused tests prove:

- scalar calibration changes only the selected reader contribution;
- diagonal calibration is channelwise after `W_O`;
- selected-edge zero makes the calibration delta identically zero; and
- an unselected reader is bitwise unchanged.

Together with the existing per-head and pre-healing audit tests, 11 focused
tests passed.

No trained causal claim is made because the preflight stop preceded every
ladder rung.

## Task and identity result

The frozen state-B decisions and mismatch-ID sets reproduced exactly for all
four preflight paths:

| Path | Correct | Decisions/IDs | Max stored-margin difference | `1e-5` |
|---|---:|---|---:|---|
| native full | 1257/1280 | exact | 1.7642974853515625e-5 | fail |
| joint program-head lesion | 1256/1280 | exact | 1.621246337890625e-5 | fail |
| whole selected-circuit node-zero | 710/1280 | exact | 7.867813110351562e-6 | pass |
| selected-edge zero | 820/1280 | exact | 6.556510925292969e-6 | pass |

The locked policy requires every baseline path to be within `1e-5`. Native
full and joint-head lesion exceed it, so the aggregate preflight fails.
Epsilon was not changed.

The pre-healing audit does not store the two raw candidate logits; it stores
their decision and candidate margin. The preflight therefore compared every
available stored per-input quantity and recorded fresh candidate-logit hashes
for any future, separately authorized same-path investigation.

OWT was not evaluated because the stop rule fired first.

## Limitation and disposition

This terminal result is a reproducibility-boundary result, not evidence that
read-side calibration succeeds or fails. The discrepancy is small enough to
leave every projected decision and mismatch set unchanged, but it is larger
than the preregistered numerical identity tolerance on two paths. The protocol
explicitly forbids relaxing epsilon or continuing after that observation.

All prior artifacts remain unchanged. The registration is commit `1046efc`.
The follow-up made zero optimizer steps and did not run any ladder rung.

## Final Phase Q result

The completed registered continuation produced a formally verified
GPT-2-derived quote-closing circuit over the frozen, hash-pinned 1,280-prompt
domain D. The model of record is the rung-3 program-local W_V/W_O calibrated
checkpoint; every non-program parameter was frozen and hash-verified. It
achieves 1,280/1,280 full and circuit-only decisions with OpenWebText
perplexity 25.5707, below the locked 28.6176 budget.

The project set out to verify extracted circuits and found it had to construct
the object it could verify: for attention, symbolic replacement is the method
rather than a retreat, because naive encoding is intractable (measured),
extracted neural circuits are near-exact but not exact beyond their extraction
data (measured across three protocols), and the distilled artifact's
faithfulness to the deployed model is deductive rather than interpretive.

### Causal result

The pre-healing audit showed that redundancy predated calibration. Jointly
lesioning program heads `7.11` and `9.0` leaves 1,256/1,280 prompts correct in
both the untouched baseline and the calibrated model, while the registered
whole-circuit lesion leaves 710/1,280. Calibration cannot create a bypass:
every trainable contribution is program-local and vanishes under the
corresponding lesion, and all remaining parameters are hash-identical. The
registered final criterion therefore uses leakage identity, joint
program-set necessity, whole-circuit necessity, and formal edge necessity
rather than demanding that the original model's redundant auxiliary head
become individually necessary.

Re-extraction selected the three-edge causal core:

```text
emb → mlp_0 → attn_7_h_11 → logits
```

Head `9.0` remains a frozen program in the full checkpoint but lies outside
the selected core. The circuit's mechanism shape—early-MLP quote features
followed by one copy head—qualitatively converges with the string-closing
mechanism reported by Gao et al., here found independently in a densely
trained model.

### Exact four-property verification

The FP32 encoder matches PyTorch with maximum candidate-logit error
1.11 × 10⁻⁸. A semantics-preserving exact fold certifies every used LeakyReLU
branch, contracts each concrete input to exact rational constants, and leaves
only ground linear-real-arithmetic comparisons. Folded and independent
monolithic evaluators agree by exact rational equality on all eight preserved
anchors.

All four registered properties pass on every prompt in D:

| Property | Result |
|---|---|
| Functional equivalence | 1,280/1,280 |
| Content invariance | 1,280/1,280 |
| Edge necessity | all 3 edges; 640 decision-changing witnesses per edge |
| Continuous robustness | 1,280/1,280 at ε = 0.01 |

The minimum exact certified radius is approximately 0.01514986. The records
attribute 5,911,040 assertions and contain no normalization branches or
bilinear attention terms; *unknown* is structurally unreachable. Full
per-input proof records are under `folded_verification/`.

### Declared boundary

The legacy synthetic sequence `[10, 1, 10]` is a genuine outside-D
counterexample, reproduced exactly by folded and monolithic evaluators. It is
preserved as a constructive boundary exhibit rather than represented as a
claim over D. Bracket type is the separate scale boundary: its only
exactly-generalizing selected circuit retains all 144 attention heads, so it
is reported as a localization result rather than verified.

The checkpoint hash, domain hashes, circuit and program hashes, causal gates,
verification records, and Phase Q commits are pinned in
`artifacts/gpt2-phase-q-agent/evidence_manifest.json`.
