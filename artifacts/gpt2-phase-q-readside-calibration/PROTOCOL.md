# Phase Q constrained causal-integration follow-up

Registered 2026-07-27 from `daf939e616d031cf3429285aff8d5664083d235f`
on `codex/phase-q-agent`. This is a distinct follow-up to the terminal Phase Q
result. It does not alter that result or any prior artifact.

## Frozen inputs and execution

The declared bounded domain `D` is the ordered 1,280-row `quote_close` domain
in `artifacts/gpt2-behavior-domain-bounded-v1/domain.json`. Its manifest,
prompt-set, and ordered-ID hashes are recorded in `registration.json`; there
is no development/gate split and no out-of-domain claim.

State B is reconstructed by loading the norm-free source checkpoint and
installing exactly the two frozen programs `attn_7_h_11` and `attn_9_h_0`,
without constructing an optimizer or taking a step. The source weights,
program file, and selected circuit are pinned by SHA-256 in
`registration.json`.

Every baseline and post-calibration comparison uses one pinned FP32 path:
CUDA FP32, TF32 disabled, autocast disabled, deterministic algorithms enabled,
cuDNN deterministic enabled and benchmarking disabled, seed 42, fixed
domain order, batch size 8, and the same controlled-forward implementation.
The preflight must first reproduce the stored audit decisions, mismatch sets,
candidate logits, and candidate margins within `1e-5`. Failure is terminal;
epsilon will not be relaxed.

## Causal intervention

For rungs 1 and 2, both programs and every original GPT-2 parameter are frozen.
Calibration acts only on each programmed head contribution as read by retained
downstream circuit nodes. It is not applied at the head's residual write, so
unselected readers observe the uncalibrated contribution. Cache/decomposition
may be used but must be numerically checked against the direct path. When the
selected edges are zeroed, the calibration delta vanishes by construction.

Rung 3 is the registered weaker fallback: program-local `W_V/W_O` may change,
which changes the residual write. Selected-edge zero is therefore
observational only for rung 3; joint-head and whole-node lesion identity remain
mandatory.

## Frozen escalation ladder

1. **Scalar read-side gains.** Two FP32 gains, one per programmed-head/read
   edge, initialized to 1.0. Adam, learning rate `0.01`, no weight decay,
   full `D` batches in frozen order, at most 4,000 optimizer steps and 4,000
   objective evaluations. Evaluate gates every 100 steps. Stop at the first
   checkpoint passing exact native-full and circuit-only accuracy, identity,
   and OWT gates. If none passes, advance.
2. **Diagonal per-channel read-side calibration.** Two 768-channel FP32
   diagonal vectors initialized from the terminal scalar gains (broadcast).
   AdamW, learning rate `0.002`, weight decay `1e-4`, fixed minibatches of 64,
   gradient norm clip 1.0, at most 6,000 steps and 6,000 objective evaluations.
   Evaluate `D` every 200 steps and OWT at each exact-on-`D` checkpoint. Stop
   at the first full gate pass; otherwise advance.
3. **Program-local `W_V/W_O` fallback.** Only `W_V/W_O` belonging to the two
   frozen program heads are trainable; programs and all other parameters stay
   frozen. AdamW, learning rate `5e-5`, weight decay `0.01`, fixed minibatches
   of 32, gradient norm clip 1.0, at most 4,000 steps and 4,000 objective
   evaluations. Evaluate `D` every 200 steps and OWT at each exact-on-`D`
   checkpoint. Stop at the first full gate pass. If none passes, terminate the
   follow-up; no new rung, objective, threshold, or epsilon may be introduced.

All rungs optimize cross-entropy against `P(x)` on all rows of `D`, plus a
fixed OWT preservation term of weight `0.10` using the preregistered OWT
evaluation corpus/order already used by Phase Q. No MLP, embedding, LM head,
shared residual write (rungs 1–2), or outside-circuit parameter is trainable.

## Locked gates

- native full agreement with `P(x)`: 1,280/1,280;
- circuit-only agreement: 1,280/1,280;
- OWT perplexity at most `28.617593822841776`;
- joint program-head lesion reproduces the state-B per-input decisions and
  mismatch set, with candidate logits and margins within `1e-5`;
- whole-selected-circuit node-zero reproduces state B (710/1,280 and exact
  mismatch set), with logits and margins within `1e-5`;
- for rungs 1–2, selected-edge zero reproduces state B (820/1,280 and exact
  mismatch set), with logits and margins within `1e-5`;
- every non-program/non-calibration parameter is cryptographically
  hash-identical;
- the full unsampled migration/lesion sweep and re-extraction drift check pass
  under the registered causal criterion.

For rungs 1–2, causal passage means exact task gates plus identity under all
three locked lesions, selected-edge delta exactly zero by construction, no
parameter drift, and no newly sufficient path in the unsampled sweep. For rung
3, the same criterion applies except selected-edge identity is recorded as
observational and is not causal evidence; joint-head and whole-node identity
remain mandatory.

On passage, export FP32 plus exact decimal/rational calibration constants,
rerun all locked gates, re-extract, run encoder-versus-PyTorch sanity, and run
all four SMT properties over `D`. Scalar/diagonal constants enter the encoder
only as linear constants.

## Registered baseline geometry

The stored state-B native full forward has 23 negative-margin failures. The
joint-program lesion has 24. All 23 native failures are a subset of the lesion
failures; the sole lesion-only ID is
`v4:quote_close:single:366310ced48097af`. Exact ID-set hashes and margin
summaries are in `registration.json`. This is characterization only and gives
no permission for lesion drift.
