# Protocol amendment: constrained-calibration causal criterion

Registered 2026-07-27 from commit
`ee3577463ed4691e42f47d12fc2ad525609c7053`, before any new model
evaluation, extraction, pruning experiment, or SMT run.

This amendment preserves every earlier registration, terminal report,
checkpoint, and evidence file unchanged. It authorizes no training, tuning, or
new objective before verification. It corrects the migration proxy used for
the already constrained program-local calibration design.

## Causal criterion

Individual program-head necessity is replaced by the conjunction of:

1. **Leakage identity.** All non-calibration/original weights must retain their
   registered hashes, and the registered lesion logits, margins, and decisions
   must be invariant. This battery has already passed bitwise.
2. **Joint installed-program necessity in the otherwise-full model.** Removing
   the complete installed program-head set must reduce correctness below
   `1280/1280`. The registered result is `1256/1280`.
3. **Whole-selected-circuit necessity.** Zeroing every selected circuit node
   must reduce correctness below `1280/1280`. The registered result is
   `710/1280`.
4. **Circuit-internal edge necessity.** The registered SMT edge-necessity
   property must evaluate every edge in the chosen fixed circuit. Exact failed
   edges and solver witnesses must be preserved.

Exact full-model and circuit-only correctness remain `1280/1280` on the
declared bounded domain `D`. The frozen OpenWebText perplexity ceiling remains
`28.617593822841776`.

Individual necessity was a proxy for optimizer-built migration during
unconstrained healing. Here, original-parameter hashes plus bitwise lesion
identity carry that inferential burden directly: calibration could not have
created an unregistered bypass while those parameters and causal responses
remained identical. The untouched/pre-heal lesion matrix already demonstrated
strong native redundancy. Consequently, individual-head necessity primarily
tests a property of the original redundant mechanism, not leakage caused by
the constrained calibration.

The registered mechanism interpretation is:

- full model with `attn_7_h_11` zeroed: `1280/1280`;
- full model with `attn_9_h_0` zeroed: `1280/1280`;
- full model with both zeroed: `1256/1280`;
- circuit with `attn_7_h_11` zeroed: `640/1280`;
- circuit with `attn_9_h_0` zeroed: `1280/1280`.

Thus `attn_7_h_11` is circuit-internally load-bearing, while `attn_9_h_0` is
auxiliary/redundant. This is an interpretability result, not a calibration
defect.

## Deterministic lean-candidate selection

Before testing either disposition, construct one lean candidate from the
rung-3 checkpoint:

- exclude `attn_9_h_0` globally and hard-zero its pre-`W_O` output on every
  path;
- retain only the frozen restricted-DSL program at `attn_7_h_11`;
- retain the already trained, permitted `attn_7_h_11` program-local `W_V`,
  `b_V`, and bias-free `W_O` slices from the rung-3 checkpoint;
- do not restore `attn_9_h_0` as a neural Q/K head;
- perform no training, tuning, threshold search, or parameter adjustment.

Evaluate, in fixed domain order: exact full behavior on `D`, exact
circuit-only behavior on `D`, full-model `attn_7_h_11` lesion necessity,
whole-circuit lesion necessity, original/non-permitted parameter integrity,
program and pruning-mask integrity, and full frozen OpenWebText perplexity.

The lean candidate is selected if and only if all of these gates pass:

- full correctness is `1280/1280`;
- circuit-only correctness is `1280/1280`;
- removing `attn_7_h_11` reduces correctness below `1280/1280`;
- zeroing the whole selected circuit reduces correctness below `1280/1280`;
- OpenWebText perplexity is at most `28.617593822841776`;
- every integrity check passes.

If it passes, it becomes the fixed flagship and `attn_9_h_0` is the removed
auxiliary head. If any gate fails, the fixed flagship is the existing
two-program rung-3 calibrated model under the amended set-level criterion, and
`attn_9_h_0` is reported as auxiliary.

## Fixed-model extraction and causal replay

On the selected fixed model, rerun per-head extraction over all 1,280 ordered
rows of `D`, requiring projected agreement `1.0`. Any globally removed head is
permanently forbidden during extraction and all later stages; no pruned path
may be re-enabled. Record the selected edge set and topology, row margins,
retained program heads, domain hashes, and extraction settings.

Then replay the complete amended migration/lesion battery on that exact fixed
model and topology: leakage identity, joint installed-program necessity,
whole-selected-circuit necessity, all applicable head lesions as
interpretability measurements, exact full and circuit-only behavior, and the
OpenWebText budget.

## Export and formal verification

Export FP32 values and their exact rational representations for all selected
constants, including calibrated `W_V`, `b_V`, bias-free `W_O` slices and any
hard-pruning mask. Programs remain exact restricted-DSL programs. The SMT
topology must exactly match the selected/pruned topology and must not encode a
neural fallback for a removed program head.

Run encoder-versus-PyTorch sanity before any property claim. Then run the four
registered properties, including circuit-internal edge necessity, first at
`max_length = 3` and then through the already registered scaling sequence until
the solver timeout. Preserve property status, witnesses or counterexamples,
assertion categories, and solve times at every attempted length.

### Edge-necessity result policy

No post-result pruning is authorized in this continuation. Edges are evaluated
in the deterministic canonical order `(child node, parent node)` after
lexicographic node-name sorting. For every failed edge-necessity query, preserve
the solver status, exact edge, witness assignment/input, projected decision and
margin data, and encoder/topology hashes. Failed edges remain in the exported
circuit and are reported honestly as auxiliary/redundant; they are not silently
removed and the solver is not rerun on an adapted topology. This fixed
report-only policy is the registered alternative to a solver-witness
minimality/pruning loop.

All selection, behavioral, and property claims are bounded to `D` or the
registered formal domains. No held-out-generalization claim is authorized.
