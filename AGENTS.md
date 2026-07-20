Phases A and B are independent and cheap — run them in parallel; everything expensive is gated behind them.

**Phase 0 — Infrastructure (prerequisite for everything downstream)**

1. **Per-head circuit extraction.** Refactor `CircuitGraph` from block-level `attn_i` nodes to `attn_i_h_j`, with per-head masking in `controlled_forward` (hook the per-head outputs before W_O, zero deleted heads). Re-run the existing block-level extractions as a regression check — the union of retained heads should reproduce the block-level circuits. Everything in B and C consumes this.
2. **Restricted attention-program DSL + synthesis harness.** Primitives: token identity, position, relative distance, thresholds, additive weights, normalization by branch-constant sums — no spaCy/NLTK/embeddings. Scoring: IoU for ranking during search, but the acceptance metric is exact projected agreement on the extraction domain (or "close enough to heal," see B3). Reuse the Hayes-style LM-synthesis loop with your restricted prompt.
3. **Program-head module.** A drop-in attention head whose weight matrix comes from the program (function of tokens/positions only); W_Q/W_K deleted, W_V/W_O trainable, program frozen. Plus its SMT encoding: per input, weights are rational constants, so the head contributes only linear constraints — write the encoder now and unit-test against the PyTorch module.

**Phase A — LayerNorm removal (small model first, then scale)**

4. **A1.** Train a small-model variant: sparsemax + LeakyReLU + *standard LN* (your current small model uses BandNorm; this is the removal candidate). Exit: 100% candidate accuracy on both tasks.
5. **A2.** Baroni-style removal fine-tune on A1 (gradual LN attenuation schedule). Exit: accuracy retained without any norm. This also answers the open science question — whether removal survives sparsemax's score-scale sensitivity — cheaply, before any GPT-2 compute.
6. **A3.** Verification payoff measurement: encode the norm-free small model, re-run all four properties, and log categorized assertions plus per-property solve time. Compare quantitative costs only with topology-matched or per-edge/per-norm-instance normalized measurements. Note the qualitative win explicitly: G_T is now affine + argmax, so the robustness property has no branch certificates and no possible *unknown* outcomes. This is a publishable table row on its own.
7. **A4 (expensive; gate on A2 succeeding).** GPT-2 scale: train sparsemax + LeakyReLU + LN, then removal fine-tune. **Preregistered decision gate:** choose norm-free only if its final post-fold OpenWebText eval loss is strictly below the BandNorm-only comparator of 3.3180. If removal wins, the recommended verifiable recipe changes and BandNorm becomes a documented negative result; if it loses or diverges, BandNorm survives as the from-scratch answer and you've measured the trade both ways. Either outcome updates the results table.

**Phase B — Program-head pilot (small model; days, not weeks)**

8. **B1.** Per-head extraction on the 8K model, both tasks, min_agreement = 1.0.
9. **B2.** Synthesize programs for the retained heads against their attention maps on the 128-input domain. Sparsemax's exact zeros should make targets crisper than softmax's dense maps — note whether that holds.
10. **B3.** Freeze programs, heal the rest (small model retrains in minutes — sweep freely). Exit: 100% candidate accuracy on both tasks.
11. **B4.** Migration check on the healed model M′: re-extract; require the program heads are *necessary* (ablating them breaks the projected behavior) and *non-bypassed* (no neural path outside them restores it).
12. **B5.** Verify all four properties on M′'s circuit; log encoding cost vs the original circuit — this is your first "circuit with zero bilinear terms" datapoint. **Gate for Phase C:** B3–B5 all pass.

**Phase C — GPT-2-scale integration (the milestone run)**

13. **C1.** Choose the base model per A4: norm-free sparsemax+LeakyReLU if removal won, else the band-norm-sparsemax checkpoint.
14. **C2.** Per-head ACDC for quote_close and bracket_type, threshold sweep, min_agreement = 1.0, on the same extraction domains as before.
15. **C3.** Synthesize restricted-DSL programs for the circuit heads. If a head resists (syntactic/coreference-type behavior), record it and fall back: replace the fittable heads, keep that head neural — partial replacement still shrinks the NRA core proportionally.
16. **C4.** Freeze + heal on OWT. Stopping criteria: 1.000 projected agreement on the extraction domain *and* a pre-registered perplexity budget (pick the number before the run — e.g. "≤ Hayes's +16% at comparable replacement fraction" — so the result can't be goalpost-shifted).
17. **C5.** Migration check at scale — the step most likely to fail, since real models implement syntax redundantly. If the heal routes around the programs: ablation-aware healing (stochastically ablate non-circuit paths during fine-tuning so the loss can't lean on backups). That counter-move is itself a contribution — it's the "architectures/training to reduce redundancy" cell of your Slide 4 table made concrete, so budget for it rather than treating it as failure.
18. **C6.** Verification of the healed circuit: SMT-encoder-vs-PyTorch sanity test first (your existing `test_smt_encoder.py` pattern), then the four properties at max_length 3, scaling length until the solver taps out. Even partial success here — *any* nontrivial property proven on a GPT-2-scale-derived circuit — is the paper's named next milestone, achieved.
19. **C7.** Unified cost table: every component's verifiability priced in the same currency — sparsemax +0.063, BandNorm +0.196 or removal +X (from A4), program heads +Y perplexity (from C4). That table is the new headline figure.

Suggested execution order: **0.1–0.3 → A1–A3 ∥ B1–B5 → A4 → C → D.** Kill criteria worth writing down now: if A2 fails (removal incompatible with sparsemax), Phase C proceeds on BandNorm and A4 is skipped; if B4 shows migration even at toy scale, solve ablation-aware healing there before spending any GPT-2 compute; if C3 can't fit the circuit heads even partially, the salvage output is the counterexample analysis of *why* those heads resist token-level description — which is an interpretability result in its own right.

## Execution Status — 2026-07-20

- **Phase 0.1–0.3:** complete.
- **A1–A3:** complete. The norm-free model retains 100% candidate accuracy, all four properties verify on both tasks, and the final robustness map is affine plus argmax.
- **B1–B5:** complete. Per-head extraction, restricted-program synthesis, ablation-aware healing, migration checks, and four-property verification all pass on both tasks.
- **Remediation Task 1:** complete. Assertion-source attribution, per-property solve times, normalized cost columns, and the available five-edge quote-close comparison are recorded; bracket type has no equal-edge pair and is explicitly skipped.
- **Remediation Task 2:** complete. The epsilon values match at 0.01, both matched BandNorm tasks classify as branch-adjacent, and the norm-free contrast verifies at every swept radius.
- **Remediation Task 3:** complete in one chase round. The pre-intervention drift artifact is preserved; `attn_1_h_1` was programmed; the final quote circuit has only the intended program heads, passes migration, and verifies all four properties with zero active neural-attention bilinear terms. The plain-heal failure artifact remains unchanged.
- **A4:** complete. The sparsemax + LeakyReLU + standard-LayerNorm source stopped at step 200,000 with OpenWebText eval loss 3.1968865. Sequential attenuation reached the fixed-std endpoint for all 25 norms, and the folded norm-free model achieved post-fold loss 3.2056017 (perplexity 24.6703), only +0.0087152 loss versus its source and 0.1123983 below the locked 3.3180 gate. The fold passed in FP32 with 1.000 top-1 agreement and a 3.67e-5 eval-loss delta.
- **C1:** complete. The preregistered selector chose `artifacts/gpt2-norm-free` as the Phase C base; BandNorm remains the documented from-scratch negative result rather than the recommended recipe.
- **Open:** Phase C steps C2–C7 remain GPU-scale work.
