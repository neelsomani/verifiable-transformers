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
- **Phase C held-out-generalization track:** complete and stopped at v4. The full
  norm-free model was exact on both fresh 512-row task gates; the selected
  bracket circuit was exact, while the selected quote circuit missed one
  double-quote example (511/512). No symbolic programs were installed and no
  healing or verification was performed on this track. The terminal evidence
  is preserved; there will be no protocol v5.
- **Open:** the bounded-domain quote flagship below is the active Phase C
  continuation. Bracket surgery and the other follow-ups remain sequenced
  behind it.

## Execution Agenda — 2026-07-22

The bounded-domain quote flagship is the critical path. Its declared domain
`D` is the ordered union of the frozen v4 development and gate manifests:
1,280 unique, hashed prompts per task. There is no held-out split in this
continuation. The locked gates are OpenWebText perplexity at most 28.6176,
exact agreement with `P(x)` on every row of `D`, and a full unsampled
migration/lesion sweep. This is a bounded-domain claim, not an out-of-sample
generalization claim. Everything else below is sequenced so it does not alter
that experiment.

### Phase Q — Quote flagship (in flight)

1. Register the bounded-domain continuation before the run: `D` is the 1,280
   frozen hashes per task, there is no gate split, and the gates are perplexity
   at most 28.6176, exact-on-`D`, and the full unsampled migration/lesion sweep.
2. Re-extract `quote_close` on all 1,280 rows using a threshold sweep with
   `min_agreement = 1.0`, and record the retained-head count.
3. Synthesize programs for exactly those retained heads on all 1,280 rows,
   with the LM proposer enabled and a larger candidate budget. If a head
   resists, first try the small-scale healing-absorbs-slack precedent: allow
   high-but-imperfect individual fidelity while requiring the joint pre-heal
   composition check to referee acceptance. Only if that fails may the
   scan-primitive DSL extension activate, scoped to that head.
4. Run an encoding smoke test on the new circuit before healing: record the
   assertion count and solve time for one input so verification cost is known
   before the expensive run.
5. Install quote programs only; do not install bracket's 79 candidates. Run
   core-aware healing, the joint pre-heal composition check, the locked gates,
   the full lesion sweep, the migration check, and a re-extraction drift check.
   Use the chase protocol if needed, with an iteration guard of two.
6. Export in FP32, run the encoder-vs-PyTorch sanity check, and verify all four
   properties over the declared 1,280-row bounded domain.

**Kill criterion:** if healing cannot achieve exact-on-`D` within the
perplexity budget after the core-aware objective plus one chase round, stop and
report. Do not introduce another healing objective at GPT-2 scale.

### Phase R — Terminal report and documentation

This phase may proceed during Phase Q's training waits, but pending claims must
remain visibly pending until Q lands.

7. Write the GPT-2 results section covering: the quote verified
   zero-bilinear circuit (pending Q); the sparsity-versus-exactness frontier of
   17 edges/99.92% versus 340 edges, 144 heads, and exactness; the v2-to-v4
   gate trajectory; the plain-heal lesion result; and bracket as a boundary
   measurement rather than unfinished surgery.
8. Complete the documentation cascade: update `SCALABILITY.md` with the v4
   outcome and bounded-domain continuation; add the GPT-2 section to
   `VERIFIED_DISTILLATION.md`; update the README introduction only after Q
   lands; and, if the arXiv paper is revised, update both Limitations and Future
   Work.
9. Maintain commit discipline throughout: preserve the v4 stop bundle as the
   terminal record, commit Q evidence as it lands, and include evidence
   manifests with checkpoint hashes. Model weights and checkpoint state remain
   outside Git under the existing artifact policy.

### Phase S — Registered localization follow-up

10. Run the cheap probe first: apply the extraction plus untouched
    generalization-gate protocol to Gao et al.'s released weight-sparse
    checkpoints without training. Register the exact model, the domain
    generator adapted to its tokenizer, and the gate protocol before running.
    The question is whether weight-sparse training yields a small bracket
    circuit that passes an untouched gate.
11. Use the probe as a decision gate:
    - **Pass:** justify fused-recipe pretraining (weight sparsity × sparsemax ×
      LeakyReLU × norm removal) as a separate project with new baselines, a
      re-derived perplexity budget, and a registered domain-reuse decision.
    - **Fail:** report bracket diffuseness as plausibly task-intrinsic even
      under sparsity pressure, and do not pretrain the fused recipe.

### Phase T — Publication assembly

12. After Q and R, choose the artifact shape: either revise the arXiv paper
    with the removal, program-mediated verification, GPT-2 flagship, and
    frontier results, or publish the distillation/localization work as a
    second paper citing the original. The working recommendation is a second
    paper, "from verifiable architectures to verified distillation," with the
    Phase S probe determining whether it ends on a positive localization
    result or a measured boundary.
13. Update the talk. The existing Future Directions slide now contains mostly
    completed work, so the original video needs either a revised ending or a
    sequel.

### Explicitly parked

- **Bracket surgery on the 340-edge circuit:** superseded by Phase S. If
  localization is trainable, operate on the sparse model's smaller bracket
  circuit instead.
- **Mean-ablation semantics:** unregistered diagnostic only, unless Phase S
  encounters the same extraction limit.
- **Softmax generalization of the program-head method:** future work, not part
  of the current experiment.
- **Counterexample-guided DSL expansion:** activates only if Phase Q step 3's
  registered fallbacks fail.

The critical path is: declare the bounded domain, re-extract, synthesize, heal,
and verify. Phases R through T are reporting and registered follow-up work.
