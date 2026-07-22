# Scalability

In this section, we show that the same verifiable operators in the small Transformer model above can be used to stably train a GPT-2 scale model.

## Baseline GPT-2 Description

This repository includes a reproducible baseline using `gpt2` (124M) with OpenWebText training and WikiText-103 evaluation.

We evaluate both a pretrained GPT-2 reference run and a locally reproduced reference GPT-2-small model. We define the baseline as our own locally reproduced reference GPT-2-small run, using the exact same tokenizer, preprocessing, context length, optimization recipe, token budget, and evaluation code as all later architecture variants.

We first train a local reference GPT-2-small baseline under the same pipeline used for all experiments. Alternative attention and normalization variants are judged relative to this local baseline.

The local reference baseline should be stable and reproducible under the configured training recipe, and all subsequent architecture changes should be compared against this exact run setup.

Primary comparison metrics:

* OpenWebText validation loss under identical preprocessing and evaluation steps.
* WikiText-103 perplexity under identical tokenizer and evaluation code.

Note: exact absolute numbers depend on hardware scale, effective batch size, training duration, and data preprocessing details. Relative comparisons are the primary acceptance signal.

## Baseline Training

Run this first to record pretrained GPT-2 reference numbers with the same evaluation protocol:

```bash
python scripts/gpt2/eval_pretrained.py \
  --model_name gpt2 \
  --block_size 1024 \
  --stride 1024 \
  --output_json artifacts/pretrained-gpt2-reference-metrics.json
```

Current local reference snapshot (from `artifacts/pretrained-gpt2-reference-metrics.json`):

- OpenWebText validation (`eval_percent=1.0`, `max_samples=10000`): loss `3.1187`, perplexity `22.6160`
- WikiText-103 validation (`max_samples=null`, full split): loss `3.4353`, perplexity `31.0403`

These values are a local anchor for relative comparisons. Minor drift is expected across environments and dependency versions.

Then train the local baseline (8 GPUs) until OWT hits the threshold:

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/gpt2/train.py \
  --config configs/gpt2_baseline.json \
  --output_dir artifacts/gpt2-baseline
```

This run uses early stopping on OpenWebText validation loss by default (`early_stop_eval_loss = 3.2` in the config).

Default dev behavior:

- Train on OpenWebText.
- Evaluate on OpenWebText validation during training.
- Early stop on OpenWebText validation loss.

Optional WikiText dev modes (opt-in only):

- `--evaluate_wikitext_at_end`: run WikiText-103 once after training.
- `--use_wikitext_as_dev`: run periodic WikiText eval during training (coarse cadence).
- `--target_wikitext_ppl <value>`: enable WikiText-target early stopping (e.g. `--target_wikitext_ppl 43`).
- `--wikitext_eval_every_n_evals <N>`: evaluate WikiText every N Trainer eval events.
- `--reset_optimizer_on_resume`: resume from checkpoint weights while resetting optimizer/scheduler/scaler/rng state.

When `--use_wikitext_as_dev` or `--target_wikitext_ppl` is enabled, OpenWebText early stopping is disabled by default unless you explicitly pass `--early_stop_eval_loss`.

Performance defaults in the baseline config:

- Preprocessing uses multiprocessing (`preprocessing_num_proc`) and caches processed datasets to disk.
- Subsequent runs reuse cached datasets instead of retokenizing/regrouping.
- `eval_steps` and `save_steps` are intentionally set high to reduce overhead during long runs.
- `torch_compile` is enabled by default.
- `bf16=true`, `fp16=false`, and gradient checkpointing is off unless memory constraints require it.

Evaluate WikiText-103 perplexity:

```bash
python scripts/gpt2/eval_wikitext.py \
  --model_path artifacts/gpt2-baseline \
  --split validation \
  --block_size 1024 \
  --stride 1024 \
  --output_json artifacts/gpt2-baseline-wikitext103-validation.json
```

Plot training curves:

```bash
python scripts/utils/plot_training_curves.py \
  --run_dir artifacts/gpt2-baseline \
  --output_png artifacts/gpt2-baseline/training_curves.png
```

This reads `trainer_state.json` from the run directory and plots train/eval loss versus training step.

## Baseline Results

Current local baseline outcome (this repository run):

* OpenWebText validation loss: 3.1340 (at step 260000)
* OpenWebText validation perplexity: 22.9650 (at step 260000)
* Relative to current pretrained reference (OWT loss 3.1187, perplexity 22.6160): +0.0153 loss and +0.3490 perplexity
* WikiText-103 perplexity: 52.9820 (at step 260000)
* Relative to current pretrained reference (WikiText-103 perplexity 31.0403): +21.9417 perplexity

Interpretation:

* WikiText-103 perplexity is substantially worse than the pretrained GPT-2 reference in this run.
* This baseline setup is optimized for the OpenWebText training pipeline and uses OpenWebText-based stopping criteria by default.
* We use this run as a local baseline anchor for relative architecture comparisons, not as a claim of best absolute WikiText-103 performance.

This run met the configured early-stop target (`eval_loss <= 3.2`).

## Verifiable Replacement Evaluations

We use the same training script and process, with architecture variants controlled via config (`norm_variant`, `attn_variant`) to keep training/eval pipeline constant. We first establish the pretrained GPT-2 reference and local baseline above, then compare these variants against that local baseline.

### LayerNorm Replacement Only

**DyT (Failed)**:

Config: `configs/norm_dyt.json`

DyTNorm replaces LayerNorm with an affine → clamp → affine transformation. It is fully SMT-encodable, but it is not a true normalization layer: it provides neither data-dependent centering nor scale control. In practice, training improved briefly, then gradient norms rose and the run diverged, consistent with uncontrolled residual accumulation.

**Verifiable PWL Norm v1 (Failed)**

Config: `configs/norm_pwl_v1.json`

PiecewiseLinearNorm v1 restored two important normalization ingredients — mean-centering and adaptive rescaling via mean absolute deviation (MAD) — while remaining SMT-encodable. However, its gain function used coarse discontinuous buckets `[4.0, 2.0, 1.0, 0.5, 0.25]`, which caused abrupt scaling changes and could over-amplify low-MAD tokens. Empirically, it trained at first but later showed rising gradient norms and eventual divergence.

**Verifiable PWL Norm v2: Soft Leaky Clamp (Failed)**

Config: `configs/norm_pwl_v2.json`

v2 simplified the norm to mean subtraction, leaky piecewise-linear clamp, fixed scaling by 0.5, and learned bias. This results in better gradient flow relative to a hard clamp, but the clamp remained unbounded: outside the clamp region, activations still grew linearly. The result was better early optimization but eventual explosion once residual accumulation returned. The key lesson was that gradient flow alone is not enough; the norm must also enforce explicit magnitude control.

**Verifiable PWL Norm v3: Bounded PWL Clamp (Stable but High Error)**

Config: `configs/norm_pwl_v3.json`

The progression v1 → v2 → v3 isolates the core requirement: v1 had insufficient scale control, hard clamp bounded activations but killed gradients, and v2 preserved gradients but left activations unbounded. The resulting target is bounded activations and nonzero gradients.

v3 implements this with a bounded piecewise-linear saturator:
- `x < -3.0` → constant `-2.3`
- `-3.0 ≤ x < -2.0` → linear slope `0.3`
- `-2.0 ≤ x ≤ 2.0` → identity
- `2.0 < x ≤ 3.0` → linear slope `0.3`
- `x > 3.0` → constant `2.3`

The norm itself is: mean subtraction, bounded PWL clamp, fixed scale `0.5`, and learned bias. Interpretation:
- v3 is the first **stable and trainable** norm-only replacement
- it solves the main late-stage instability problem
- but it is still **substantially worse than the local GPT-2 baseline** on both OWT (4.1350 @ ~300k step) and WikiText (189.3 @ 400k), so the main result here is stability rather than parity with standard LayerNorm

**Signed L1 Band Norm: Projection-Based Normalization (Best Replacement; Superseded by Removal)**

v3's failure mode was elementwise saturation: every coordinate was independently clamped, which destroyed dynamic range and prevented the model from learning useful representations. The v1-v3 progression tried to approximate LayerNorm by multiplying centered inputs by a scale factor (whether data-dependent buckets, soft clamps, or bounded PWL). This approach fundamentally conflicts with SMT-friendliness because proper normalization requires dividing by a data-dependent variance estimate.

Signed L1 BandNorm takes a completely different approach: **projection-based normalization** instead of elementwise clamping. This is the same kind of operation that made sparsemax workable: sparsemax is a projection onto the probability simplex, and it trains despite support changes. Projection onto an L1 ball is a known exact threshold/sort operation, naturally piecewise-linear and SMT-encodable.

The operator:
1. Center: `c = x - mean(x)`
2. Split into positive and negative masses: `p = max(c, 0)`, `n = max(-c, 0)`
3. Control each mass separately to preserve zero-mean structure:
   - If mass too small: additive lift over active coordinates (piecewise affine)
   - If mass too large: project onto L1 ball using top-k thresholding (like sparsemax)
4. Recombine: `z = p' - n'`
5. Apply learned affine: `output = gamma * z + beta`

This differs fundamentally from v3:
- v3: elementwise bounded clamp + residual scaling (0.25 branch scale + 0.98 state contraction)
- BandNorm: projection normalization + **standard GPT-2 residual path** (no modification)

v3 made the model bounded but destroyed dynamic range. BandNorm enforces a scale invariant without clamping every coordinate independently, much closer to what normalization actually needs to do.

Config: `configs/norm_band_norm.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/gpt2/train.py \
  --config configs/norm_band_norm.json \
  --output_dir artifacts/norm-band-norm-only \
  --disable_auto_resume \
  --use_wikitext_as_dev \
  --target_wikitext_ppl 53 \
  --wikitext_eval_every_n_evals 1
```

Results:

**OpenWebText (validation):**

* Best eval loss: **3.3180 @ 220k steps** (tapers off around 180K)
* Relative to baseline (3.1340): **+0.1840** loss

**WikiText-103 (validation):**

* Perplexity: **61.89 @ 220k** (tapers off around 180-200K)
* Relative to baseline (52.98): **+8.91** perplexity

Interpretation:

* Signed L1 BandNorm is the strongest verifiable normalization candidate so far.
* It is stable and trainable under the GPT-2-small OpenWebText recipe.
* It substantially closes the gap introduced by earlier clamp-based verifiable normalizers.
* It remains worse than standard LayerNorm and sparsemax-only, so normalization is still the main performance bottleneck.
* Projection-based normalization appears much more viable than elementwise clamp-based normalization.
* Superseded: the A4 LayerNorm-removal result below beats BandNorm by 0.1124 loss, so Phase C uses the norm-free model; BandNorm is retained as the measured from-scratch replacement result.

### Attention Replacement Only (Sparsemax)

Sparsemax replaces softmax attention weighting with a piecewise-linear alternative (alpha-entmax family, alpha=2) that can produce exact zeros in the attention distribution. This is fully SMT-encodable.

Implementation:
- Uses standard LayerNorm (not verifiable norm variants)
- Replaces softmax with sparsemax via monkey-patching GPT2Attention.forward
- Runs sparsemax computation in fp32 for numerical stability (upcasts from bf16)
- Includes fail-fast verification on startup to ensure patch is active
- Compatible with transformers 4.49.0

Verification (run before full training to make sure patch is working):
```bash
python scripts/utils/verify_sparsemax.py
```

Config: `configs/attn_sparsemax.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/gpt2/train.py \
  --config configs/attn_sparsemax.json \
  --output_dir artifacts/attn-sparsemax-only \
  --disable_auto_resume \
  --use_wikitext_as_dev \
  --target_wikitext_ppl 53 \
  --wikitext_eval_every_n_evals 1
```

Results:

**OpenWebText (validation):**

* Best eval loss: **3.1973 @ 180k steps**
* Relative to baseline (3.1340): **+0.0633** loss

**WikiText-103 (validation):**

* Perplexity: **55.7227 @ 160k**
* Relative to baseline (52.9820): **+2.7407** perplexity

Interpretation:

* Sparsemax is **not equal to baseline** but achieves strong performance
* **Very close on OpenWebText** (+2.0% relative loss increase)
* **Modestly worse on WikiText-103** (+5.2% relative perplexity increase)
* **Dramatically better than verifiable norm replacements** (Step 2a variants)
* This demonstrates that a fully SMT-encodable attention mechanism can achieve near-baseline performance while maintaining exact zeros in attention distributions

### Sparsemax + LeakyReLU + Standard LayerNorm (A4 Run 1)

This isolation run is the source checkpoint for GPT-2 LayerNorm removal and a
standalone measurement of adding LeakyReLU to sparsemax while retaining standard
LayerNorm. It stopped at step 200,000 when the preregistered OpenWebText
validation criterion (`eval_loss <= 3.2`) was first met.

* OpenWebText validation loss: **3.1968865** (perplexity **24.4563**)
* Delta from the local baseline: **+0.0628865** loss
* WikiText-103 validation perplexity: **57.1855**
* Effective global batch: **256** sequences of 1,024 tokens
* Nominal token budget at stopping: **52,428,800,000** tokens
* Status: **removal input; A4 run 1 of 2**

The exact run configuration, summaries, and training curve are recorded under
`artifacts/gpt2-sparsemax-leaky-layernorm/`.

### LayerNorm Removal (A4 Run 2)

Starting from the preceding standard-LayerNorm checkpoint, we fine-tuned for
5,000 optimizer steps while sequentially replacing each of GPT-2-small's 25
normalization instances with a fixed-standard-deviation affine map. The last
transition completed at step 3,400, leaving 1,600 optimizer steps at the fully
attenuated endpoint. With an effective global batch of 256 sequences and 1,024
tokens per sequence, the removal run processed a nominal 1.311 billion tokens.

Results after folding every affine map into its consumer were:

* Pre-fold OpenWebText validation loss: **3.2055650**
* Post-fold OpenWebText validation loss: **3.2056017** (perplexity **24.6703**)
* Incremental removal cost versus the LayerNorm source: **+0.0087152** loss,
  **+0.2141** perplexity (**+0.8753%**)
* Locked BandNorm-only gate: **3.3180**
* Gate margin: **0.1123983** loss in favor of norm-free

The initially reported large absolute logit differences were a BF16 comparison
artifact: folding reorders operations, so the two mathematically equivalent
graphs round differently under BF16 autocast. The recovery pass first confirmed
that every saved attenuation value was 1.0, every calibration flag was false,
and every frozen standard deviation was finite and positive. It then performed
the fold in FP64 and compared both graphs in FP32. Maximum absolute logit error
was **6.58e-5**, relative L2 error was **9.09e-7**, top-1 agreement was **1.000**,
and the pre/post-fold validation-loss delta was **3.67e-5**. All were inside the
declared fail-closed recovery thresholds.

The A4 decision is therefore **norm-free**: Phase C uses the norm-free
sparsemax-and-LeakyReLU model, while BandNorm remains a measured from-scratch negative
result. In the unified cost table, the `layernorm_removal` delta is the
incremental cost relative to its LayerNorm source; its total loss delta from the
local 3.1340 baseline is +0.0716017. On the full 251,048-token WikiText-103
validation stream, the folded model has loss **3.9610** and perplexity
**52.5124**: an improvement of **4.6731** perplexity versus its LayerNorm source
and **0.4696** versus the local baseline. Folding the final norm also unties the
output projection from the token embedding, so the saved model has 163,049,041
parameters; this storage cost is retained as part of the measured recipe.

The exact schedule, endpoint state, fold validation, final decision, effective
configuration, and checkpoint training curve are recorded under
`artifacts/gpt2-norm-free/`. The materialized C1 choice is
`artifacts/gpt2-phase-c-base.json`; `evidence_manifest.json` links these records
to the checksummed, externally stored folded-model archive.

### Combined Verifiable Replacements (Norm + Attention + LeakyReLU)

This combines the two verifiable components from steps 2a and 2b, in addition to replacing the GELU activation function.

This represents the **end-to-end verifiable Transformer**: normalization, attention, and activations are fully SMT-encodable.

Config: `configs/band_norm_sparsemax.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/gpt2/train.py \
  --config configs/band_norm_sparsemax.json \
  --output_dir artifacts/band-norm-sparsemax \
  --disable_auto_resume \
  --use_wikitext_as_dev \
  --target_wikitext_ppl 53 \
  --wikitext_eval_every_n_evals 1
```

Results:

**OpenWebText (validation):**

* Best eval loss: **3.3300 @ 240k steps**
* Relative to baseline (3.1340): **+0.1960** loss
* Relative to Signed L1 BandNorm-only (3.3180): **+0.0120** loss

**WikiText-103 (validation):**

* Perplexity: **62.11 @ 240k**
* Relative to baseline (52.98): **+9.12** perplexity
* Relative to Signed L1 BandNorm-only (61.89): **+0.22** perplexity

Interpretation:

* The full SMT-encodable stack (BandNorm + sparsemax + LeakyReLU) trains stably
* Combining sparsemax attention and LeakyReLU with Signed L1 BandNorm introduces only a small degradation relative to Signed L1 BandNorm alone (+0.36% OWT loss, +0.35% WikiText perplexity)
* The main remaining performance gap is due to normalization, not attention or activation
* This represents a viable end-to-end verifiable Transformer architecture

## Small-Model Verification Remediation

The original Table 2 runs and the matched BandNorm retrain both use
$\epsilon_0=0.01$. The matched retrain is a different checkpoint under the same
intended architecture and task definition, so its branch-unstable result does
not contradict the earlier certified checkpoint. On both tasks, the matched
BandNorm circuit is already branch-adjacent at $\epsilon_0/10=0.001$ and remains
branch-unstable at $\epsilon_0/4$, $\epsilon_0/2$, and $\epsilon_0$; this shows
that BandNorm branch-certifiability is seed-dependent. The norm-free circuits
are certified at all four radii because the final decision is affine plus
argmax and has no normalization branch certificate. Full results are recorded
in `artifacts/robustness_eps_sweep.json`.

Verification-cost totals are not compared across circuits with different
topologies. Quantitative comparisons use assertions per edge, solve seconds per
edge, and norm-attributable assertions per norm instance; raw totals are used
only for the five-edge quote-close pair selected by the equal-edge topology
check. Bracket type is skipped for that matched comparison because the exact
sweeps contain no shared edge count.

The quote-close mechanism drift was frozen in
`artifacts/mechanism_drift.json` before intervention. One bounded chase round
then replaced `attn_1_h_1` with a restricted position program and re-healed the
model with combined and individual ablation pressure on both the full graph and
the preregistered circuit. Re-extraction retains `attn_0_h_0` and
`attn_1_h_1` as individually necessary program heads, migration passes, and
all four properties verify with zero active neural-attention bilinear terms.
The earlier plain-heal migration failure remains unchanged as the motivating
negative result.

## Circuit Extraction from GPT-2 Scale Verifiable Model

Once the verifiable model is trained, we extract pruned circuits responsible for specific behaviors and formally verify their properties using SMT solvers.

### Phase C behavior-domain protocol v2

The original GPT-2 task domain contained 16 unique templates per task, each
repeated eight times to produce 128 rows. Protocol v2 replaces this with unique,
position- and content-varied prompts while preserving the original templates as
a labeled regression subset. The old artifacts remain protocol-v1 records and
are not reused as protocol-v2 evidence.

The deterministic protocol is fixed in
`configs/gpt2_behavior_domain_v2.json`. It creates, for each task, 256 unique
synthesis prompts and 256 disjoint gate prompts, balanced across the two
candidate classes. The generator consults no model outputs. Its build step
records the tokenizer-vocabulary hash, prompt hashes, contextual opener-token
alignment, candidate-token checks, and the observed range of opener positions.
The gate split is not used to select programs or program subsets.

```bash
python scripts/gpt2/build_behavior_domains.py \
  --tokenizer_path artifacts/gpt2-norm-free \
  --output_dir artifacts/gpt2-behavior-domains-v2 \
  --config configs/gpt2_behavior_domain_v2.json
```

Before C2, the base model must score 1.000 against the explicit reference
program P(x) on both splits. C2 and C3 use only `synthesis.json`. After per-head
synthesis, `select_joint_program_subset.py` adds programs while checking the
full and circuit-only forwards for both synthesis tasks together, then
evaluates the fixed subset once on
`gate.json`; a failure stops before healing and does not trigger gate-specific
adaptation. Healing targets P(x), with the separately checked base decisions
coinciding with P(x), and keeps the locked OpenWebText perplexity budget of
28.617593822841776.

The v2 core-aware objective applies task loss to the selected circuit, samples
non-circuit paths, and rotates joint and individual program lesions through
both the full graph and the preregistered circuit. Suppression visit counts are
logged. A healing result passes only after a complete, unsampled lesion sweep
shows exact circuit-only P(x) accuracy, individual necessity in both full and
core forwards, and no joint bypass, in addition to exact full-model P(x)
accuracy and the perplexity gate. `scripts/gpt2/run_phase_c.py` enforces this
ordering and writes v2 artifacts under `artifacts/*-v2` paths.

### Phase C behavior-domain protocol v3

The v2 run stopped before healing because its selected `bracket_type` circuit,
although exact on all 256 synthesis prompts, was correct on only 247 of the 256
untouched gate prompts. Protocol v2 remains unchanged as that run's record.

Protocol v3 permanently reclassifies both v2 splits as development data. Its
512 development prompts per task are exactly the union of the v2 synthesis and
gate prompts. The next 256 rows from the same deterministic, model-independent
generator form a fresh gate and are disjoint from every v2 prompt. The old v2
gate is therefore burned as gate material and cannot be reused for evaluation.

Before any v3 result is computed, circuit selection is fixed as follows: test
the existing v2 threshold circuits on all 512 development prompts; among those
exact against P(x), select the circuit with the fewest edges, breaking ties by
lower threshold. Those circuits were extracted using only v2 synthesis rows,
not its old gate. If a task has no exact existing candidate, rerun C2 for that
task on the 512-row development domain. C3 is then rerun on all 512 development
prompts. The fresh v3 gate is first evaluated after circuit selection and never
changes a circuit or program subset.

The locked protocol is `configs/gpt2_behavior_domain_v3.json`; new artifacts
are written under `artifacts/*-v3`, while all v2 artifacts are preserved.

The v3 run also stopped before healing. The selected `bracket_type` circuit was
exact on all 512 development prompts but correct on 255 of 256 fresh gate
prompts; the full model and quote circuit remained exact. Together with v2's
247/256 result, this indicates that bracket type is less localized under
zero-ablation than quote closing: low-scoring contributions pruned on
development remain load-bearing for a small held-out subset.

### Final Phase C behavior-domain protocol v4

Protocol v4 is the final exact-generalization attempt; there is no protocol
v5. All 768 v3 development and gate prompts are permanently burned as gate
material and become v4 development data. The next 512 deterministic prompts
per task are the final untouched gate, balanced 256/256 across the two classes.
The finite generator has ample capacity: each quote stratum contains 1,184
unique prompts and each bracket stratum 1,187, while v4 consumes 640 per
stratum. Every development and gate stratum contains all eight templates, at
least 12 opener-token positions, and at least 14 token-length values. Prompt
digests and these coverage requirements are locked in
`configs/gpt2_behavior_domain_v4.json`.

V4 evaluates every available v2/v3 threshold circuit on the 768-row
development domain. Among exact circuits it first maximizes the minimum
per-example signed correct-token margin, then minimizes edge count, then uses
the lower threshold. This robustness-first ordering is preregistered because
the consecutive bracket failures occurred at the circuit boundary despite
exact development agreement; it selects for boundary slack rather than
parsimony. If no existing candidate is exact, only the failing task is
re-extracted on all 768 rows. Programs are then resynthesized on those rows.

The v3 failing brace prompt is recorded retrospectively with its stratum,
template, prior selected-circuit margin, and every alternative candidate's
margin. The report also states whether robustness-first selection among the
v3-development-exact candidates would have classified it correctly.

If the final v4 gate fails, Phase C stops before healing. The result is reported
as measured near-exact faithfulness, including per-task and per-stratum gate
agreement, and no exact circuit claim is made. Larger or lower-threshold
circuits require a separate future preregistration with new evaluation data;
they are not selected against the v4 gate.

The final v4 run stopped under that rule on 2026-07-22. Robustness-first
selection chose a 17-edge quote-close circuit at threshold 0.01 and, after the
existing candidates were insufficient, a newly extracted 340-edge bracket-type
circuit at threshold 0.01. Both were exact against P(x) on all 768 development
prompts. On the untouched 512-prompt gate, the full norm-free model was exact
on both tasks and the bracket-type circuit was exact (512/512), while the
quote-close circuit scored 511/512 overall: 256/256 for single quotes and
255/256 for double quotes. The sole mismatch was
`v4:quote_close:double:88ee2f642c32fef2`.

The failure occurred during the base-full-and-circuit preflight. Candidate
program synthesis had completed, but no joint program subset was selected or
filtered, no program was installed, and healing, migration checks, and SMT
verification did not run. Therefore this is a near-exact zero-ablation circuit
result, not a program-composition or healing result. The exact-generalization
track is closed with no protocol v5. The locked manifests, selected circuits,
synthesis output, and stop report are preserved under the corresponding
`artifacts/gpt2-*-v4/` paths.

Before extracting circuits, test whether the model actually exhibits the target behaviors. This prevents wasting time extracting "circuits" for behaviors the model does not perform.

The behavior scanner tests 2 categories:
- `quote_close`: Single vs double quote closing (varied templates)
- `bracket_type`: `]` vs `}` distinction (varied templates)

Metrics computed:
- Binary accuracy (correct token logit > incorrect token logit)
- Mean logit difference (correct - incorrect)
- Log probabilities for both tokens
- Rank of correct token in vocabulary

Viability thresholds:
- **Strong**: accuracy ≥ 0.85 AND logit_diff ≥ 1.0
- **Viable**: accuracy ≥ 0.70 AND logit_diff > 0.0
- **None**: below viable threshold

Run the behavior viability scan:

```bash
python scripts/gpt2/extract.py \
  --model_path artifacts/gpt2-norm-free \
  --scan_behaviors \
  --domain_manifest artifacts/gpt2-behavior-domains-v4/development.json \
  --batch_size 8 \
  --output_dir artifacts/gpt2-circuits-v4/base-scan-development
```

Historical protocol-v1 results (band_norm_sparsemax model @ checkpoint-240000):

| Task | Binary Accuracy | Mean Logit Diff | Viability |
|------|----------------|-----------------|-----------|
| `quote_close` | 1.000 | 6.22 | **strong** |
| `bracket_type` | 1.000 | 4.93 | **strong** |

This generates:
- `artifacts/gpt2-circuits-v4/base-scan-development/behavior_scan/behavior_scan.json` - Detailed metrics
- `artifacts/gpt2-circuits-v4/base-scan-development/behavior_scan/behavior_scan.txt` - Human-readable report

Use the scan results to decide which tasks to extract circuits for. Focus on behaviors marked "viable" or "strong" for meaningful results.

### Extract Pruned Circuits using ACDC

Once behaviors are confirmed viable, extract a pruned circuit responsible for each behavior using a simplified ACDC-style algorithm.

Circuit extraction is not claim-neutral. The extraction objective determines what kind of claim the resulting circuit supports. In this repo, circuits are extracted using zero-ablation semantics:

$$C_E(x) = \text{the original model restricted to retained edges } E,\text{ with deleted edges contributing }0.$$

Under zero ablation, the extracted circuit is a genuine subgraph/subnetwork of the trained model. Kept edges use the original trained weights and activations; deleted edges are removed.

ACDC is greedy and order-dependent, so the extracted circuit is not guaranteed to be globally minimal.

### Claim Discipline

Use the following rule:

* If the circuit is extracted to preserve the full model's projected decisions, it can be described as a faithful projected circuit for that behavior.
* If the circuit is extracted to optimize task accuracy against a symbolic reference program, it is a task circuit, not necessarily the model's actual mechanism.
* If the circuit improves over the full model, it may be a useful task subnetwork, but it should not be described as faithfully representing the full model.

For formal claims about an actual extracted subset of the model, the circuit should preserve the full model's behavior on the same domain used for the later claim.

This is especially important for generalization or impossibility results. If a circuit is pruned only to preserve success on inputs up to some limit $k$, then failures beyond $k$ may be pruning artifacts. To support claims about failure or extrapolation, the extraction domain must include both the success cases and the failure/extrapolation cases.

### Output Projection

For task-specific circuits, we do not need to preserve the full vocabulary distribution. Instead, we preserve the model's behavior on a task-relevant output projection.

For each task, define a candidate token set $T$:

* `quote_close`: T = {', "}
* `bracket_type`: T is the set containing ] and }

The projected decision is:

$$d_T(F,x) = \arg\max_{t \in T} F_t(x)$$

where $F$ is either the full model or the extracted circuit.

ACDC should remove an edge only if removal preserves the full model's projected behavior on the extraction domain:

$$d_T(C_E,x)=d_T(M,x)$$

In practice, this is implemented with candidate-logit KL and a hard projected-agreement guard:

$$\text{KL}\left(\text{softmax}(M_T(x)) \| \text{softmax}(C_{E,T}(x))\right)$$

where $M_T(x)$ and $C_{E,T}(x)$ are the logits restricted to the candidate token set $T$.

### Extraction Methodology

The extractor:

* Defines a per-head computational graph over residual-stream components:
  * `emb`
  * `attn_i_h_j`
  * `mlp_i`
  * `logits`
* Runs the full model on task prompts.
* Computes full-model projected logits over the task candidate set $T$.
* Iteratively removes edges if removal does not significantly change projected behavior.
* Uses zero-ablation semantics for deleted edges.
* Cleans up the graph to retain only `emb → logits` paths.
* Reports sufficiency, inverse-ablation, and projected-agreement metrics.

The most important faithfulness metric is:

$$\Pr_{x \in D}[d_T(C_E,x)=d_T(M,x)]$$

For a strong circuit claim over a bounded domain, this should be `1.000` on that domain.

Run circuit extraction:

```bash
python scripts/gpt2/extract.py \
  --model_path artifacts/gpt2-norm-free \
  --extract_circuit quote_close \
  --n_examples 768 \
  --domain_manifest artifacts/gpt2-behavior-domains-v4/development.json \
  --threshold 0.01 \
  --min_agreement 1.0 \
  --output_dir artifacts/gpt2-circuits-v4/base/quote_close_t0.01
```

Available tasks: `quote_close`, `bracket_type`

The threshold parameter strongly affects circuit quality. Run a threshold sweep to find the smallest circuit that preserves full-model projected behavior:

```bash
# Run sweep (tests 6 thresholds: 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
bash scripts/gpt2/sweep_thresholds.sh quote_close

# Compare results and get recommendation
python scripts/gpt2/compare_sweeps.py \
  --sweep_dir artifacts/gpt2-circuits-v4/base \
  --task quote_close
```

The comparison tool shows the circuit quality and size for each threshold. Use the recommended circuit for verification.

### Task-Specific Extraction Details

**Quote closing**:

Goal: Extract the subcircuit responsible for choosing `'` vs `"` as the next token.

Candidate set: T = {', "}

Reference behavior: If the prompt contains an unmatched single quote, predict `'`. If the prompt contains an unmatched double quote, predict `"`.

Illustrative prompts (protocol v1; current extraction uses the deterministic
v2-v4 domain manifests with varied opener positions, lengths, and content):

```text
x = 'hello world
x = "hello world
print('hello world
print("hello world
message = 'foo bar
message = "foo bar
```

Required extraction guard: $d_T(C_E,x)=d_T(M,x)$ for all extraction examples.

Formal properties to verify after extraction:

1. **Functional correctness over the quote projection**: $\forall x \in D_{\text{quote}},\quad d_T(C_E,x)=P_{\text{quote}}(x)$

2. **Content invariance**: Changing non-quote content should not change the quote decision: $R_{\text{same quote}}(x,x') \Rightarrow d_T(C_E,x)=d_T(C_E,x')$

3. **Edge necessity**: For retained edges $e$, check whether removing $e$ changes the projected behavior on some bounded input: $\forall e \in E(C),\quad \exists x \in D_{\text{quote}}: d_T(C_E,x)\neq d_T(C_E\setminus e,x)$

4. **Continuous robustness**: For every perturbation $\eta$ to the final residual with $\max_i |\eta_i| \le \epsilon$, the projected decision remains unchanged: $\forall x \in D_q,\ \forall \eta \in \mathbb{R}^d,\ \max_i |\eta_i| \le \epsilon \Rightarrow G_T(r_E(x)+\eta)=P_q(x)$, where $D_q$ is the quote domain, $P_q$ is the quote reference program, and $G_T(r)$ is the projected decision after final normalization and unembedding.

**Bracket type**:

Goal: Extract the subcircuit responsible for choosing `]` vs `}`.

Candidate set: T is the set containing ] and }

Reference behavior: If the prompt opens with `[`, predict `]`. If the prompt opens with `{`, predict `}`.

Illustrative prompts (protocol v1; current extraction uses the deterministic
v2-v4 domain manifests with varied opener positions, lengths, and content):

```text
x = [a, b, c
x = {a, b, c
items = [foo, bar
items = {foo, bar
return [x, y, z
return {x, y, z
```

This is a bracket-vs-brace type distinction.

Required extraction guard: $d_T(C_E,x)=d_T(M,x)$ for all extraction examples.

Formal properties to verify after extraction:

1. **Functional correctness over the bracket projection**: $\forall x \in D_{\text{bracket}},\quad d_T(C_E,x)=P_{\text{bracket}}(x)$

2. **Content invariance**: Changing filler/content tokens should not flip the bracket decision if the opening delimiter is unchanged: $R_{\text{same delimiter}}(x,x') \Rightarrow d_T(C_E,x)=d_T(C_E,x')$

3. **Delimiter sensitivity**: Changing `[` to `{` should flip the projected decision: $x' = \text{replace-opening-delimiter}(x) \Rightarrow d_T(C_E,x)\neq d_T(C_E,x')$

4. **Edge necessity**: For retained edges $e$, check whether removing $e$ changes the projected behavior on some bounded input: $\forall e \in E(C),\quad \exists x \in D_{\text{bracket}}: d_T(C_E,x)\neq d_T(C_E\setminus e,x)$

5. **Continuous robustness**: For every perturbation $\eta$ to the final residual with $\max_i |\eta_i| \le \epsilon$, the projected decision remains unchanged: $\forall x \in D_b,\ \forall \eta \in \mathbb{R}^d,\ \max_i |\eta_i| \le \epsilon \Rightarrow G_T(r_E(x)+\eta)=P_b(x)$, where $D_b$ is the bracket domain, $P_b$ is the bracket reference program, and $G_T(r)$ is the projected decision after final normalization and unembedding.

### Circuit Quality Results

**Historical protocol-v1 results.** The tables below are from the BandNorm+sparsemax
checkpoint using the earlier block-level graph (325 possible edges) and the
duplicated 16-template v1 domain. They are preserved as protocol-v1 records.
Current circuits use the norm-free base, the per-head graph, and the v2-v4
domain manifests; their sweeps and selections are recorded under
`artifacts/gpt2-circuits-v2/`, `artifacts/gpt2-circuits-v3/`, and
`artifacts/gpt2-circuits-v4/`.

Circuits extracted using zero ablation, candidate_kl metric, and min_agreement=1.0 guard.

**Quote closing**:

Threshold sweep results:

| Threshold | Edges | Full Acc | Circuit Acc | Agreement | Cand KL | Full Margin | Circ Margin |
|----------:|------:|---------:|------------:|----------:|--------:|------------:|------------:|
| 0.005 | 78 | 1.000 | 1.000 | 1.000 | 0.054 | 6.225 | 3.114 |
| 0.010 | 85 | 1.000 | 1.000 | 1.000 | 0.057 | 6.225 | 2.960 |
| 0.020 | 56 | 1.000 | 1.000 | 1.000 | 0.062 | 6.225 | 2.912 |
| 0.050 | 135 | 1.000 | 1.000 | 1.000 | 0.295 | 6.225 | 1.023 |
| 0.100 | 116 | 1.000 | 1.000 | 1.000 | 0.392 | 6.225 | 0.698 |
| 0.200 | 135 | 1.000 | 1.000 | 1.000 | 0.380 | 6.225 | 0.711 |

**Selected circuit: threshold=0.020**
- Edges: 56 / 325 (17.2%)
- Projected agreement: 1.000
- Candidate KL: 0.062
- Selected path: `artifacts/circuits_sweep/quote_close_t0.02/circuit.json`

**Bracket type**:

Threshold sweep results:

| Threshold | Edges | Full Acc | Circuit Acc | Agreement | Cand KL | Full Margin | Circ Margin |
|----------:|------:|---------:|------------:|----------:|--------:|------------:|------------:|
| 0.005 | 115 | 1.000 | 1.000 | 1.000 | 0.025 | 4.931 | 3.736 |
| 0.010 | 76 | 1.000 | 1.000 | 1.000 | 0.084 | 4.931 | 2.287 |
| 0.020 | 69 | 1.000 | 1.000 | 1.000 | 0.179 | 4.931 | 1.571 |
| 0.050 | 114 | 1.000 | 1.000 | 1.000 | 0.201 | 4.931 | 1.470 |
| 0.100 | 51 | 1.000 | 0.500 | 0.500 | 0.811 | 4.931 | 1.249 |
| 0.200 | 51 | 1.000 | 1.000 | 1.000 | 0.180 | 4.931 | 1.671 |

**Selected circuit: threshold=0.200**
- Edges: 51 / 325 (15.7%)
- Projected agreement: 1.000
- Candidate KL: 0.180
- Selected path: `artifacts/circuits_sweep/bracket_type_t0.2/circuit.json`

## Formal Verification of Extracted Circuits

Once circuits are extracted, we can in theory formally verify their properties using SMT solvers. The properties vary by circuit and are described above in 3b.

Naive SMT encoding of extracted *neural* GPT-2-scale circuits is not currently tractable: the retained heads contribute bilinear QK and value-aggregation terms at hidden width 768. This is the motivation for the program-replacement route in [VERIFIED_DISTILLATION.md](VERIFIED_DISTILLATION.md): replacing a circuit's attention heads with token/position programs removes those bilinear terms entirely, which at small scale roughly halved per-edge encoding cost and made all four properties provable. Verifying a program-healed GPT-2-scale circuit is the goal of Phase C; the commands below exercise the encoder path and would in principle prove the target properties given a tractable circuit.

The SMT verification system is implemented in `/scripts/smt/` (core encoders) and `/scripts/gpt2/` (GPT-2 specific) with the following modules:

Core SMT encoders (`scripts/smt/`):
- `encoders.py`: SMT encodings of verifiable components (BandNorm, sparsemax, LeakyReLU, attention, MLP)
- `circuit.py`: Circuit forward pass with edge masking
- `domain.py`: Input sequence generation for bounded verification
- `properties.py`: Property verification functions using Z3
- `utils.py`: Utility functions

GPT-2 specific (`scripts/gpt2/`):
- `model_weights.py`: Model weight extraction for GPT-2
- `verify.py`: Verification script for GPT-2 circuits

**Usage:**

Scripts to run sanity tests to verify SMT encoder matches PyTorch circuit:

```bash
# Sanity test for quote_close
python scripts/gpt2/test_smt_encoder.py \
  --model_path artifacts/gpt2-norm-free \
  --circuit_path artifacts/gpt2-circuits-v4/base-selected/quote_close/circuit.json \
  --task quote_close \
  --tolerance 1e-2

# Sanity test for bracket_type
python scripts/gpt2/test_smt_encoder.py \
  --model_path artifacts/gpt2-norm-free \
  --circuit_path artifacts/gpt2-circuits-v4/base-selected/bracket_type/circuit.json \
  --task bracket_type \
  --tolerance 1e-2
```

Scripts to run formal verification, starting with `--max_length 3`:

```bash
# Verify quote_close circuit at length 3
python scripts/gpt2/verify.py \
  --circuit_path artifacts/gpt2-circuits-v4/base-selected/quote_close/circuit.json \
  --task quote_close \
  --output_dir artifacts/verification/quote_close \
  --model_path artifacts/gpt2-norm-free \
  --max_length 3 \
  --timeout_ms 60000

# Verify bracket_type circuit at length 3
python scripts/gpt2/verify.py \
  --circuit_path artifacts/gpt2-circuits-v4/base-selected/bracket_type/circuit.json \
  --task bracket_type \
  --output_dir artifacts/verification/bracket_type \
  --model_path artifacts/gpt2-norm-free \
  --max_length 3 \
  --timeout_ms 60000
```

The GPT-2 verification script is intended to check all projected properties (functional equivalence, content invariance, edge necessity, and continuous robustness), but as mentioned, these checks are not currently tractable for the extracted GPT-2-scale circuits.
