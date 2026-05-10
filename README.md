# Verifiable Transformers

We design a Transformer variant whose entire forward pass can be encoded exactly in an SMT solver, so the solver can reason about the model’s behavior over all inputs in a bounded domain. Once the full model is SMT-representable, we formally check properties like:

* whether the model is exactly equivalent to a given symbolic program on all sequences of length ≤ n,
* whether particular connections are necessary for the model’s behavior (circuit minimality)
* whether the model obeys structural constraints (like local attention), and
* impossibility results on circuits (e.g., counts above k collapse to the same internal state)

The core idea is to make the whole Transformer verifiable so these properties can be proven or refuted over the entire input domain, not just tested on examples.

There are two parts of the Transformer architecture that are traditionally difficult to SMT encode. First, the attention mechanism. Starting from the Deep Sets characterization of permutation-invariant functions, we re-derive the [structural form of attention](https://www.neelsomaniblog.com/p/a-minimal-route-to-transformer-attention) and show that it naturally decomposes into three learnable components: an aggregator ρ, a relevance scoring function u, and a content map v. We then characterize the verifiable subset of this space: all attention mechanisms whose computation can be expressed using only affine maps, piecewise-linear transformations, thresholding, and Top-k style selection - primitives that admit exact SMT encodings. Second, we apply a similar approach to the LayerNorm. (A third non-verifiable component, the GELU activation function, is easier to address.)

We show, end-to-end, that a Transformer can be trained and then formally analyzed at the circuit level. In this work, we focus on extracting circuits for Python syntax tasks:

* **Quote closing**: Distinguishing single quote `'` vs double quote `"` continuation
* **Bracket type**: Distinguishing `]` vs `}` for list vs dict closing
* **Induction (ABCAB)**: Pattern completion (A B C ... A B → predict C)

These tasks allow us to demonstrate proofs of bounded correctness, structural properties of the circuit, and impossibility/generalization limits. This work suggests a new direction for interpretable and certifiable sequence modeling.

## Hardware Setup

The following was used to produce the results below. This heavy setup isn't strictly required, but recommended for iteration speed.

- GPUs: `8x A100 80GB` (or `8x H100 80GB`)
- Per-GPU VRAM target: `40GB+` (prefer `80GB`)
- CPU RAM: `128GB` recommended (`64GB` minimum)
- Storage: `200GB+` fast SSD for dataset cache/checkpoints/logs
- CPU: `16+` cores preferred for data loading/tokenization

Note for RunPod:

```bash
mkdir -p /workspace/.cache /workspace/.config /workspace/.git-templates
mkdir -p /workspace/.hf/{datasets,transformers,hub,tmp,accelerate}
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export XDG_CONFIG_HOME=/workspace/.config
export GIT_CONFIG_GLOBAL=/workspace/.gitconfig
export GIT_CONFIG_SYSTEM=/workspace/.gitconfig_system
export GIT_TEMPLATE_DIR=/workspace/.git-templates
export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export HUGGINGFACE_HUB_CACHE=/workspace/.hf/hub
export HF_DATASETS_TMP=/workspace/.hf/tmp
export ACCELERATE_CONFIG_DIR=/workspace/.hf/accelerate
export PIP_CACHE_DIR=/workspace/.cache/pip
export TMPDIR=/workspace/tmp
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or on RunPod:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"

pip install --upgrade pip
pip install transformers datasets evaluate tqdm matplotlib psutil pyyaml packaging huggingface_hub safetensors
pip install accelerate --no-deps
```

If your environment already provides PyTorch (for example, RunPod images), install only the project dependencies above and keep the preinstalled torch build.

## Formal Definitions

**Notation:**
- $M$ = verifiable Transformer
- $P$ = symbolic reference program
- $\Sigma$ = token alphabet
- $\Sigma^{\leq n} := \bigcup_{\ell=1}^{n} \Sigma^{\ell}$ = all sequences of length at most $n$
- $y_M(x)$ = output for model $M$ on input $x$
- $C$ = algorithmic circuit of $M$
- $C \setminus e$ = circuit with connection $e$ removed
- $\varphi_M(x)$ = output projection of $M$

For a bounded domain $\Sigma^{\leq n}$, we verify the following properties:

| Property | Informal Definition | Formal Definition |
|----------|---------------------|-------------------|
| **Functional Equivalence** | The model is equivalent to a specific symbolic program on all inputs of length ≤ n. For a code generation model trained on simple transformations (e.g. string manipulation), we can prove that for all inputs ≤ n, the model is equivalent to a reference implementation. If the property fails, the solver returns a concrete counterexample. | $\forall x \in \Sigma^{\leq n}, \quad y_M(x) = P(x)$ |
| **Circuit Minimality** | An algorithmic circuit contains the minimal number of edges. We can prove that removing an attention pathway does not change the output for any input ≤ n. This provides a principled way to eliminate potentially harmful circuits. | $\forall e \in E(C), \quad \exists x \in \Sigma^{\leq n} \text{ such that } C(x) \neq (C \setminus e)(x)$ |
| **Structural Invariants** | The model obeys structural constraints (e.g. local attention). We can guarantee that sensitive information (e.g. earlier tokens containing secrets) cannot influence outputs beyond a fixed window. | $\forall x \in \Sigma^{\leq n}, \quad S(M, x)$ where $S$ encodes locality, sparsity, monotonicity, causality, etc. |
| **Impossibility Results** | The model produces identical outputs for two classes of inputs. We can prove that for all inputs ≤ n, the model cannot distinguish between two programs that differ only in variable renaming (e.g. renaming x to y throughout). This establishes that the model has learned a representation invariant to variable identity. | $\forall x, x' \in \Sigma^{\leq n}, \quad R(x, x') \Rightarrow \varphi_M(x) = \varphi_M(x')$ where $R$ identifies input pairs the model cannot distinguish |

## Step 1: Baseline GPT-2 (Open-Source)

This repository includes a reproducible baseline using `gpt2` (124M) with OpenWebText training and WikiText-103 evaluation.

We evaluate both a pretrained GPT-2 reference run and a locally reproduced reference GPT-2-small model.

### Baseline Definition and Comparison Protocol

We define the baseline as our own locally reproduced reference GPT-2-small run, using the exact same tokenizer, preprocessing, context length, optimization recipe, token budget, and evaluation code as all later architecture variants.

We first train a local reference GPT-2-small baseline under the same pipeline used for all experiments.

Alternative attention and normalization variants are judged relative to this local baseline.

### Expected Baseline Criteria

The local reference baseline should be stable and reproducible under the configured training recipe, and all subsequent architecture changes should be compared against this exact run setup.

Primary comparison metrics:

* OpenWebText validation loss under identical preprocessing and evaluation steps.
* WikiText-103 perplexity under identical tokenizer and evaluation code.

Note: exact absolute numbers depend on hardware scale, effective batch size, training duration, and data preprocessing details. Relative comparisons are the primary acceptance signal.

### Train (OWT)

Run this first to record pretrained GPT-2 reference numbers with the same evaluation protocol:

```bash
python scripts/eval_pretrained_reference.py \
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
python -m torch.distributed.run --nproc_per_node=8 scripts/train_experiment.py \
  --config configs/step1_gpt2_small_openwebtext_resume_stable.json \
  --output_dir artifacts/step1-gpt2-small-openwebtext
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

### Switching To Stable Resume Config

To continue from a known-good checkpoint with a lower, safer learning rate, use `configs/step1_gpt2_small_openwebtext_resume_stable.json` plus `--reset_optimizer_on_resume`.

This keeps checkpoint model weights but resets optimizer/scheduler/scaler/rng by creating a reset-resume checkpoint copy under `output_dir/resume_reset_checkpoints`.

Example (opt-in WikiText target stopping):

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/train_experiment.py \
  --config configs/step1_gpt2_small_openwebtext_resume_stable.json \
  --output_dir artifacts/step1-gpt2-small-openwebtext \
  --resume_from_checkpoint artifacts/step1-gpt2-small-openwebtext/checkpoint-40000 \
  --reset_optimizer_on_resume \
  --use_wikitext_as_dev \
  --target_wikitext_ppl 43 \
  --wikitext_eval_every_n_evals 1
```

Checkpoint resume behavior:

- The training script auto-resumes from the latest checkpoint in `output_dir` when present.
- To continue for more iterations, rerun with a larger `--max_steps`.
- To resume from a specific checkpoint, use `--resume_from_checkpoint <path>`.
- To disable auto-resume, pass `--disable_auto_resume`.
- Run progress heartbeat is written to `output_dir/run_status.json` (stages like preprocessing, training, done, failed, interrupted).
- Catastrophic divergence guard is supported to stop and save if loss/gradients explode.

For quick smoke tests:

```bash
python scripts/train_experiment.py \
  --config configs/step1_gpt2_small_openwebtext_resume_stable.json \
  --output_dir artifacts/step1-smoke \
  --max_train_samples 50000 \
  --max_eval_samples 2000
```

### Evaluate WikiText-103 Perplexity

```bash
python scripts/eval_wikitext103.py \
  --model_path artifacts/step1-gpt2-small-openwebtext \
  --split validation \
  --block_size 1024 \
  --stride 1024 \
  --output_json artifacts/step1-wikitext103-validation.json
```

### Plot Training Curves

```bash
python scripts/plot_training_curves.py \
  --run_dir artifacts/step1-gpt2-small-openwebtext \
  --output_png artifacts/step1-gpt2-small-openwebtext/training_curves.png
```

This reads `trainer_state.json` from the run directory and plots train/eval loss versus training step.

### Baseline Results

Current local baseline outcome (this repository run):

* OpenWebText validation loss: 3.1340 (at step 260000)
* OpenWebText validation perplexity: 22.9650 (at step 260000)
* Relative to current pretrained reference (OWT loss 3.1187, perplexity 22.6160): +0.0153 loss and +0.3490 perplexity
* WikiText-103 perplexity: 52.9820 (at step 260000)
* Relative to current pretrained reference (WikiText-103 perplexity 31.0403): +21.9417 perplexity

Interpretation:

* WikiText-103 perplexity is substantially worse than the pretrained GPT-2 reference in this run.
* This Step 1 setup is optimized for the OpenWebText training pipeline and uses OpenWebText-based stopping criteria by default.
* We use this run as a local baseline anchor for relative architecture comparisons, not as a claim of best absolute WikiText-103 performance.

This run met the configured early-stop target (`eval_loss <= 3.2`).

## Step 2: Verifiable Replacement Evaluations

We use the same training script and process, with architecture variants controlled via config (`norm_variant`, `attn_variant`) to keep training/eval pipeline constant. We first establish the pretrained GPT-2 reference and local Step 1 baseline above, then compare these variants against that local baseline.

### Step 2a: LayerNorm replacement only

#### DyT (Failed)

Config: `configs/step2a_norm_dyt_only.json`

DyTNorm replaces LayerNorm with an affine → clamp → affine transformation. It is fully SMT-encodable, but it is not a true normalization layer: it provides neither data-dependent centering nor scale control. In practice, training improved briefly, then gradient norms rose and the run diverged, consistent with uncontrolled residual accumulation.

#### Verifiable PWL Norm v1 (Failed)

Config: `configs/step2a_norm_verifiable_pwl_v1.json`

PiecewiseLinearNorm v1 restored two important normalization ingredients — mean-centering and adaptive rescaling via mean absolute deviation (MAD) — while remaining SMT-encodable. However, its gain function used coarse discontinuous buckets `[4.0, 2.0, 1.0, 0.5, 0.25]`, which caused abrupt scaling changes and could over-amplify low-MAD tokens. Empirically, it trained at first but later showed rising gradient norms and eventual divergence.

#### Verifiable PWL Norm v2: Soft Leaky Clamp (Failed)

Config: `configs/step2a_norm_verifiable_pwl_v2.json`

v2 simplified the norm to mean subtraction, leaky piecewise-linear clamp, fixed scaling by 0.5, and learned bias. This results in better gradient flow relative to a hard clamp, but the clamp remained unbounded: outside the clamp region, activations still grew linearly. The result was better early optimization but eventual explosion once residual accumulation returned. The key lesson was that gradient flow alone is not enough; the norm must also enforce explicit magnitude control.

#### Verifiable PWL Norm v3: Bounded PWL Clamp (Stable but High Error)

Config: `configs/step2a_norm_verifiable_pwl_v3.json`

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

#### Signed L1 Band Norm: Projection-Based Normalization (Recommended)

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

Config: `configs/step2a_norm_signed_l1_band_norm.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/train_experiment.py \
  --config configs/step2a_norm_signed_l1_band_norm.json \
  --output_dir artifacts/step2a-signed-l1-band-norm-only \
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

### Step 2b: Attention replacement only (sparsemax)

Sparsemax replaces softmax attention weighting with a piecewise-linear alternative (alpha-entmax family, alpha=2) that can produce exact zeros in the attention distribution. This is fully SMT-encodable.

Implementation:
- Uses standard LayerNorm (not verifiable norm variants)
- Replaces softmax with sparsemax via monkey-patching GPT2Attention.forward
- Runs sparsemax computation in fp32 for numerical stability (upcasts from bf16)
- Includes fail-fast verification on startup to ensure patch is active
- Compatible with transformers 4.49.0

Verification (run before full training to make sure patch is working):
```bash
python scripts/verify_sparsemax.py
```

Config: `configs/step2b_attn_sparsemax_only.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/train_experiment.py \
  --config configs/step2b_attn_sparsemax_only.json \
  --output_dir artifacts/step2b-attn-sparsemax-only \
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

### Step 2c: Combined verifiable replacements (norm + attention + LeakyReLU)

This combines the two verifiable components from steps 2a and 2b, in addition to replacing the GELU activation function.

This represents the **end-to-end verifiable Transformer**: normalization, attention, and activations are fully SMT-encodable.

Config: `configs/step2c_band_norm_sparsemax.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/train_experiment.py \
  --config configs/step2c_band_norm_sparsemax.json \
  --output_dir artifacts/step2c-band-norm-sparsemax \
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

## Step 3: Circuit Extraction and Formal Verification

Once the verifiable model is trained, we extract pruned circuits responsible for specific behaviors and formally verify their properties using SMT solvers.

### Step 3a: Verify behaviors are reliably evoked before circuit extraction

Before extracting circuits, test whether the model actually exhibits the target behaviors. This prevents wasting time extracting "circuits" for behaviors the model does not perform.

The behavior scanner tests 3 categories:
- `quote_close`: Single vs double quote closing (varied templates)
- `bracket_type`: `]` vs `}` distinction (varied templates)
- `induction_ABCAB`: Pattern completion (A B C ... A B → predict C)

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
python scripts/circuits/extract_circuit.py \
  --model_path artifacts/step2c-band-norm-sparsemax/checkpoint-240000 \
  --scan_behaviors \
  --n_examples 256 \
  --batch_size 8 \
  --output_dir artifacts/circuits
```

#### Results (Step 2c model @ checkpoint-240000):

| Task | Binary Accuracy | Mean Logit Diff | Viability |
|------|----------------|-----------------|-----------|
| `quote_close` | 1.000 | 6.22 | **strong** |
| `bracket_type` | 1.000 | 4.62 | **strong** |
| `induction_ABCAB` | 0.887 | 3.67 | **strong** |

This generates:
- `artifacts/circuits/behavior_scan/behavior_scan.json` - Detailed metrics
- `artifacts/circuits/behavior_scan/behavior_scan.txt` - Human-readable report

Use the scan results to decide which tasks to extract circuits for. Focus on behaviors marked "viable" or "strong" for meaningful results.

### Step 3b: Extract pruned circuits using ACDC

Once behaviors are confirmed viable, extract a pruned circuit responsible for each behavior using a simplified ACDC-style algorithm.

Circuit extraction is not claim-neutral. The extraction objective determines what kind of claim the resulting circuit supports. In this repo, circuits are extracted using zero-ablation semantics:

$$C_E(x) = \text{the original model restricted to retained edges } E,\text{ with deleted edges contributing }0.$$

Under zero ablation, the extracted circuit is a genuine subgraph/subnetwork of the trained model. Kept edges use the original trained weights and activations; deleted edges are removed.

ACDC is greedy and order-dependent, so the extracted circuit is not guaranteed to be globally minimal.

#### Claim discipline

Use the following rule:

* If the circuit is extracted to preserve the full model's projected decisions, it can be described as a faithful projected circuit for that behavior.
* If the circuit is extracted to optimize task accuracy against a symbolic reference program, it is a task circuit, not necessarily the model's actual mechanism.
* If the circuit improves over the full model, it may be a useful task subnetwork, but it should not be described as faithfully representing the full model.

For formal claims about an actual extracted subset of the model, the circuit should preserve the full model's behavior on the same domain used for the later claim.

This is especially important for generalization or impossibility results. If a circuit is pruned only to preserve success on inputs up to some limit $k$, then failures beyond $k$ may be pruning artifacts. To support claims about failure or extrapolation, the extraction domain must include both the success cases and the failure/extrapolation cases.

#### Output projection

For task-specific circuits, we do not need to preserve the full vocabulary distribution. Instead, we preserve the model's behavior on a task-relevant output projection.

For each task, define a candidate token set $T$:

* `quote_close`: T = {', "}
* `bracket_type`: T is the set containing ] and }
* `induction_ABCAB`: $T$ is the set of candidate pattern tokens, e.g. `red`, `blue`, `green`, `cat`, etc.

The projected decision is:

$$d_T(F,x) = \arg\max_{t \in T} F_t(x)$$

where $F$ is either the full model or the extracted circuit.

ACDC should remove an edge only if removal preserves the full model's projected behavior on the extraction domain:

$$d_T(C_E,x)=d_T(M,x)$$

In practice, this is implemented with candidate-logit KL and a hard projected-agreement guard:

$$\text{KL}\left(\text{softmax}(M_T(x)) \| \text{softmax}(C_{E,T}(x))\right)$$

where $M_T(x)$ and $C_{E,T}(x)$ are the logits restricted to the candidate token set $T$.

#### Extraction methodology

The extractor:

* Defines a coarse computational graph over residual-stream components:
  * `emb`
  * `attn_i`
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

#### Run circuit extraction

Quote closing:

```bash
python scripts/circuits/extract_circuit.py \
  --model_path artifacts/step2c-band-norm-sparsemax/checkpoint-240000 \
  --extract_circuit quote_close \
  --n_examples 128 \
  --threshold 0.01 \
  --ablation zero \
  --output_dir artifacts/circuits
```

Bracket type:

```bash
python scripts/circuits/extract_circuit.py \
  --model_path artifacts/step2c-band-norm-sparsemax/checkpoint-240000 \
  --extract_circuit bracket_type \
  --n_examples 128 \
  --threshold 0.01 \
  --ablation zero \
  --output_dir artifacts/circuits
```

Induction:

```bash
python scripts/circuits/extract_circuit.py \
  --model_path artifacts/step2c-band-norm-sparsemax/checkpoint-240000 \
  --extract_circuit induction_ABCAB \
  --n_examples 128 \
  --threshold 0.01 \
  --ablation zero \
  --output_dir artifacts/circuits
```

Available tasks: `quote_close`, `bracket_type`, `induction_ABCAB`

This generates:

* `artifacts/circuits/<task>/circuit.json` — circuit edges, metrics, and extraction log
* `artifacts/circuits/<task>/circuit.dot` — Graphviz visualization
* `artifacts/circuits/<task>/summary.txt` — human-readable summary

Key parameters:

* `--ablation zero`: deleted edges contribute zero.
* `--threshold`: maximum KL increase allowed when removing an edge.
* `--n_examples`: number of extraction examples.
* `--trim_rounds`: additional trimming passes after ACDC. Start with `0`; extra trimming can harm faithfulness.

#### Task-specific extraction details

##### Quote closing: `quote_close`

Goal: Extract the subcircuit responsible for choosing `'` vs `"` as the next token.

Candidate set: $T_{\text{quote}} = \{', "\}$

Reference behavior: If the prompt contains an unmatched single quote, predict `'`. If the prompt contains an unmatched double quote, predict `"`.

Recommended extraction domain: single-quote and double-quote prompts with varied but token-aligned content

Example prompts:

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

##### Bracket type: `bracket_type`

Goal: Extract the subcircuit responsible for choosing `]` vs `}`.

Candidate set: $T_{\text{bracket}} = \{], \}\}$

Reference behavior: If the prompt opens with `[`, predict `]`. If the prompt opens with `{`, predict `}`.

Recommended extraction domain: Use token-aligned bracket/brace pairs:

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

##### Induction: `induction_ABCAB`

Goal: Extract the subcircuit responsible for ABCAB-style pattern completion.

Candidate set: $T_{\text{induction}} = \{\text{pattern candidate tokens}\}$

Example token pool: ` red`, ` blue`, ` green`, ` cat`, ` dog`, ` tree`, ` car`, ` book`, ` city`, ` river`, ...

Reference behavior: A B C ... A B → predict C

Example prompt: ` red blue green foo bar baz red blue`

Correct next token: ` green`

Induction circuits are more vulnerable to pruning artifacts than quote/bracket circuits. The full model may be imperfect. Therefore extraction should preserve the full model's projected decision, not merely maximize correctness against the symbolic rule.

Required extraction guard: $d_T(C_E,x)=d_T(M,x)$ on the extraction domain.

Formal properties to verify after extraction:

1. **Restricted functional correctness**: For bounded ABCAB patterns: $\forall A,B,C,F \in D,\quad d_T(C_E, A,B,C,F,A,B)=C$

2. **Token-renaming equivariance**: Consistent renaming of the pattern tokens should rename the output: $d_T(C_E, r(x)) = r(d_T(C_E,x))$

3. **Filler invariance**: Changing filler tokens should not change the predicted continuation token, assuming the ABCAB structure is preserved: $R_{\text{same ABCAB}}(x,x') \Rightarrow d_T(C_E,x)=d_T(C_E,x')$

4. **Failure/generalization claims**: Only make impossibility or failure claims on domains included in the extraction/faithfulness check. For example, to claim that an induction circuit cannot generalize beyond a pattern length $k$, the extraction domain must include both patterns of length $\leq k$ and patterns of length $> k$, and the circuit must preserve the full model's projected decisions on both. Otherwise, the failure may simply be caused by pruning.

#### Results (Step 2c model @ checkpoint-240000)

Results to be filled in after extraction with zero ablation and candidate KL metric.

### Step 3c: Formal verification

Once circuits are extracted, we formally verify their properties using SMT solvers. The properties vary by circuit and are described above in 3b.

#### Running SMT verification

The SMT verification system is implemented in `/scripts/smt_verify/` with the following modules:

- `encoders.py`: SMT encodings of verifiable components (BandNorm, sparsemax, LeakyReLU, attention, MLP)
- `circuit.py`: Circuit forward pass with edge masking
- `bounded_domain.py`: Input sequence generation for bounded verification
- `properties.py`: Property verification functions using Z3
- `model_weights.py`: Model weight extraction and lightweight test weights

**Usage:**

```bash
# Verify quote_close circuit
python scripts/verify_circuit.py \
  --circuit_path outputs/circuits/quote_close/circuit.json \
  --task quote_close \
  --output_dir outputs/verification/quote_close \
  --model_path artifacts/step2c-band-norm-sparsemax/checkpoint-240000

# Verify induction circuit
python scripts/verify_circuit.py \
  --circuit_path outputs/circuits/induction_ABCAB/circuit.json \
  --task induction_ABCAB \
  --output_dir outputs/verification/induction_ABCAB \
  --model_path artifacts/step2c-band-norm-sparsemax/checkpoint-240000
```
