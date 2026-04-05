# Verifiable Transformers

We design a Transformer variant whose entire forward pass can be encoded exactly in an SMT solver, so the solver can reason about the model’s behavior over all inputs in a bounded domain. Once the full model is SMT-representable, we formally check properties like:

* whether the model is exactly equivalent to a given symbolic program on all sequences of length ≤ n,
* whether particular connections are necessary for the model’s behavior (circuit minimality)
* whether the model obeys structural constraints (like local attention), and
* impossibility results on circuits (e.g., counts above k collapse to the same internal state)

The core idea is to make the whole Transformer verifiable so these properties can be proven or refuted over the entire input domain, not just tested on examples.

There are two parts of the Transformer architecture that are traditionally difficult to SMT encode. First, the attention mechanism. Starting from the Deep Sets characterization of permutation-invariant functions, we re-derive the structural form of attention and show that it naturally decomposes into three learnable components: an aggregator ρ, a relevance scoring function u, and a content map v. We then characterize the verifiable subset of this space: all attention mechanisms whose computation can be expressed using only affine maps, piecewise-linear transformations, thresholding, and Top-k style selection - primitives that admit exact SMT encodings. Second, we apply a similar approach to the LayerNorm.

We show, end-to-end, that a Transformer can be trained and then formally analyzed at the circuit level: we extract the learned addition mechanism, prove bounded correctness, prove structural properties of the circuit, and prove impossibility/generalization limits. This work suggests a new direction for interpretable and certifiable sequence modeling.

## Hardware Requirements

To target strong local baseline reproduction and stable multi-GPU training, use datacenter-class multi-GPU training.

- Recommended: `8x A100 80GB` (or `8x H100 80GB`)
- Acceptable (slower, more tuning sensitive): `4x A100 40/80GB`
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

## Individual Verifiable Replacement Evaluations

We use the same training script and process, with architecture variants controlled via config (`norm_variant`, `attn_variant`) to keep training/eval pipeline constant. We first establish the pretrained GPT-2 reference and local Step 1 baseline above, then compare these variants against that local baseline.

### Step 2a: LayerNorm replacement only

#### DyT: Failed Replacement

Config: `configs/step2a_norm_dyt_only.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/train_experiment.py \
  --config configs/step2a_norm_dyt_only.json \
  --output_dir artifacts/step2a-norm-dyt-only \
  --disable_auto_resume \
  --use_wikitext_as_dev \
  --target_wikitext_ppl 53 \
  --wikitext_eval_every_n_evals 1
```

Early DyT-only runs were not stable under the baseline GPT-2 recipe. Although training initially progressed and WikiText-103 perplexity improved, the run later regressed and eventually diverged, indicating that the current DyT formulation is not yet a drop-in LayerNorm replacement. In particular, the current module is affine-clamp-affine rather than a true normalization layer, so it likely fails to provide the residual-scale control that GPT-2 relies on.

#### Piecewise Linear Norm v1: Failed Replacement

PiecewiseLinearNorm v1 was an SMT-encodable normalization layer designed as a closer drop-in replacement for LayerNorm than DyT. Unlike DyT, which is purely affine-clamp-affine, PiecewiseLinearNorm v1 included actual centering and adaptive scaling based on input statistics, while maintaining SMT-friendly piecewise linearity.

Config: `configs/step2a_norm_pwl_only.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/train_experiment.py \
  --config configs/step2a_norm_pwl_only.json \
  --output_dir artifacts/step2a-norm-pwl-only \
  --disable_auto_resume \
  --use_wikitext_as_dev \
  --target_wikitext_ppl 53 \
  --wikitext_eval_every_n_evals 1
```

This first PiecewiseLinearNorm attempt was also not stable under the baseline GPT-2 recipe. Relative to DyT, it restored two important ingredients of normalization: centering and adaptive rescaling using mean absolute deviation (MAD). However, the gain function used coarse discontinuous buckets with large jumps (4.0, 2.0, 1.0, 0.5, 0.25), and low-MAD tokens could be amplified too aggressively. This likely made optimization brittle and weakened stable residual-scale control across depth.

Empirically, training progressed at first, but later showed the same overall pattern as DyT: early loss improvement followed by rising gradient norms and eventual divergence. In the observed run, OpenWebText eval loss was about 4.03 at 60k steps and about 4.02 at 80k steps, while WikiText-103 perplexity remained very high at about 182 and 171 respectively. Gradient norm rose from roughly 1 early in training to roughly 9 by around 80k steps, and then to about 93.6 when the catastrophic divergence guard fired near 98.8k steps.

These results suggest that simply restoring centering plus coarse MAD bucketization is still not enough. A fully verifiable replacement appears to need smoother bounded scale control.

#### Verifiable PWL Norm v2 with Residual Gating

Verifiable PWL Norm v2 is the next fully verifiable normalization attempt. It keeps the entire model SMT-encodable while addressing the main failure modes of both earlier norm replacements.

The design goal is to preserve true centering and adaptive rescaling, but avoid the brittle discontinuities of PiecewiseLinearNorm v1. Instead of coarse gain buckets, this variant uses a continuous piecewise-linear approximation to inverse MAD with a conservative bounded gain range. It also adds per-branch residual gating, using learned scalar gates clamped into [0, 1], to provide an additional SMT-friendly stabilizer for deep residual optimization.

This variant is still fully verifiable:

- normalization uses only affine maps, mean reductions, abs, comparisons, clamp, and piecewise-linear functions
- residual gating uses only scalar multiplication and clamp
- no LayerNorm remains anywhere in the verifiable variant

Config: `configs/step2a_norm_verifiable_pwl_gated.json`

```bash
python -m torch.distributed.run --nproc_per_node=8 scripts/train_experiment.py \
  --config configs/step2a_norm_verifiable_pwl_gated.json \
  --output_dir artifacts/step2a-norm-verifiable-pwl-gated \
  --disable_auto_resume \
  --use_wikitext_as_dev \
  --target_wikitext_ppl 53 \
  --wikitext_eval_every_n_evals 1
```

### Step 2b: Attention replacement only (PWL/alpha-entmax style)

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

Implementation details:

- `norm_variant=dyt` replaces GPT-2 LayerNorm modules with a piecewise-linear DyT-style normalization substitute.
- `attn_variant=sparsemax` replaces softmax attention weighting with sparsemax (alpha-entmax family, alpha=2).
