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

To target benchmark-quality replication (matching OpenWebText validation loss and WikiText-103 perplexity within ~1-2%), use datacenter-class multi-GPU training.

- Recommended: `8x A100 80GB` (or `8x H100 80GB`)
- Acceptable (slower, more tuning sensitive): `4x A100 40/80GB`
- Per-GPU VRAM target: `40GB+` (prefer `80GB`)
- CPU RAM: `128GB` recommended (`64GB` minimum)
- Storage: `200GB+` fast SSD for dataset cache/checkpoints/logs
- CPU: `16+` cores preferred for data loading/tokenization

## Step 1: Baseline GPT-2 (Open-Source)

This repository includes a reproducible baseline using `gpt2` (124M) with OpenWebText training and WikiText-103 evaluation.

Model initialization behavior:

- The baseline model is initialized from architecture config (`AutoConfig.from_pretrained("gpt2")`) and trained from scratch.
- The baseline model is initialized from architecture config (`GPT2Config.from_pretrained("gpt2")`) and trained from scratch with `GPT2LMHeadModel`.
- It does not initialize LM weights from pretrained checkpoints.

### Files

- Training script: `scripts/train_step1.py`
- WikiText-103 perplexity script: `scripts/eval_wikitext103.py`
- Pretrained reference eval script: `scripts/eval_pretrained_reference.py`
- Baseline config: `configs/step1_gpt2_small_openwebtext.json`
- Dependencies: `requirements.txt`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train (OWT)

Run this first to record pretrained GPT-2 reference numbers with the same evaluation protocol:

```bash
python scripts/eval_pretrained_reference.py \
  --model_name gpt2 \
  --block_size 1024 \
  --stride 1024 \
  --output_json artifacts/pretrained-gpt2-reference-metrics.json
```

Then train the local baseline:

```bash
python scripts/train_step1.py \
  --config configs/step1_gpt2_small_openwebtext.json \
  --output_dir artifacts/step1-gpt2-small-openwebtext
```

This run uses early stopping on OpenWebText validation loss by default (`early_stop_eval_loss = 3.2` in the config).

Performance defaults in the baseline config:

- Preprocessing uses multiprocessing (`preprocessing_num_proc`) and caches processed datasets to disk.
- Subsequent runs reuse cached datasets instead of retokenizing/regrouping.
- `eval_steps` and `save_steps` are intentionally set high to reduce overhead during long runs.
- `torch_compile` is enabled by default.
- `bf16=true`, `fp16=false`, and gradient checkpointing is off unless memory constraints require it.

Checkpoint resume behavior:

- The training script auto-resumes from the latest checkpoint in `output_dir` when present.
- To continue for more iterations, rerun with a larger `--max_steps`.
- To resume from a specific checkpoint, use `--resume_from_checkpoint <path>`.
- To disable auto-resume, pass `--disable_auto_resume`.

Distributed launch example (8 GPUs):

```bash
torchrun --nproc_per_node=8 scripts/train_step1.py \
  --config configs/step1_gpt2_small_openwebtext.json \
  --output_dir artifacts/step1-gpt2-small-openwebtext
```

For quick smoke tests:

```bash
python scripts/train_step1.py \
  --config configs/step1_gpt2_small_openwebtext.json \
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
