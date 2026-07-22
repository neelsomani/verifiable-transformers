# Verifiable Transformers

We design a Transformer variant whose forward pass is built from SMT-representable primitives, enabling exact formal reasoning for sufficiently small circuits and bounded domains. Once the full model is SMT-representable, we formally check properties like:

* whether a circuit's task decision is exactly equivalent to a given symbolic program on every sequence in the finite task domain,
* whether particular retained connections are necessary for a circuit's task behavior,
* whether a circuit is invariant to task-irrelevant input changes, and
* whether a circuit's task decision is robust to bounded perturbations of the final residual.

The core idea is to make the architecture formally representable so that, when the hidden-state width and circuit size are tractable, these properties can be proven or refuted over an entire bounded domain rather than merely tested on examples.

There are two parts of the Transformer architecture that are traditionally difficult to SMT encode. First, the attention mechanism. Starting from the Deep Sets characterization of permutation-invariant functions, we re-derive the [structural form of attention](https://www.neelsomaniblog.com/p/a-minimal-route-to-transformer-attention) and show that it naturally decomposes into three learnable components: an aggregator ρ, a relevance scoring function u, and a content map v. We then characterize the verifiable subset of this space: all attention mechanisms whose computation can be expressed using only affine maps, piecewise-linear transformations, thresholding, and Top-k style selection - primitives that admit exact SMT encodings. Sparsemax attention is our trainable point in this space, but its QK scoring remains bilinear; we go further by replacing the attention heads of extracted task circuits with synthesized symbolic programs of token positions and identities, installed at inference time and pinned in place by ablation-aware fine-tuning, which eliminates the bilinear terms from the circuit encoding entirely (see [docs/VERIFIED_DISTILLATION.md](docs/VERIFIED_DISTILLATION.md)). Second, normalization. We train with an SMT-representable replacement for LayerNorm (Signed L1 BandNorm), but our later experiments show that removing LayerNorm after training is both cheaper in loss and strictly more provable - certificate-based robustness checks of BandNorm can return *unknown* at branch boundaries, a failure mode a norm-free model eliminates by construction (see [docs/SCALABILITY.md](docs/SCALABILITY.md)). (A third non-verifiable component, the GELU activation function, is replaced by LeakyReLU at no measured cost.)

We show, end-to-end, that a Transformer can be trained and then formally analyzed at the circuit level. In this work, we focus on symbolic syntax-like tasks:

* **Quote closing**: Distinguishing single quote `'` vs double quote `"` continuation
* **Bracket type**: Distinguishing `]` vs `}` for list vs dict closing

These tasks allow us to demonstrate proofs of bounded correctness, edge necessity, task-relevant invariance, and robustness. At GPT-2 scale, the architecture remains SMT-representable, but naive full-width SMT encoding is not yet tractable; formal proofs are intended for sufficiently small models or sufficiently sparse circuits. This work suggests a new direction for interpretable and certifiable sequence modeling.

## Formal Definitions

**Notation:**
- $P$ = symbolic reference program
- $D_\text{task}$ = finite exhaustive task domain
- $T$ = task-specific candidate token set, e.g. `{', "}` for quote closing or `{], }}` for bracket type
- $F_T(x)$ = logits of model or circuit $F$ restricted to candidate set $T$
- $d_T(F,x) := \arg\max_{t \in T} F_t(x)$ = task decision of $F$
- $C_E$ = extracted circuit with retained edge set $E$
- $C_{E\setminus\{e\}}$ = circuit after removing retained edge $e$
- $r_E(x)$ = final residual of circuit $C_E$ before final normalization
- $G_T(r)$ = task decision after applying final normalization and unembedding to residual $r$

For each task, we define a finite exhaustive task domain $D_\text{task}$ and a task-specific candidate token set $T$. The verifier proves claims about the task decision $d_T$, not equality of all vocabulary logits. This is the relevant notion for these symbolic tasks: quote closing only needs to decide between `'` and `"`, and bracket type only needs to decide between `]` and `}`. In the small-model experiments, $D_\text{task}$ is exhausted rather than sampled.

| Property | Informal Description | Formal Definition |
|----------|---------------------|-------------------|
| **Functional Equivalence** | The circuit's task decision agrees with the symbolic reference program on every input in the finite task domain. If the property fails, the solver returns a concrete counterexample. | $\forall x \in D_\text{task}, \quad d_T(C_E,x)=P(x)$ |
| **Edge Necessity** | Every retained edge is behaviorally necessary for the task: for each retained edge, there exists an input where removing that edge changes the circuit's task decision. This does not prove that ACDC found the globally smallest possible circuit; it proves that no retained edge is redundant under this criterion. | $\forall e \in E,\quad \exists x \in D_\text{task}\ \text{such that}\ d_T(C_E,x)\neq d_T(C_{E\setminus\{e\}},x)$ |
| **Invariance / Indistinguishability** | The circuit is provably insensitive to task-irrelevant variation. For example, in quote closing, changing filler/content tokens cannot change the task decision when the opening quote type is fixed. Equivalently, through the task output, the circuit cannot distinguish inputs related by $R$. | $\forall x,x'\in D_\text{task},\quad R(x,x')\Rightarrow d_T(C_E,x)=d_T(C_E,x')$ |
| **Continuous Robustness** | The circuit's task decision is stable under bounded perturbations to its final residual, such as bounded final-activation noise or quantization error at that interface. For every input in the task domain and every perturbation $\eta$ satisfying $\|\eta\|_\infty \le \epsilon$, the task decision remains unchanged. | $\forall x\in D_\text{task},\ \forall \eta\in\mathbb{R}^{d},\ \|\eta\|_\infty\le\epsilon \Rightarrow G_T(r_E(x)+\eta)=G_T(r_E(x))$. If functional equivalence is also verified, then $G_T(r_E(x)+\eta)=P(x)$. |

The robustness check is branch-certified. For each input, the verifier first proves that the traced final BandNorm branch remains stable throughout the $\epsilon$-ball. It then proves that no task-decision flip is possible within that branch. If branch stability cannot be certified, the verifier reports an unknown result rather than a proof.

## Architecture

We use a GPT-style decoder-only Transformer, but replace the components that are hard to encode exactly in SMT.

### Signed L1 BandNorm

Standard LayerNorm is difficult to encode exactly because it requires division by a data-dependent standard deviation. We replace it with Signed L1 BandNorm, a projection-based normalization operator.

Given an input vector `x`, BandNorm:

1. centers the vector: `c = x - mean(x)`
2. splits positive and negative mass: `p = max(c, 0)` and `n = max(-c, 0)`
3. controls the L1 mass of each side:
   - if mass is too large, project onto an L1 ball using thresholding
   - if mass is too small after projection, add a bounded lift over active coordinates
4. recombines the signed vector: `z = p' - n'`
5. recenters: `z = z - mean(z)`
6. applies a learned affine map: `output = gamma * z + beta`

This keeps the role of normalization — centering and scale control — while using only affine operations, comparisons, max/threshold logic, and projection-style primitives. (See [docs/SCALABILITY.md](docs/SCALABILITY.md) for the process that came to this construction.)

### Sparsemax Attention

Standard softmax attention uses exponentials and division, which are not SMT-friendly. We replace softmax with sparsemax, which projects attention scores onto the probability simplex.

Attention still computes ordinary query, key, and value projections:

```text
q = W_q x
k = W_k x
v = W_v x
```

Then causal attention scores are computed and sparsemax is applied:

```text
a = sparsemax(scores)
output = sum_j a_j v_j
```

Sparsemax is piecewise-linear and can produce exact zeros, so attention weights can be encoded using thresholding and linear constraints. The attention score computation is still bilinear in `q` and `k`, so it is SMT-representable but can become expensive at large hidden widths or sequence lengths.

### LeakyReLU MLP

GPT-2 normally uses GELU in the MLP block. GELU is not piecewise-linear, so we replace it with LeakyReLU:

```text
LeakyReLU(x) = x        if x >= 0
             = alpha*x  otherwise
```

This gives a simple exact SMT encoding while preserving nonlinearity and avoiding dead activations.

Each Transformer block therefore has the same high-level pre-norm residual structure as GPT-style Transformers, but uses BandNorm in place of LayerNorm, sparsemax in place of softmax attention, and LeakyReLU in place of GELU.

The final residual is normalized with BandNorm and projected through the unembedding matrix. For task-specific verification, we usually restrict the output check to a small candidate token set `T`, such as the two quote tokens for quote closing or the two closing delimiter tokens for bracket type. This avoids encoding the full vocabulary while preserving the projected decision relevant to the task.

## Environment Setup

### Hardware Recommendation

The following was used to produce the GPT-2-scale scalability results in the appendix. This heavy setup isn't strictly required, but recommended for iteration speed.

- GPUs: `8x A100 80GB` (or `8x H100 80GB`)
- Per-GPU VRAM target: `40GB+` (prefer `80GB`)
- CPU RAM: `128GB` recommended (`64GB` minimum)
- Storage: `200GB+` fast SSD for dataset cache/checkpoints/logs
- CPU: `16+` cores preferred for data loading/tokenization

### Dependency Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or on RunPod:

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

python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"

pip install --upgrade pip
pip install transformers datasets evaluate tqdm matplotlib psutil pyyaml packaging huggingface_hub safetensors
pip install accelerate --no-deps
```

If your environment already provides PyTorch (for example, RunPod images), install only the project dependencies above and keep the preinstalled torch build.

## Quickstart

### Small End-to-End Verifiable Transformer

Train a minimal SMT-representable Transformer on two symbolic tasks:

1. **quote_close**: Match opening quotes (' or ")
2. **bracket_type**: Match opening brackets ([ or {)

**Key features:**
- Custom 32-token vocabulary (exhaustive finite domains)
- Only SMT-representable components (BandNorm + sparsemax + LeakyReLU)
- ~8K parameters (tractable for SMT encoding)
- Multitask model with task-specific circuit extraction

**Train the model:**
```bash
python scripts/small/train.py \
  --output_dir artifacts/small \
  --batch_size 64 \
  --max_steps 5000
```

Training stops automatically when both tasks achieve 100% candidate accuracy.

**Extract task-specific circuits:**
```bash
# Extract circuit for each task
for task in quote_close bracket_type; do
  python scripts/small/extract.py \
    --model_path artifacts/small/checkpoint-final \
    --task $task \
    --threshold_sweep \
    --output_dir artifacts/small_circuits/$task
done
```

**Export SMT-compatible weights:**
```bash
python scripts/small/extract_weights.py \
  --checkpoint artifacts/small/checkpoint-final \
  --output artifacts/small/smt_weights.json
```

This writes the model weights and SMT metadata in the format consumed by `scripts/smt/circuit.py`.

**Validate extracted circuits and run SMT verification:**
```bash
for task in quote_close bracket_type; do
  python scripts/small/verify.py \
    --task $task \
    --sanity_check \
    --checkpoint artifacts/small/checkpoint-final \
    --weights_path artifacts/small/smt_weights.json \
    --circuit_path artifacts/small_circuits/$task/circuit.json \
    --output_dir artifacts/small_circuits/$task/verification
done
```

The `--sanity_check` step is a cheap exhaustive PyTorch validation of the extracted circuit over the task domain. It confirms that `controlled_forward()` with the retained circuit edges still solves the task before any SMT properties run. Results are written to:

```text
artifacts/small_circuits/<task>/verification/verification_results.json
```

SMT verification uses branch-certified sparsemax, BandNorm, and LeakyReLU encodings by default. The verifier first checks that each traced branch certificate is satisfiable, then checks the requested property; an invalid certificate is reported as an error or unknown result, not as a proof.

By default, `verify.py` runs `functional_equivalence` and `edge_necessity`. To run every implemented property, pass the full property list:

```bash
for task in quote_close bracket_type; do
  python scripts/small/verify.py \
    --task $task \
    --sanity_check \
    --checkpoint artifacts/small/checkpoint-final \
    --weights_path artifacts/small/smt_weights.json \
    --circuit_path artifacts/small_circuits/$task/circuit.json \
    --output_dir artifacts/small_circuits/$task/verification \
    --properties functional_equivalence content_invariance edge_necessity continuous_robustness
done
```

**Properties to verify:**

| Task | Properties |
|------|-----------|
| quote_close | Functional equivalence, content invariance, edge necessity, continuous robustness |
| bracket_type | Functional equivalence, content invariance, edge necessity, continuous robustness |

**Results:**

| Task | Inputs | Circuit Edges | PyTorch Circuit Validation | Functional Equivalence | Content Invariance | Edge Necessity | Continuous Robustness |
|------|-------:|--------------:|-----------------------------|------------------------|--------------------|----------------|-----------------------|
| quote_close | 128 | 3 | PASSED | VERIFIED | VERIFIED | VERIFIED | VERIFIED |
| bracket_type | 128 | 6 | PASSED | VERIFIED | VERIFIED | VERIFIED | VERIFIED |

## Scalability Appendix

The main experiments target a small end-to-end verifiable Transformer where SMT verification is intended to be tractable.

We also trained GPT-2-scale models using the same verifiable operators to test optimization and scaling behavior. These larger models show that the architecture can train stably at GPT-2 scale, but naive SMT verification of extracted GPT-2-scale circuits is not currently tractable.

See [docs/SCALABILITY.md](docs/SCALABILITY.md) for:

- GPT-2 baseline reproduction
- Verifiable normalization and attention replacement experiments
- GPT-2-scale BandNorm + sparsemax + LeakyReLU results
- Circuit extraction results from the GPT-2-scale model
- Current SMT tractability limitations
- LayerNorm removal at GPT-2 scale (norm-free model selection for downstream verification)
- Verified distillation: program-replaced attention heads and zero-bilinear verified circuits ([docs/VERIFIED_DISTILLATION.md](docs/VERIFIED_DISTILLATION.md))
