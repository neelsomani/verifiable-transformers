# Verifiable Transformers

We design a Transformer variant whose forward pass is built from SMT-representable primitives, enabling exact formal reasoning for sufficiently small circuits and bounded domains. Once the full model is SMT-representable, we formally check properties like:

* whether the model is exactly equivalent to a given symbolic program on all sequences of length ≤ n,
* whether particular connections are necessary for the model’s behavior (edge necessity)
* whether the model obeys structural constraints (like local attention), and
* impossibility results on circuits (e.g., counts above k collapse to the same internal state)

The core idea is to make the architecture formally representable so that, when the hidden-state width and circuit size are tractable, these properties can be proven or refuted over an entire bounded domain rather than merely tested on examples.

There are two parts of the Transformer architecture that are traditionally difficult to SMT encode. First, the attention mechanism. Starting from the Deep Sets characterization of permutation-invariant functions, we re-derive the [structural form of attention](https://www.neelsomaniblog.com/p/a-minimal-route-to-transformer-attention) and show that it naturally decomposes into three learnable components: an aggregator ρ, a relevance scoring function u, and a content map v. We then characterize the verifiable subset of this space: all attention mechanisms whose computation can be expressed using only affine maps, piecewise-linear transformations, thresholding, and Top-k style selection - primitives that admit exact SMT encodings. Second, we apply a similar approach to the LayerNorm. (A third non-verifiable component, the GELU activation function, is easier to address.)

We show, end-to-end, that a Transformer can be trained and then formally analyzed at the circuit level. In this work, we focus on extracting circuits for Python syntax tasks:

* **Quote closing**: Distinguishing single quote `'` vs double quote `"` continuation
* **Bracket type**: Distinguishing `]` vs `}` for list vs dict closing

These tasks allow us to demonstrate proofs of bounded correctness, structural properties of the circuit, and impossibility/generalization limits. At GPT-2 scale, the architecture remains SMT-representable, but naive full-width SMT encoding is not yet tractable; formal proofs are intended for sufficiently small models or sufficiently sparse circuits. This work suggests a new direction for interpretable and certifiable sequence modeling.

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

| Property | Informal Description | Formal Definition |
|----------|---------------------|-------------------|
| **Functional Equivalence** | The model is equivalent to a specific symbolic program on all inputs of length ≤ n. For a code generation model trained on simple transformations (e.g. string manipulation), we can prove that for all inputs ≤ n, the model is equivalent to a reference implementation. If the property fails, the solver returns a concrete counterexample. | $\forall x \in \Sigma^{\leq n}, \quad y_M(x) = P(x)$ |
| **Edge Necessity** | Every retained edge is behaviorally necessary on the bounded domain: for each edge, there exists an input where removing that edge changes the circuit's projected output. This does not prove that ACDC found the globally smallest possible circuit; it proves that no retained edge is obviously redundant under the checked criterion. | $\forall e \in E(C), \quad \exists x \in \Sigma^{\leq n} \text{ such that } C(x) \neq (C \setminus e)(x)$ |
| **Structural Invariants** | The model obeys structural constraints (e.g. local attention). We can guarantee that sensitive information (e.g. earlier tokens containing secrets) cannot influence outputs beyond a fixed window. | $\forall x \in \Sigma^{\leq n}, \quad S(M, x)$ where $S$ encodes locality, sparsity, monotonicity, causality, etc. |
| **Impossibility Results** | The model produces identical outputs for two classes of inputs. We can prove that for all inputs ≤ n, the model cannot distinguish between two programs that differ only in variable renaming (e.g. renaming x to y throughout). This establishes that the model has learned a representation invariant to variable identity. | $\forall x, x' \in \Sigma^{\leq n}, \quad R(x, x') \Rightarrow \varphi_M(x) = \varphi_M(x')$ where $R$ identifies input pairs the model cannot distinguish |
| **Continuous Robustness** | The circuit's projected decision is stable under continuous perturbations to its internal state, such as quantization or bounded activation noise. For every input in a bounded domain and every perturbation $\eta$ to the circuit's final residual satisfying $\|\eta\|_\infty \le \epsilon$, the projected decision remains unchanged. | Let $r_E(x)$ be the final residual of circuit $C_E$. Let $G_T(r)$ be the token in candidate set $T$ with highest logit after final normalization and unembedding. We verify: $\forall x \in \Sigma^{\leq n},\ \forall \eta \in \mathbb{R}^{d},\ \|\eta\|_\infty \le \epsilon \Rightarrow G_T(r_E(x)+\eta)=G_T(r_E(x))$. If functional correctness is also verified, then $G_T(r_E(x)+\eta)=P(x)$. |

## Architecture

We use a GPT-style decoder-only Transformer, but replace the components that are hard to encode exactly in SMT.

### Signed L1 BandNorm

Standard LayerNorm is difficult to encode exactly because it requires division by a data-dependent standard deviation. We replace it with Signed L1 BandNorm, a projection-based normalization operator.

Given an input vector `x`, BandNorm:

1. centers the vector: `c = x - mean(x)`
2. splits positive and negative mass: `p = max(c, 0)` and `n = max(-c, 0)`
3. controls the L1 mass of each side:
   - if mass is too small, add a bounded lift over active coordinates
   - if mass is too large, project onto an L1 ball using thresholding
4. recombines the signed vector: `z = p' - n'`
5. applies a learned affine map: `output = gamma * z + beta`

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

GPT-2 normally uses GELU in the MLP block. GELU is not piecewise-linear, so we replace it with **LeakyReLU**:

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

Train a minimal SMT-representable Transformer on three symbolic tasks:

1. **quote_close**: Match opening quotes (' or ")
2. **bracket_type**: Match opening brackets ([ or {)
3. **add_mod_5**: Addition modulo 5 (0-4)

**Key features:**
- Custom 32-token vocabulary (exhaustive finite domains)
- Only SMT-representable components (BandNorm + sparsemax + LeakyReLU)
- ~11K parameters (tractable for SMT encoding)
- Multitask model with task-specific circuit extraction

**Train the model:**
```bash
python scripts/small/train.py \
  --output_dir artifacts/small \
  --batch_size 64 \
  --max_steps 5000
```

Training stops automatically when all three tasks achieve 100% candidate accuracy.

**Extract task-specific circuits:**
```bash
# Extract circuit for each task
for task in quote_close bracket_type add_mod_5; do
  python scripts/small/extract.py \
    --model_path artifacts/small/checkpoint-final \
    --task $task \
    --threshold_sweep \
    --output_dir artifacts/small_circuits/$task
done
```

**Properties to verify:**

| Task | Properties |
|------|-----------|
| quote_close | Functional correctness, content invariance, quote sensitivity, edge necessity, continuous robustness |
| bracket_type | Functional correctness, content invariance, delimiter sensitivity, edge necessity, continuous robustness |
| add_mod_5 | Functional correctness, commutativity, edge necessity, continuous robustness |

## Scalability Appendix

The main experiments target a small end-to-end verifiable Transformer where SMT verification is intended to be tractable.

We also trained GPT-2-scale models using the same verifiable operators to test optimization and scaling behavior. These larger models show that the architecture can train stably at GPT-2 scale, but naive SMT verification of extracted GPT-2-scale circuits is not currently tractable.

See [docs/SCALABILITY.md](docs/SCALABILITY.md) for:

- GPT-2 baseline reproduction
- Verifiable normalization and attention replacement experiments
- GPT-2-scale BandNorm + sparsemax + LeakyReLU results
- Circuit extraction results from the GPT-2-scale model
- Current SMT tractability limitations
