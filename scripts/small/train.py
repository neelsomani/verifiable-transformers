#!/usr/bin/env python3
"""
Training script for small verifiable Transformer.

Trains one multitask model on two symbolic tasks:
- quote_close: Match opening quotes
- bracket_type: Match opening brackets

The model uses SMT-representable components for formal verification.
"""

import argparse
import json
import os
import sys
from types import MethodType
from typing import Dict, List

# Keep this tiny custom model on one GPU by default. HuggingFace Trainer will
# otherwise use implicit DataParallel when multiple GPUs are visible, which can
# put patched GPT-2 modules and inputs on different CUDA devices.
if (
    "CUDA_VISIBLE_DEVICES" not in os.environ
    and "LOCAL_RANK" not in os.environ
    and "RANK" not in os.environ
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

# Import small verifiable components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.small import (
    SmallVerifiableDataset,
    get_eval_dataset,
    collate_fn,
    vocab,
    VOCAB_SIZE,
)
from scripts.small.config import SmallVerifiableConfig, get_default_config


# ============================================================================
# Model Components
# ============================================================================

class SignedL1BandNorm(nn.Module):
    """
    SMT-friendly normalization by signed L1 mass projection.
    """

    def __init__(
        self,
        hidden_size: int,
        l1_low_per_dim: float = 0.55,
        l1_high_per_dim: float = 1.05,
        affine: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.l1_low = float(l1_low_per_dim) * hidden_size
        self.l1_high = float(l1_high_per_dim) * hidden_size
        self.half_low = self.l1_low / 2.0
        self.half_high = self.l1_high / 2.0

        if affine:
            self.gamma = nn.Parameter(torch.ones(hidden_size))
            self.beta = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        # Used only for the pathological all-zero centered vector case.
        pos_fallback = torch.zeros(hidden_size)
        pos_fallback[0::2] = 1.0
        neg_fallback = 1.0 - pos_fallback
        self.register_buffer("pos_fallback", pos_fallback)
        self.register_buffer("neg_fallback", neg_fallback)

    def _project_nonnegative_l1_ball(self, y: torch.Tensor, radius: float) -> torch.Tensor:
        """Euclidean projection of nonnegative y onto {z >= 0, sum(z) <= radius}."""
        mass = y.sum(dim=-1, keepdim=True)
        needs_projection = mass > radius

        sorted_y, _ = torch.sort(y, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_y, dim=-1)

        d = y.size(-1)
        arange = torch.arange(1, d + 1, device=y.device, dtype=y.dtype)
        view_shape = [1] * y.dim()
        view_shape[-1] = d
        arange = arange.view(view_shape)

        candidates = (cumsum - radius) / arange
        # rho is the largest index k where sorted_y[k] > candidates[k]
        mask = sorted_y > candidates
        # Find the last True index along dim=-1
        rho = mask.long().sum(dim=-1, keepdim=True) - 1
        rho = torch.clamp(rho, min=0, max=d - 1)

        tau = candidates.gather(dim=-1, index=rho)
        z = torch.clamp(y - tau, min=0.0)

        # Use original if no projection needed
        result = torch.where(needs_projection, z, y)
        return result

    def _additive_lift(self, y: torch.Tensor, mass: torch.Tensor, target: float, fallback: torch.Tensor):
        """
        Additive lift over active coordinates to reach target mass.

        This is SMT-friendlier than proportional mass rescaling: the denominator
        is the active coordinate count rather than the input mass.
        """
        deficit = torch.relu(target - mass)

        # Active coordinates (nonzero)
        active = (y > 0).to(y.dtype)
        active_count = active.sum(dim=-1, keepdim=True)

        # Use fallback pattern if no active coordinates
        use_fallback = active_count < 1e-8
        fallback = fallback.to(dtype=y.dtype, device=y.device)
        fallback = fallback.view(*([1] * (y.dim() - 1)), -1).expand_as(y)

        active = torch.where(use_fallback.expand_as(active), fallback, active)
        active_count = active.sum(dim=-1, keepdim=True).clamp_min(1.0)

        # Distribute deficit equally over active coordinates
        delta = deficit / active_count
        return y + delta * active

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        # Center
        x_mean = hidden_states.mean(dim=-1, keepdim=True)
        x = hidden_states - x_mean

        # Decompose into positive and negative parts
        pos = torch.relu(x)
        neg = torch.relu(-x)

        # Project each half separately onto L1 ball
        pos_proj = self._project_nonnegative_l1_ball(pos, self.half_high)
        neg_proj = self._project_nonnegative_l1_ball(neg, self.half_high)

        # Additive lift if mass is too low
        pos_mass = pos_proj.sum(dim=-1, keepdim=True)
        neg_mass = neg_proj.sum(dim=-1, keepdim=True)

        pos_expanded = self._additive_lift(pos_proj, pos_mass, self.half_low, self.pos_fallback)
        neg_expanded = self._additive_lift(neg_proj, neg_mass, self.half_low, self.neg_fallback)

        # Recombine
        y = pos_expanded - neg_expanded

        # Recenter to match SMT encoder
        y = y - y.mean(dim=-1, keepdim=True)

        # Apply learned affine
        if self.gamma is not None:
            y = y * self.gamma.float() + self.beta.float()

        if not torch.isfinite(y).all():
            raise RuntimeError("SignedL1BandNorm produced non-finite values")

        return y.to(orig_dtype)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation with fp32 numerical stability."""
    dtype = logits.dtype
    z = logits.float()

    # Replace any existing infs before sort/cumsum.
    z = torch.where(torch.isfinite(z), z, torch.full_like(z, -1e4))

    # Sparsemax is translation-invariant; this prevents large-value cumsum issues.
    z = z - z.max(dim=dim, keepdim=True).values

    sorted_z, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = torch.cumsum(sorted_z, dim=dim)

    size = z.size(dim)
    r = torch.arange(1, size + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.dim()
    view_shape[dim] = size
    r = r.view(view_shape)

    support = 1 + r * sorted_z > cumsum
    k = support.long().sum(dim=dim, keepdim=True).clamp(min=1)

    tau = (cumsum.gather(dim, k - 1) - 1.0) / k.to(z.dtype)
    out = torch.clamp(z - tau, min=0.0)

    if not torch.isfinite(out).all():
        raise RuntimeError("sparsemax produced non-finite values")

    return out.to(dtype)


_sparsemax_call_count = 0


def sparsemax_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    """Sparsemax attention function for GPT2."""
    global _sparsemax_call_count
    _sparsemax_call_count += 1

    # Compute attention scores: Q @ K^T
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # Scale
    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Causal mask
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
    mask_value = torch.full(
        [],
        -1e4,
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    # Apply attention mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # SPARSEMAX instead of softmax (with fp32 stability)
    attn_weights = sparsemax(attn_weights, dim=-1)

    # Cast back to value dtype
    attn_weights = attn_weights.to(value.dtype)

    # Apply head mask if provided
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # Weighted sum of values
    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


def gpt2_forward_with_sparsemax(
    self,
    hidden_states,
    layer_past=None,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    use_cache=False,
    output_attentions=False,
):
    """GPT2Attention.forward replacement that uses sparsemax_attention_forward."""
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined."
            )
        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    # Transformers 4.49 no longer exposes _split_heads; reshape inline.
    query = query.view(*query.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(*key.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(*value.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    # Use sparsemax attention instead of eager/sdpa
    attn_output, attn_weights = sparsemax_attention_forward(
        self, query, key, value, attention_mask, head_mask
    )

    # Merge heads inline: [batch, heads, seq, head_dim] -> [batch, seq, embed_dim].
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*attn_output.shape[:-2], self.embed_dim)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


class LeakyReLU(nn.Module):
    """LeakyReLU activation function."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x >= 0, x, self.negative_slope * x)


# ============================================================================
# Model Creation
# ============================================================================


def create_small_model(config: SmallVerifiableConfig) -> GPT2LMHeadModel:
    """
    Create a small GPT2-style model with custom vocabulary and architecture.

    Args:
        config: SmallVerifiableConfig instance

    Returns:
        GPT2LMHeadModel with custom components
    """
    # Create GPT2Config from our config
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_seq_len,
        n_embd=config.d_model,
        n_layer=config.n_layers,
        n_head=config.n_heads,
        n_inner=config.d_mlp,
        activation_function=config.activation_variant,
        resid_pdrop=config.resid_pdrop,
        embd_pdrop=config.embd_pdrop,
        attn_pdrop=config.attn_pdrop,
        layer_norm_epsilon=1e-5,
        initializer_range=config.initializer_range,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        use_cache=False,
        tie_word_embeddings=config.tie_embeddings,
    )

    # Register custom activation if needed
    if config.activation_variant == "leaky_relu":
        from transformers.activations import ACT2FN

        class ConfiguredLeakyReLU(LeakyReLU):
            def __init__(self, **kwargs):
                super().__init__(config.leaky_relu_negative_slope)

        ACT2FN["leaky_relu"] = ConfiguredLeakyReLU

    # Create model
    model = GPT2LMHeadModel(gpt2_config)

    # Apply custom normalization
    if config.norm_variant == "signed_l1_band_norm":
        for block in model.transformer.h:
            # Replace ln_1 and ln_2
            block.ln_1 = SignedL1BandNorm(
                config.d_model,
                l1_low_per_dim=config.norm_l1_low_per_dim,
                l1_high_per_dim=config.norm_l1_high_per_dim,
            )
            block.ln_2 = SignedL1BandNorm(
                config.d_model,
                l1_low_per_dim=config.norm_l1_low_per_dim,
                l1_high_per_dim=config.norm_l1_high_per_dim,
            )
        # Replace final layer norm
        model.transformer.ln_f = SignedL1BandNorm(
            config.d_model,
            l1_low_per_dim=config.norm_l1_low_per_dim,
            l1_high_per_dim=config.norm_l1_high_per_dim,
        )
    elif config.norm_variant == "none":
        for block in model.transformer.h:
            block.ln_1 = nn.Identity()
            block.ln_2 = nn.Identity()
        model.transformer.ln_f = nn.Identity()
        model.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)

    # Apply sparsemax attention
    for block in model.transformer.h:
        block.attn._verifiable_attention_variant = config.attn_variant
        if config.attn_variant == "sparsemax":
            block.attn.forward = MethodType(gpt2_forward_with_sparsemax, block.attn)

    # Tie embeddings if requested
    if config.tie_embeddings:
        model.tie_weights()

    return model


def initialize_from_checkpoint(model: GPT2LMHeadModel, checkpoint: str) -> None:
    """Initialize a matched architecture, translating LayerNorm affine names.

    This is useful for controlled normalization comparisons: all shared weights
    can start from the same trained model while BandNorm's additional branch
    buffers retain their defined defaults.
    """
    from safetensors.torch import load_file

    weights_path = os.path.join(checkpoint, "model.safetensors")
    if os.path.exists(weights_path):
        source = load_file(weights_path)
    else:
        weights_path = os.path.join(checkpoint, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No model weights found in {checkpoint}")
        source = torch.load(weights_path, map_location="cpu")

    target = model.state_dict()
    translated = {}
    unhandled = []
    for source_name, value in source.items():
        candidates = [source_name]
        if source_name.endswith((".ln_1.weight", ".ln_2.weight", ".ln_f.weight")):
            candidates.append(source_name[: -len("weight")] + "gamma")
        elif source_name.endswith((".ln_1.bias", ".ln_2.bias", ".ln_f.bias")):
            candidates.append(source_name[: -len("bias")] + "beta")
        target_name = next(
            (
                name
                for name in candidates
                if name in target and target[name].shape == value.shape
            ),
            None,
        )
        if target_name is None:
            unhandled.append(source_name)
            continue
        target[target_name] = value.to(dtype=target[target_name].dtype)
        translated[source_name] = target_name
    if unhandled:
        raise RuntimeError(
            f"Initialization checkpoint has incompatible keys: {unhandled}"
        )
    model.load_state_dict(target, strict=True)
    print(
        f"Initialized {len(translated)} checkpoint tensors from {weights_path}"
    )


# ============================================================================
# Custom Trainer
# ============================================================================


class FinalTokenTrainer(Trainer):
    """
    Custom trainer that computes loss on the final prompt position.

    GPT2LMHeadModel shifts labels internally, which causes a mismatch:
    - With labels[-1] = target, the model trains position t-1 to predict position t
    - But we want to train the full prompt (position t) to predict the next token

    This trainer computes loss directly on final-position logits without label shifting.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch.nn.functional as F

        targets = inputs.pop("targets")
        input_ids = inputs["input_ids"]

        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]  # Final prompt position logits

        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits during training")

        loss = F.cross_entropy(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss during training")

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_task(
    model: GPT2LMHeadModel,
    task_name: str,
    device: torch.device,
) -> Dict:
    """
    Evaluate model on a specific task's exhaustive domain.

    Returns metrics:
    - accuracy: Full vocabulary accuracy
    - candidate_accuracy: Accuracy restricted to valid candidates
    - mean_candidate_margin: Average logit margin between correct and incorrect candidates
    - confusion: Confusion table for candidates
    """
    was_training = model.training
    model.eval()

    try:
        examples = get_eval_dataset(task_name)
        candidates = sorted(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task_name]))

        correct_full = 0
        correct_candidate = 0
        margins = []
        confusion = {c: {c2: 0 for c2 in candidates} for c in candidates}

        with torch.no_grad():
            for example in examples:
                input_ids = torch.tensor(example["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                target = example["target"]

                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last position logits

                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite logits during evaluation for task {task_name}")

                # Full vocab prediction
                pred_full = logits.argmax().item()
                if pred_full == target:
                    correct_full += 1

                # Candidate prediction
                candidate_logits = logits[candidates]
                pred_candidate_idx = candidate_logits.argmax().item()
                pred_candidate = candidates[pred_candidate_idx]

                if pred_candidate == target:
                    correct_candidate += 1

                # Confusion matrix
                if target in candidates:
                    confusion[target][pred_candidate] += 1

                # Margin: correct logit - max incorrect logit
                if target in candidates:
                    target_logit = logits[target]
                    incorrect_candidates = [c for c in candidates if c != target]
                    if incorrect_candidates:
                        max_incorrect_logit = logits[incorrect_candidates].max()
                        margin = (target_logit - max_incorrect_logit).item()
                        margins.append(margin)

        n = len(examples)
        accuracy = correct_full / n if n > 0 else 0.0
        candidate_accuracy = correct_candidate / n if n > 0 else 0.0
        mean_margin = sum(margins) / len(margins) if margins else 0.0

        return {
            "task": task_name,
            "n_examples": n,
            "accuracy": accuracy,
            "candidate_accuracy": candidate_accuracy,
            "mean_candidate_margin": mean_margin,
            "confusion": confusion,
        }
    finally:
        if was_training:
            model.train()


class TaskEvaluationCallback(TrainerCallback):
    """Callback to evaluate per-task metrics during training."""

    def __init__(self, eval_every_n_steps: int = 100, output_dir: str = None):
        self.eval_every_n_steps = eval_every_n_steps
        self.output_dir = output_dir
        self.metrics_history = []
        self.best_checkpoint_saved = False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0:
            device = next(model.parameters()).device

            print(f"\n{'='*60}")
            print(f"Step {state.global_step} - Task Evaluation")
            print(f"{'='*60}")

            metrics = {}
            all_perfect = True

            for task_name in ["quote_close", "bracket_type"]:
                task_metrics = evaluate_task(model, task_name, device)
                metrics[task_name] = task_metrics

                print(f"\n{task_name}:")
                print(f"  Accuracy: {task_metrics['accuracy']:.4f}")
                print(f"  Candidate Accuracy: {task_metrics['candidate_accuracy']:.4f}")
                print(f"  Mean Margin: {task_metrics['mean_candidate_margin']:.4f}")

                if task_metrics["candidate_accuracy"] < 1.0:
                    all_perfect = False

            metrics["step"] = state.global_step
            metrics["all_tasks_perfect"] = all_perfect
            self.metrics_history.append(metrics)

            if all_perfect and not self.best_checkpoint_saved:
                print(f"\n{'='*60}")
                print("✓ All tasks achieved perfect candidate accuracy!")
                print("Saving best checkpoint...")

                # Save best checkpoint
                if self.output_dir:
                    import os
                    from transformers import Trainer
                    best_dir = os.path.join(self.output_dir, "checkpoint-best")
                    model.save_pretrained(best_dir)
                    print(f"Best checkpoint saved to: {best_dir}")

                print(f"{'='*60}\n")
                self.best_checkpoint_saved = True
                control.should_training_stop = True

        return control


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train small verifiable Transformer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/small",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON (default: use default config)",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional matching checkpoint used only for initialization; "
            "LayerNorm weight/bias names translate to BandNorm gamma/beta."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.003,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=100,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--task_sampling",
        type=str,
        default="balanced",
        choices=["balanced", "proportional", "all"],
        help="Task sampling strategy",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load or create config
    if args.config:
        config = SmallVerifiableConfig.load(args.config)
    else:
        config = get_default_config()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config.save(os.path.join(args.output_dir, "config.json"))

    # Save vocabulary
    vocab.save_vocab(os.path.join(args.output_dir, "vocab.json"))

    print("Small Verifiable Transformer Training")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Model config:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Create model
    print("\nCreating model...")
    model = create_small_model(config)
    if args.init_checkpoint:
        initialize_from_checkpoint(model, args.init_checkpoint)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Verify sparsemax is active
    if config.attn_variant == "sparsemax":
        print("\nVerifying sparsemax patch...")
        global _sparsemax_call_count
        _sparsemax_call_count = 0

        dummy = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        with torch.no_grad():
            model(dummy)

        assert _sparsemax_call_count > 0, (
            f"Sparsemax not active! Got {_sparsemax_call_count} calls"
        )
        print(f"✓ Sparsemax verified: {_sparsemax_call_count} calls")

    # Create dataset
    print(f"\nCreating dataset (task_sampling={args.task_sampling})...")
    train_dataset = SmallVerifiableDataset(task_sampling=args.task_sampling)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.eval_every,
        save_steps=args.eval_every,
        save_total_limit=3,
        seed=args.seed,
        data_seed=args.seed,
        bf16=False,
        fp16=False,
        max_grad_norm=1.0,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Create callback
    eval_callback = TaskEvaluationCallback(
        eval_every_n_steps=args.eval_every,
        output_dir=args.output_dir,
    )

    # Create trainer (use FinalTokenTrainer for correct loss computation)
    trainer = FinalTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[eval_callback],
    )

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_dir = os.path.join(args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)

    # Save final metrics
    print("\nFinal evaluation...")
    device = next(model.parameters()).device
    final_metrics = {}
    for task_name in ["quote_close", "bracket_type"]:
        task_metrics = evaluate_task(model, task_name, device)
        final_metrics[task_name] = task_metrics
        print(f"\n{task_name}:")
        print(f"  Accuracy: {task_metrics['accuracy']:.4f}")
        print(f"  Candidate Accuracy: {task_metrics['candidate_accuracy']:.4f}")
        print(f"  Mean Margin: {task_metrics['mean_candidate_margin']:.4f}")

    # Save metrics history
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "final_metrics": final_metrics,
            "history": eval_callback.metrics_history,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to: {final_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
