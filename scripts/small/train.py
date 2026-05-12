#!/usr/bin/env python3
"""
Training script for small verifiable Transformer.

Trains one multitask model on three symbolic tasks:
- quote_close: Match opening quotes
- bracket_type: Match opening brackets
- add_mod_5: Addition modulo 5

The model uses SMT-representable components for formal verification.
"""

import argparse
import json
import os
import sys
from types import MethodType
from typing import Dict, List

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
# Model Components (copied from train_experiment.py)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Center
        x_mean = hidden_states.mean(dim=-1, keepdim=True)
        x = hidden_states - x_mean

        # Decompose into positive and negative parts
        pos = torch.relu(x)
        neg = torch.relu(-x)

        # Compute positive and negative L1 mass
        pos_mass = pos.sum(dim=-1, keepdim=True)
        neg_mass = neg.sum(dim=-1, keepdim=True)

        # Project each half separately
        pos_proj = self._project_nonnegative_l1_ball(pos, self.half_high)
        neg_proj = self._project_nonnegative_l1_ball(neg, self.half_high)

        # Expand from low to high if mass is too small
        pos_deficit = torch.relu(self.half_low - pos_mass)
        neg_deficit = torch.relu(self.half_low - neg_mass)

        pos_is_zero = (pos_mass < 1e-8)
        neg_is_zero = (neg_mass < 1e-8)

        pos_dir = torch.where(pos_is_zero.expand_as(pos), self.pos_fallback, pos)
        neg_dir = torch.where(neg_is_zero.expand_as(neg), self.neg_fallback, neg)

        pos_dir_norm = pos_dir / (pos_dir.sum(dim=-1, keepdim=True) + 1e-8)
        neg_dir_norm = neg_dir / (neg_dir.sum(dim=-1, keepdim=True) + 1e-8)

        pos_expanded = pos_proj + pos_deficit * pos_dir_norm
        neg_expanded = neg_proj + neg_deficit * neg_dir_norm

        # Recombine
        y = pos_expanded - neg_expanded

        # Apply learned affine
        if self.gamma is not None:
            y = y * self.gamma + self.beta

        return y


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation with fp32 numerical stability."""
    dtype = logits.dtype
    logits_fp32 = logits.float()

    # Sort descending
    sorted_logits, _ = torch.sort(logits_fp32, dim=dim, descending=True)

    # Compute cumulative sum and range
    cumsum = torch.cumsum(sorted_logits, dim=dim)
    size = sorted_logits.size(dim)
    arange = torch.arange(1, size + 1, device=logits.device, dtype=logits_fp32.dtype)

    # Compute support threshold
    support_mask = sorted_logits > (cumsum - 1.0) / arange
    k = support_mask.long().sum(dim=dim, keepdim=True)
    k = torch.clamp(k, min=1, max=size)

    tau = (cumsum.gather(dim, k - 1) - 1.0) / k.float()

    # Compute output
    z = logits_fp32 - tau
    return torch.clamp(z - tau, min=0.0).to(dtype)


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
    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

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

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

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

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
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
    )

    # Register custom activation if needed
    if config.activation_variant == "leaky_relu":
        from transformers.activations import ACT2FN
        ACT2FN["leaky_relu"] = LeakyReLU(config.leaky_relu_negative_slope)

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

    # Apply sparsemax attention
    if config.attn_variant == "sparsemax":
        for block in model.transformer.h:
            block.attn.forward = MethodType(gpt2_forward_with_sparsemax, block.attn)

    # Tie embeddings if requested
    if config.tie_embeddings:
        model.tie_weights()

    return model


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
    model.eval()

    examples = get_eval_dataset(task_name)
    candidates = list(vocab.get_candidates(vocab.TASK_NAME_TO_TOKEN[task_name]))

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


class TaskEvaluationCallback(TrainerCallback):
    """Callback to evaluate per-task metrics during training."""

    def __init__(self, eval_every_n_steps: int = 100):
        self.eval_every_n_steps = eval_every_n_steps
        self.metrics_history = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0:
            device = next(model.parameters()).device

            print(f"\n{'='*60}")
            print(f"Step {state.global_step} - Task Evaluation")
            print(f"{'='*60}")

            metrics = {}
            all_perfect = True

            for task_name in ["quote_close", "bracket_type", "add_mod_5"]:
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

            if all_perfect:
                print(f"\n{'='*60}")
                print("✓ All tasks achieved perfect candidate accuracy!")
                print(f"{'='*60}\n")
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
        default="artifacts/small_verifiable",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON (default: use default config)",
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
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Create callback
    eval_callback = TaskEvaluationCallback(eval_every_n_steps=args.eval_every)

    # Create trainer
    trainer = Trainer(
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
    for task_name in ["quote_close", "bracket_type", "add_mod_5"]:
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
