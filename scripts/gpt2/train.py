import argparse
import glob
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from itertools import chain
from types import MethodType
from typing import Optional
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
class DyTNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, clamp_value: float = 4.0):
        super().__init__()
        self.pre_scale = torch.nn.Parameter(torch.ones(hidden_size))
        self.pre_bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.post_scale = torch.nn.Parameter(torch.ones(hidden_size))
        self.post_bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.clamp_value = clamp_value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        transformed = hidden_states * self.pre_scale + self.pre_bias
        transformed = torch.clamp(transformed, min=-self.clamp_value, max=self.clamp_value)
        return transformed * self.post_scale + self.post_bias


class PiecewiseLinearNorm(torch.nn.Module):
    """
    SMT-friendly normalization surrogate using piecewise-linear operations.

    Instead of dividing by a data-dependent standard deviation, this module:
    1. Centers each token vector by subtracting the mean
    2. Computes mean absolute deviation (MAD) as a scale proxy
    3. Bucketizes MAD into fixed ranges and applies a constant multiplier per bucket
    4. Clamps the scaled values elementwise
    5. Applies learned affine parameters (gamma, beta)

    The entire forward pass is piecewise linear except for comparisons, max/min,
    abs, and mean reductions, making it suitable for SMT encoding.
    """
    def __init__(self, hidden_size: int, clamp_value: float = 4.0):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
        self.beta = torch.nn.Parameter(torch.zeros(hidden_size))
        self.clamp_value = clamp_value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Step 1: Center by subtracting per-token mean
        x_mean = hidden_states.mean(dim=-1, keepdim=True)
        x_centered = hidden_states - x_mean

        # Step 2: Compute mean absolute deviation (MAD) as scale proxy
        mad = torch.abs(x_centered).mean(dim=-1, keepdim=True)

        # Step 3: Bucketize MAD and apply constant multiplier per bucket
        # This avoids data-dependent division while approximating normalization
        scale = torch.where(
            mad < 0.25,
            torch.tensor(4.0, device=mad.device, dtype=mad.dtype),
            torch.where(
                mad < 0.5,
                torch.tensor(2.0, device=mad.device, dtype=mad.dtype),
                torch.where(
                    mad < 1.0,
                    torch.tensor(1.0, device=mad.device, dtype=mad.dtype),
                    torch.where(
                        mad < 2.0,
                        torch.tensor(0.5, device=mad.device, dtype=mad.dtype),
                        torch.tensor(0.25, device=mad.device, dtype=mad.dtype)
                    )
                )
            )
        )
        x_scaled = x_centered * scale

        # Step 4: Clamp elementwise to bounded range
        x_clamped = torch.clamp(x_scaled, min=-self.clamp_value, max=self.clamp_value)

        # Step 5: Apply learned affine transformation
        output = x_clamped * self.gamma + self.beta

        return output


def soft_clamp(x, c=2.0, leak=0.1):
    """
    Leaky piecewise-linear clamp (unbounded).

    For |x| <= c: returns x (identity)
    For |x| > c: returns c + leak * (x - c) with appropriate sign

    This is fully SMT-encodable and provides soft saturation instead of
    hard clipping, which improves gradient flow but is unbounded.
    """
    return torch.where(
        x < -c,
        -c + leak * (x + c),
        torch.where(
            x > c,
            c + leak * (x - c),
            x,
        ),
    )


def bounded_pwl_clamp(x):
    """
    Bounded piecewise-linear clamp with three regions (SMT-encodable).

    Structure:
    - x < -3.0: flat at -2.3 (slope 0, bounded)
    - -3.0 <= x < -2.0: linear slope 0.3
    - -2.0 <= x <= 2.0: identity (slope 1)
    - 2.0 < x <= 3.0: linear slope 0.3
    - x > 3.0: flat at 2.3 (slope 0, bounded)

    This provides:
    - Bounded activations (saturates at ±2.3)
    - Nonzero gradients in most regions
    - No explosion
    - Fully SMT-encodable (pure piecewise-linear, no division)
    """
    return torch.where(
        x < -3.0,
        torch.tensor(-2.3, device=x.device, dtype=x.dtype),
        torch.where(
            x < -2.0,
            -2.0 + 0.3 * (x + 2.0),
            torch.where(
                x <= 2.0,
                x,
                torch.where(
                    x <= 3.0,
                    2.0 + 0.3 * (x - 2.0),
                    torch.tensor(2.3, device=x.device, dtype=x.dtype),
                ),
            ),
        ),
    )


class VerifiablePWLNorm(torch.nn.Module):
    """
    Fully verifiable normalization with simple center-clamp-scale.

    This module performs:
    1. Center: subtract per-token mean
    2. Clamp: elementwise clamp (soft leaky or bounded PWL)
    3. Scale: multiply by fixed constant 0.5
    4. Bias: add learned per-dimension bias

    The forward pass uses only:
    - mean reduction
    - affine operations
    - piecewise-linear clamp

    This is fully SMT-encodable and provides bounded activation control
    for stable residual optimization.
    """
    def __init__(self, hidden_size: int, clamp_value: float = 2.0, clamp_type: str = "soft"):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(hidden_size))
        self.clamp_value = clamp_value
        self.clamp_type = clamp_type
        self.scale = 0.5
        # Diagnostics tracking (optional, enabled by callback)
        self.track_stats = False
        self.last_mean = None
        self.last_clamp_fraction = None
        self.last_transition_fraction = None  # For bounded: in slope region
        self.last_saturation_fraction = None  # For bounded: in flat region

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Center by subtracting per-token mean
        x_mean = hidden_states.mean(dim=-1, keepdim=True)
        x = hidden_states - x_mean

        # 2. Apply clamp based on type
        x_before_clamp = x
        if self.clamp_type == "bounded":
            # Bounded PWL clamp (bounded activations + nonzero gradients)
            x = bounded_pwl_clamp(x)
        elif self.clamp_type == "soft":
            # Soft leaky clamp (unbounded, but better gradients than hard clamp)
            x = soft_clamp(x, c=self.clamp_value, leak=0.1)
        else:
            # Hard clamp (bounded but zero gradients outside)
            x = torch.clamp(x, min=-self.clamp_value, max=self.clamp_value)

        # 3. Apply fixed scale
        x = x * self.scale

        # 4. Add learned bias
        output = x + self.beta

        # Track statistics if enabled
        if self.track_stats:
            with torch.no_grad():
                self.last_mean = x_mean.abs().mean().item()
                abs_x = torch.abs(x_before_clamp)

                if self.clamp_type == "bounded":
                    # For bounded PWL: track transition (2-3) and saturation (>3) separately
                    in_transition = ((abs_x > 2.0) & (abs_x <= 3.0)).float()
                    in_saturation = (abs_x > 3.0).float()
                    self.last_transition_fraction = in_transition.mean().item()
                    self.last_saturation_fraction = in_saturation.mean().item()
                    # Overall nonlinear fraction (entered slope or flat region)
                    self.last_clamp_fraction = (abs_x > 2.0).float().mean().item()
                else:
                    # For soft/hard: fraction beyond clamp threshold
                    self.last_clamp_fraction = (abs_x > self.clamp_value).float().mean().item()
                    self.last_transition_fraction = None
                    self.last_saturation_fraction = None

        return output


class SignedL1BandNorm(torch.nn.Module):
    """
    SMT-friendly normalization by signed L1 mass projection.

    Properties:
    - centers each token vector
    - preserves zero-mean structure by separately controlling positive/negative mass
    - avoids data-dependent multiplication
    - uses only affine ops, ReLU/max, abs/sign-like decomposition, sort/top-k, thresholding
    - no elementwise saturation of all large coordinates
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
            self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
            self.beta = torch.nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        # Used only for the pathological all-zero centered vector case.
        pos_fallback = torch.zeros(hidden_size)
        pos_fallback[0::2] = 1.0
        neg_fallback = 1.0 - pos_fallback
        self.register_buffer("pos_fallback", pos_fallback)
        self.register_buffer("neg_fallback", neg_fallback)

        # Optional diagnostics
        self.track_stats = False
        self.last_low_fraction = None
        self.last_high_fraction = None
        self.last_mean_abs = None
        self.last_mass_mean = None

    def _project_nonnegative_l1_ball(self, y: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Euclidean projection of nonnegative y onto {z >= 0, sum(z) <= radius}.

        If sum(y) <= radius, returns y.
        Else returns max(y - tau, 0), with tau selected by sorting.
        """
        mass = y.sum(dim=-1, keepdim=True)
        needs_projection = mass > radius

        sorted_y, _ = torch.sort(y, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_y, dim=-1)

        d = y.size(-1)
        arange = torch.arange(1, d + 1, device=y.device, dtype=y.dtype)
        view_shape = [1] * y.dim()
        view_shape[-1] = d
        arange = arange.view(view_shape)

        # Support condition for projection threshold.
        support = sorted_y - (cumsum - radius) / arange > 0
        k = support.to(torch.int64).sum(dim=-1, keepdim=True).clamp(min=1)

        tau = (torch.gather(cumsum, dim=-1, index=k - 1) - radius) / k.to(y.dtype)
        projected = torch.clamp(y - tau, min=0.0)

        return torch.where(needs_projection, projected, y)

    def _lift_nonnegative_l1_mass(
        self,
        y: torch.Tensor,
        radius: float,
        fallback_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        If sum(y) < radius, add equal mass to active coordinates.

        This is additive, not multiplicative:
            y_i <- y_i + delta for active i

        That keeps the map piecewise-affine.
        """
        mass = y.sum(dim=-1, keepdim=True)
        needs_lift = mass < radius

        active = y > 0
        active_count = active.to(torch.int64).sum(dim=-1, keepdim=True)

        fallback = fallback_mask.view(*([1] * (y.dim() - 1)), y.size(-1)).to(dtype=torch.bool)
        fallback = fallback.expand_as(active)

        active = torch.where(active_count > 0, active, fallback)
        active_f = active.to(y.dtype)
        active_count = active_f.sum(dim=-1, keepdim=True).clamp(min=1.0)

        delta = torch.clamp(radius - mass, min=0.0) / active_count
        lifted = y + active_f * delta

        return torch.where(needs_lift, lifted, y)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Center.
        mean = hidden_states.mean(dim=-1, keepdim=True)
        c = hidden_states - mean

        # 2. Split into positive and negative masses.
        p = torch.clamp(c, min=0.0)
        n = torch.clamp(-c, min=0.0)

        p_mass = p.sum(dim=-1, keepdim=True)
        n_mass = n.sum(dim=-1, keepdim=True)

        # 3. Low-mass additive lift.
        p = self._lift_nonnegative_l1_mass(p, self.half_low, self.pos_fallback)
        n = self._lift_nonnegative_l1_mass(n, self.half_low, self.neg_fallback)

        # 4. High-mass projection.
        p = self._project_nonnegative_l1_ball(p, self.half_high)
        n = self._project_nonnegative_l1_ball(n, self.half_high)

        # 5. Recombine. Sum is approximately zero because masses are controlled symmetrically.
        z = p - n

        # Optional exact recentering. This is affine and SMT-safe.
        z = z - z.mean(dim=-1, keepdim=True)

        if self.gamma is not None:
            z = z * self.gamma + self.beta

        if self.track_stats:
            with torch.no_grad():
                mass = torch.abs(c).sum(dim=-1)
                self.last_mass_mean = mass.mean().item()
                self.last_mean_abs = mean.abs().mean().item()
                self.last_low_fraction = (mass < self.l1_low).float().mean().item()
                self.last_high_fraction = (mass > self.l1_high).float().mean().item()

        return z


class ResidualGate(torch.nn.Module):
    """
    SMT-friendly learned scalar gate for residual connections.

    Each gate is a single scalar parameter alpha, clamped to [0, max_value] during
    the forward pass. This provides adaptive control over residual branch
    strength while remaining exactly verifiable.
    """
    def __init__(self, init_value: float = 0.1, max_value: float = 0.5):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(init_value))
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp alpha to [0, max_value] range (SMT-friendly, unlike sigmoid)
        gate = torch.clamp(self.alpha, min=0.0, max=self.max_value)
        return gate * x


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation with fp32 numerical stability.

    Upcasts to fp32 for sort/cumsum/support computation to avoid
    bf16 rounding issues around exact thresholds.
    """
    z = logits.float()
    sorted_logits, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = torch.cumsum(sorted_logits, dim=dim)

    range_size = z.size(dim)
    view_shape = [1] * z.dim()
    view_shape[dim] = range_size
    range_tensor = torch.arange(1, range_size + 1, device=z.device, dtype=z.dtype).view(view_shape)

    support = 1 + range_tensor * sorted_logits > cumsum
    k = support.to(torch.int64).sum(dim=dim, keepdim=True).clamp(min=1)

    tau = (torch.gather(cumsum, dim=dim, index=k - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


# Global counter for sparsemax verification
_sparsemax_call_count = 0


def sparsemax_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    """
    Sparsemax attention function for transformers 4.49 GPT2.

    This replaces eager_attention_forward with sparsemax instead of softmax.

    Args:
        module: The attention module
        query: [batch, num_heads, seq, head_dim]
        key: [batch, num_heads, seq, head_dim]
        value: [batch, num_heads, seq, head_dim]
        attention_mask: Optional 4D mask [batch, 1, 1, seq] or None
        head_mask: Optional head mask

    Returns:
        (attn_output, attn_weights) where:
            attn_output: [batch, num_heads, seq, head_dim]
            attn_weights: [batch, num_heads, seq, seq]
    """
    global _sparsemax_call_count
    _sparsemax_call_count += 1

    # Compute attention scores: Q @ K^T
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # Scale by sqrt(d_k)
    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Apply causal mask for self-attention (GPT2 stores this in module.bias)
    if not module.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    # Apply attention_mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # SPARSEMAX instead of softmax (with fp32 stability)
    attn_weights = sparsemax(attn_weights, dim=-1)

    # Cast back to value dtype
    attn_weights = attn_weights.to(value.dtype)

    # Apply dropout
    attn_weights = module.attn_dropout(attn_weights)

    # Apply head_mask if provided
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # Compute output: weights @ V
    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


def gpt2_forward_with_sparsemax(
    self,
    hidden_states: Optional[tuple[torch.FloatTensor]],
    layer_past: Optional[tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    """
    GPT2Attention.forward replacement that uses sparsemax_attention_forward.

    This is based on the transformers 4.49 GPT2Attention.forward but replaces
    the attention function with sparsemax.
    """
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    # Reshape to multi-head: [batch, seq, embed_dim] -> [batch, num_heads, seq, head_dim]
    # In 4.49, this is done inline without _split_heads method
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

    # Merge heads: [batch, num_heads, seq, head_dim] -> [batch, seq, embed_dim]
    # In 4.49, this is done inline without _merge_heads method
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*attn_output.shape[:-2], self.embed_dim)

    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def apply_model_variants(model, norm_variant: str, attn_variant: str, activation_variant: str = "gelu") -> None:
    if norm_variant == "none":
        hidden_size = model.config.n_embd
        for block in model.transformer.h:
            block.ln_1 = torch.nn.Identity()
            block.ln_2 = torch.nn.Identity()
        model.transformer.ln_f = torch.nn.Identity()
        if model.lm_head.bias is None:
            replacement = torch.nn.Linear(
                hidden_size,
                model.config.vocab_size,
                bias=True,
                device=model.lm_head.weight.device,
                dtype=model.lm_head.weight.dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(model.lm_head.weight)
                replacement.bias.zero_()
            model.lm_head = replacement
        model.config.tie_word_embeddings = False
    elif norm_variant == "dyt":
        hidden_size = model.config.n_embd
        for block in model.transformer.h:
            block.ln_1 = DyTNorm(hidden_size=hidden_size)
            block.ln_2 = DyTNorm(hidden_size=hidden_size)
        model.transformer.ln_f = DyTNorm(hidden_size=hidden_size)
    elif norm_variant == "verifiable_pwl_norm_v1":
        hidden_size = model.config.n_embd
        for block in model.transformer.h:
            block.ln_1 = PiecewiseLinearNorm(hidden_size=hidden_size)
            block.ln_2 = PiecewiseLinearNorm(hidden_size=hidden_size)
        model.transformer.ln_f = PiecewiseLinearNorm(hidden_size=hidden_size)
    elif norm_variant == "verifiable_pwl_norm_v2":
        hidden_size = model.config.n_embd
        for block in model.transformer.h:
            block.ln_1 = VerifiablePWLNorm(hidden_size=hidden_size, clamp_type="soft")
            block.ln_2 = VerifiablePWLNorm(hidden_size=hidden_size, clamp_type="soft")
            # Patch forward to use fixed residual scaling
            _patch_block_with_residual_scaling(block)
        model.transformer.ln_f = VerifiablePWLNorm(hidden_size=hidden_size, clamp_type="soft")
    elif norm_variant == "verifiable_pwl_norm_v3":
        hidden_size = model.config.n_embd
        for block in model.transformer.h:
            block.ln_1 = VerifiablePWLNorm(hidden_size=hidden_size, clamp_type="bounded")
            block.ln_2 = VerifiablePWLNorm(hidden_size=hidden_size, clamp_type="bounded")
            # Patch forward to use fixed residual scaling
            _patch_block_with_residual_scaling(block)
        model.transformer.ln_f = VerifiablePWLNorm(hidden_size=hidden_size, clamp_type="bounded")
    elif norm_variant == "signed_l1_band_norm":
        hidden_size = model.config.n_embd
        for block in model.transformer.h:
            block.ln_1 = SignedL1BandNorm(
                hidden_size=hidden_size,
                l1_low_per_dim=0.55,
                l1_high_per_dim=1.05,
            )
            block.ln_2 = SignedL1BandNorm(
                hidden_size=hidden_size,
                l1_low_per_dim=0.55,
                l1_high_per_dim=1.05,
            )
            # DO NOT patch block with residual scaling for this variant
        model.transformer.ln_f = SignedL1BandNorm(
            hidden_size=hidden_size,
            l1_low_per_dim=0.55,
            l1_high_per_dim=1.05,
        )

    # Monkey-patch attention for sparsemax (transformers 4.49 approach)
    for block in model.transformer.h:
        block.attn._verifiable_attention_variant = attn_variant
        if attn_variant == "sparsemax":
            block.attn.forward = MethodType(gpt2_forward_with_sparsemax, block.attn)

    # Note: activation_variant is handled via config.activation_function before model creation
    # The model is already initialized with the correct activation from ACT2FN


def _patch_block_with_residual_scaling(block) -> None:
    """
    Monkey-patch GPT2Block forward to use additive residual updates
    followed by mild contraction of the resulting state.

    This preserves residual-branch semantics (branch outputs are deltas),
    while still making the overall state update contractive enough to
    reduce late-stage accumulation.
    """
    ATTN_RES_SCALE = 0.25
    MLP_RES_SCALE = 0.25
    STATE_SHRINK = 0.98

    def forward_with_damped_additive_residual(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        # Residual correction, then mild contraction of the new state
        hidden_states = residual + ATTN_RES_SCALE * attn_output
        hidden_states = STATE_SHRINK * bounded_pwl_clamp(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        # Residual correction, then mild contraction of the new state
        hidden_states = residual + MLP_RES_SCALE * feed_forward_hidden_states
        hidden_states = STATE_SHRINK * bounded_pwl_clamp(hidden_states)

        outputs = (hidden_states,) + outputs
        return outputs

    block.forward = MethodType(forward_with_damped_additive_residual, block)


class EvalLossThresholdStopCallback(TrainerCallback):
    def __init__(self, target_eval_loss: float):
        self.target_eval_loss = target_eval_loss

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control
        if eval_loss <= self.target_eval_loss:
            print(
                f"Early stopping triggered: eval_loss={eval_loss:.4f} <= target={self.target_eval_loss:.4f}"
            )
            control.should_training_stop = True
        return control


class CatastrophicDivergenceStopCallback(TrainerCallback):
    def __init__(
        self,
        train_loss_threshold: float | None,
        eval_loss_threshold: float | None,
        stop_on_inf_grad_norm: bool,
        min_step: int,
        train_increase_delta: float | None,
        eval_increase_delta: float | None,
        increase_patience: int,
    ):
        self.train_loss_threshold = train_loss_threshold
        self.eval_loss_threshold = eval_loss_threshold
        self.stop_on_inf_grad_norm = stop_on_inf_grad_norm
        self.min_step = min_step
        self.train_increase_delta = train_increase_delta
        self.eval_increase_delta = eval_increase_delta
        self.increase_patience = increase_patience
        self.best_train_loss = None
        self.best_eval_loss = None
        self.train_increase_count = 0
        self.eval_increase_count = 0

    def _trip(self, control: TrainerControl, message: str):
        print(f"Catastrophic divergence guard triggered: {message}")
        control.should_training_stop = True
        control.should_save = True
        return control

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or state.global_step < self.min_step:
            return control

        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")

        if self.train_loss_threshold is not None and loss is not None:
            if float(loss) >= float(self.train_loss_threshold):
                return self._trip(
                    control,
                    f"train_loss={float(loss):.4f} >= threshold={float(self.train_loss_threshold):.4f}",
                )

        if self.train_increase_delta is not None and loss is not None:
            loss_value = float(loss)
            if self.best_train_loss is None or loss_value < self.best_train_loss:
                self.best_train_loss = loss_value
                self.train_increase_count = 0
            elif loss_value >= self.best_train_loss + float(self.train_increase_delta):
                self.train_increase_count += 1
                if self.train_increase_count >= self.increase_patience:
                    return self._trip(
                        control,
                        (
                            f"train_loss increased to {loss_value:.4f} from best {self.best_train_loss:.4f} "
                            f"(delta>={float(self.train_increase_delta):.4f}) for {self.train_increase_count} logs"
                        ),
                    )
            else:
                self.train_increase_count = 0

        if self.stop_on_inf_grad_norm and grad_norm is not None:
            grad_norm_value = float(grad_norm)
            if math.isinf(grad_norm_value) or math.isnan(grad_norm_value):
                return self._trip(control, f"grad_norm={grad_norm_value}")

        return control

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None or state.global_step < self.min_step:
            return control

        eval_loss = metrics.get("eval_loss")
        if self.eval_loss_threshold is not None and eval_loss is not None:
            if float(eval_loss) >= float(self.eval_loss_threshold):
                return self._trip(
                    control,
                    f"eval_loss={float(eval_loss):.4f} >= threshold={float(self.eval_loss_threshold):.4f}",
                )

        if self.eval_increase_delta is not None and eval_loss is not None:
            eval_loss_value = float(eval_loss)
            if self.best_eval_loss is None or eval_loss_value < self.best_eval_loss:
                self.best_eval_loss = eval_loss_value
                self.eval_increase_count = 0
            elif eval_loss_value >= self.best_eval_loss + float(self.eval_increase_delta):
                self.eval_increase_count += 1
                if self.eval_increase_count >= self.increase_patience:
                    return self._trip(
                        control,
                        (
                            f"eval_loss increased to {eval_loss_value:.4f} from best {self.best_eval_loss:.4f} "
                            f"(delta>={float(self.eval_increase_delta):.4f}) for {self.eval_increase_count} evals"
                        ),
                    )
            else:
                self.eval_increase_count = 0

        return control


def evaluate_causal_lm_perplexity(model, input_ids: torch.Tensor, block_size: int, stride: int):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    nlls = []
    seq_len = input_ids.size(1)
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + block_size, seq_len)
        target_len = end - prev_end
        ids = input_ids[:, begin:end]
        labels = ids.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(ids, labels=labels)
            neg_log_likelihood = outputs.loss * target_len
        nlls.append(neg_log_likelihood)
        prev_end = end

        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    avg_nll = (total_nll / seq_len).item()
    ppl = math.exp(avg_nll)
    return avg_nll, ppl, int(seq_len)


def load_wikitext_input_ids(tokenizer, split: str, max_samples: int | None):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    joined_text = "\n\n".join(dataset["text"])
    return tokenizer(joined_text, return_tensors="pt").input_ids


class WikiTextEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        output_dir: str,
        split: str,
        block_size: int,
        stride: int,
        max_samples: int | None,
        eval_every_n_evals: int,
        target_ppl: float | None,
    ):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.split = split
        self.block_size = block_size
        self.stride = stride
        self.max_samples = max_samples
        self.eval_every_n_evals = eval_every_n_evals
        self.target_ppl = target_ppl
        self.eval_counter = 0
        self.wikitext_input_ids = None
        self.target_reached_marker = os.path.join(output_dir, "wikitext_target_reached.json")

    def _evaluate_and_save(self, model, step: int):
        if self.wikitext_input_ids is None:
            self.wikitext_input_ids = load_wikitext_input_ids(
                self.tokenizer,
                split=self.split,
                max_samples=self.max_samples,
            )

        loss, ppl, seq_len = evaluate_causal_lm_perplexity(
            model,
            input_ids=self.wikitext_input_ids,
            block_size=self.block_size,
            stride=self.stride,
        )
        metrics = {
            "dataset": "wikitext-103-raw-v1",
            "split": self.split,
            "step": int(step),
            "loss": loss,
            "perplexity": ppl,
            "seq_len": seq_len,
            "max_samples": self.max_samples,
        }
        step_path = os.path.join(self.output_dir, f"wikitext_eval_step_{int(step)}.json")
        latest_path = os.path.join(self.output_dir, "wikitext_eval_latest.json")
        with open(step_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(
            f"WikiText eval at step {int(step)}: loss={loss:.4f}, perplexity={ppl:.4f}"
        )
        return metrics

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, metrics=None, **kwargs):
        self.eval_counter += 1

        if os.path.isfile(self.target_reached_marker):
            control.should_training_stop = True
            return control

        if self.eval_every_n_evals <= 0 or self.eval_counter % self.eval_every_n_evals != 0:
            return control

        if state.is_world_process_zero:
            wikitext_metrics = self._evaluate_and_save(model=model, step=state.global_step)
            if metrics is not None:
                metrics["eval_wikitext_loss"] = wikitext_metrics["loss"]
                metrics["eval_wikitext_perplexity"] = wikitext_metrics["perplexity"]
            if self.target_ppl is not None and wikitext_metrics["perplexity"] <= self.target_ppl:
                with open(self.target_reached_marker, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "target_ppl": self.target_ppl,
                            "achieved_ppl": wikitext_metrics["perplexity"],
                            "step": int(state.global_step),
                        },
                        handle,
                        indent=2,
                    )
                print(
                    f"Early stopping triggered: WikiText perplexity={wikitext_metrics['perplexity']:.4f} <= "
                    f"target={self.target_ppl:.4f}"
                )

        if os.path.isfile(self.target_reached_marker):
            control.should_training_stop = True
        return control


class VerifiableNormDiagnosticsCallback(TrainerCallback):
    """
    Lightweight diagnostics logging for VerifiablePWLNorm and SignedL1BandNorm modules.

    Collects and saves summary statistics during evaluation:
    - VerifiablePWLNorm:
      - mean absolute value of per-token means
      - fraction of activations beyond threshold
      - for bounded variant only:
        - fraction in transition region (2.0 < |x| <= 3.0, slope 0.3)
        - fraction in saturation region (|x| > 3.0, flat)
    - SignedL1BandNorm:
      - mean L1 mass
      - fraction of tokens with mass below lower band
      - fraction of tokens with mass above upper band
    """
    def __init__(self, output_dir: str, enabled: bool):
        self.output_dir = output_dir
        self.enabled = enabled

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, eval_dataloader=None, **kwargs):
        if not self.enabled or model is None or not state.is_world_process_zero:
            return control

        # Collect all VerifiablePWLNorm and SignedL1BandNorm modules
        pwl_norm_modules = []
        l1_band_norm_modules = []
        for module in model.modules():
            if isinstance(module, VerifiablePWLNorm):
                pwl_norm_modules.append(module)
            elif isinstance(module, SignedL1BandNorm):
                l1_band_norm_modules.append(module)

        if not pwl_norm_modules and not l1_band_norm_modules:
            return control

        # Enable stats tracking
        for module in pwl_norm_modules:
            module.track_stats = True
        for module in l1_band_norm_modules:
            module.track_stats = True

        # Run a forward pass on a small batch to collect stats
        try:
            with torch.no_grad():
                # Get one batch from eval dataloader
                if eval_dataloader is None:
                    return control
                batch = next(iter(eval_dataloader))
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model(**batch)

            stats = {"step": int(state.global_step)}

            # Collect VerifiablePWLNorm statistics
            if pwl_norm_modules:
                mean_values = []
                clamp_fractions = []
                transition_fractions = []
                saturation_fractions = []

                for module in pwl_norm_modules:
                    if module.last_mean is not None:
                        mean_values.append(module.last_mean)
                    if module.last_clamp_fraction is not None:
                        clamp_fractions.append(module.last_clamp_fraction)
                    if module.last_transition_fraction is not None:
                        transition_fractions.append(module.last_transition_fraction)
                    if module.last_saturation_fraction is not None:
                        saturation_fractions.append(module.last_saturation_fraction)

                if mean_values:
                    stats["mean_abs_mean"] = float(sum(mean_values) / len(mean_values))
                    stats["mean_clamp_fraction"] = float(sum(clamp_fractions) / len(clamp_fractions))

                    # Add bounded-specific stats if available
                    if transition_fractions:
                        stats["mean_transition_fraction"] = float(sum(transition_fractions) / len(transition_fractions))
                    if saturation_fractions:
                        stats["mean_saturation_fraction"] = float(sum(saturation_fractions) / len(saturation_fractions))

            # Collect SignedL1BandNorm statistics
            if l1_band_norm_modules:
                low_fractions = []
                high_fractions = []
                mass_means = []
                mean_abs_values = []

                for module in l1_band_norm_modules:
                    if module.last_low_fraction is not None:
                        low_fractions.append(module.last_low_fraction)
                    if module.last_high_fraction is not None:
                        high_fractions.append(module.last_high_fraction)
                    if module.last_mass_mean is not None:
                        mass_means.append(module.last_mass_mean)
                    if module.last_mean_abs is not None:
                        mean_abs_values.append(module.last_mean_abs)

                if low_fractions:
                    stats["mean_low_fraction"] = float(sum(low_fractions) / len(low_fractions))
                if high_fractions:
                    stats["mean_high_fraction"] = float(sum(high_fractions) / len(high_fractions))
                if mass_means:
                    stats["mean_l1_mass"] = float(sum(mass_means) / len(mass_means))
                if mean_abs_values:
                    stats["mean_abs_mean_l1"] = float(sum(mean_abs_values) / len(mean_abs_values))

            if len(stats) > 1:  # More than just "step"
                stats_path = os.path.join(self.output_dir, "verifiable_norm_stats_latest.json")
                with open(stats_path, "w", encoding="utf-8") as handle:
                    json.dump(stats, handle, indent=2)

        except Exception as e:
            print(f"Warning: VerifiableNormDiagnosticsCallback failed: {e}")

        finally:
            # Disable stats tracking
            for module in pwl_norm_modules:
                module.track_stats = False
            for module in l1_band_norm_modules:
                module.track_stats = False

        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 experiments (baseline and architecture variants)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gpt2_baseline.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/gpt2-baseline",
        help="Training output directory.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset mode for OWT.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap for local smoke tests.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=10000,
        help="Optional eval cap.",
    )
    parser.add_argument(
        "--early_stop_eval_loss",
        type=float,
        default=None,
        help="Stop training once eval_loss is <= this threshold.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path to resume from.",
    )
    parser.add_argument(
        "--reset_optimizer_on_resume",
        action="store_true",
        help="Resume from checkpoint weights while resetting optimizer/scheduler/scaler/rng states.",
    )
    parser.add_argument(
        "--disable_auto_resume",
        action="store_true",
        help="Disable automatic resume from latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps from config.",
    )
    parser.add_argument(
        "--processed_dataset_dir",
        type=str,
        default=None,
        help="Path to save/load preprocessed tokenized dataset.",
    )
    parser.add_argument(
        "--preprocessing_num_proc",
        type=int,
        default=None,
        help="Number of CPU processes for dataset.map preprocessing.",
    )
    parser.add_argument(
        "--evaluate_wikitext_at_end",
        action="store_true",
        help="Run WikiText-103 evaluation after training completes.",
    )
    parser.add_argument(
        "--use_wikitext_as_dev",
        action="store_true",
        help="Use periodic WikiText eval as an opt-in dev criterion mode.",
    )
    parser.add_argument(
        "--target_wikitext_ppl",
        type=float,
        default=None,
        help="Optional target perplexity for early stopping based on WikiText-103.",
    )
    parser.add_argument(
        "--wikitext_eval_every_n_evals",
        type=int,
        default=0,
        help="Run WikiText eval every N Trainer eval events (0 disables periodic WikiText eval).",
    )
    parser.add_argument(
        "--wikitext_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--wikitext_max_samples",
        type=int,
        default=None,
        help="Optional sample cap for WikiText evaluation.",
    )
    parser.add_argument(
        "--wikitext_block_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--wikitext_stride",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--norm_variant",
        type=str,
        default=None,
        choices=["layernorm", "none", "dyt", "verifiable_pwl_norm_v1", "verifiable_pwl_norm_v2", "verifiable_pwl_norm_v3", "signed_l1_band_norm"],
        help="Normalization variant. Defaults to config value or layernorm.",
    )
    parser.add_argument(
        "--attn_variant",
        type=str,
        default=None,
        choices=["softmax", "sparsemax"],
        help="Attention variant. Defaults to config value or softmax.",
    )
    parser.add_argument(
        "--activation_variant",
        type=str,
        default=None,
        choices=["gelu", "relu", "leaky_relu"],
        help="MLP activation variant. Defaults to config value or gelu.",
    )
    parser.add_argument(
        "--catastrophic_train_loss_threshold",
        type=float,
        default=None,
        help="Stop training if logged train loss is >= this threshold after min-step guard.",
    )
    parser.add_argument(
        "--catastrophic_eval_loss_threshold",
        type=float,
        default=None,
        help="Stop training if eval loss is >= this threshold after min-step guard.",
    )
    parser.add_argument(
        "--stop_on_inf_grad_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop training if logged grad_norm becomes inf/nan.",
    )
    parser.add_argument(
        "--catastrophic_guard_min_step",
        type=int,
        default=None,
        help="Minimum global step before catastrophic guards activate.",
    )
    parser.add_argument(
        "--catastrophic_train_increase_delta",
        type=float,
        default=None,
        help="Trigger guard if train loss increases by this delta above best, sustained for patience.",
    )
    parser.add_argument(
        "--catastrophic_eval_increase_delta",
        type=float,
        default=None,
        help="Trigger guard if eval loss increases by this delta above best, sustained for patience.",
    )
    parser.add_argument(
        "--catastrophic_increase_patience",
        type=int,
        default=None,
        help="Number of consecutive log/eval events required for increase-based guard trigger.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    pattern = re.compile(r"^checkpoint-(\d+)$")
    candidates = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(output_dir, name)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def count_params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def write_run_status(output_dir: str, status: str, stage: str, extra: dict = None) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    payload = {
        "status": status,
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_status.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def get_distributed_context():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def processed_dataset_ready_marker_path(processed_dataset_dir: str) -> str:
    return os.path.join(processed_dataset_dir, "_READY")


def is_processed_dataset_ready(processed_dataset_dir: str) -> bool:
    if not os.path.isdir(processed_dataset_dir):
        return False
    if os.path.isfile(processed_dataset_ready_marker_path(processed_dataset_dir)):
        return True
    dataset_dict_file = os.path.join(processed_dataset_dir, "dataset_dict.json")
    train_dir = os.path.join(processed_dataset_dir, "train")
    validation_dir = os.path.join(processed_dataset_dir, "validation")
    return os.path.isfile(dataset_dict_file) and os.path.isdir(train_dir) and os.path.isdir(validation_dir)


def mark_processed_dataset_ready(processed_dataset_dir: str) -> None:
    with open(processed_dataset_ready_marker_path(processed_dataset_dir), "w", encoding="utf-8") as handle:
        json.dump({"ready": True, "updated_at_utc": datetime.now(timezone.utc).isoformat()}, handle)


def prepare_resume_checkpoint_without_optimizer(src_checkpoint: str, output_dir: str) -> tuple[str, list[str]]:
    reset_root = os.path.join(output_dir, "resume_reset_checkpoints")
    os.makedirs(reset_root, exist_ok=True)
    dst_checkpoint = os.path.join(reset_root, os.path.basename(src_checkpoint))
    if os.path.isdir(dst_checkpoint):
        shutil.rmtree(dst_checkpoint)
    shutil.copytree(src_checkpoint, dst_checkpoint)

    removed_files = []
    for pattern in ["optimizer.pt", "optimizer.bin", "scheduler.pt", "scaler.pt", "rng_state*.pth"]:
        for path in glob.glob(os.path.join(dst_checkpoint, pattern)):
            if os.path.isfile(path):
                os.remove(path)
                removed_files.append(path)

    return dst_checkpoint, removed_files


def default_processed_dataset_dir(args, cfg) -> str:
    train_tag = "all" if args.max_train_samples is None else str(args.max_train_samples)
    eval_tag = "all" if args.max_eval_samples is None else str(args.max_eval_samples)
    dataset_tag = cfg["dataset_name"].replace("/", "-")
    return os.path.join(
        "artifacts",
        "processed",
        f"{dataset_tag}_block{cfg['block_size']}_train{train_tag}_eval{eval_tag}",
    )


def tokenize_and_group(
    raw_datasets: DatasetDict,
    tokenizer,
    block_size: int,
    preprocessing_num_proc: int,
) -> DatasetDict:
    def tokenize_function(batch):
        return tokenizer(batch["text"])

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_proc,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    def group_texts(batch):
        concatenated = {
            key: list(chain.from_iterable(batch[key]))
            for key in batch.keys()
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            key: [tokens[i : i + block_size] for i in range(0, total_length, block_size)]
            for key, tokens in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_proc,
        desc=f"Grouping into blocks of {block_size}",
    )
    return grouped


def create_optimizer_with_weight_decay_exclusions(model, learning_rate: float, weight_decay: float, betas: tuple, eps: float):
    """
    Create optimizer with proper parameter grouping.

    Excludes normalization parameters from weight decay:
    - beta parameters in DyTNorm, PiecewiseLinearNorm, VerifiablePWLNorm
    - pre_scale, pre_bias, post_scale, post_bias in DyTNorm
    - gamma, beta in PiecewiseLinearNorm
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Exclude norm parameters from weight decay
        if any(substring in name for substring in [
            ".beta",  # beta in all norm variants
            ".gamma",  # gamma in PiecewiseLinearNorm
            ".pre_scale",  # DyTNorm
            ".pre_bias",  # DyTNorm
            ".post_scale",  # DyTNorm
            ".post_bias",  # DyTNorm
            "ln_f.weight",  # LayerNorm final layer weight
            "ln_f.bias",  # LayerNorm final layer bias
            "ln_1.weight",  # LayerNorm block 1 weight
            "ln_1.bias",  # LayerNorm block 1 bias
            "ln_2.weight",  # LayerNorm block 2 weight
            "ln_2.bias",  # LayerNorm block 2 bias
        ]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    write_run_status(args.output_dir, status="running", stage="initializing")

    try:
        preprocessing_num_proc = args.preprocessing_num_proc
        if preprocessing_num_proc is None:
            preprocessing_num_proc = cfg.get("preprocessing_num_proc")
        if preprocessing_num_proc is None:
            cpu_count = os.cpu_count() or 1
            preprocessing_num_proc = max(1, cpu_count // 2)

        processed_dataset_dir = args.processed_dataset_dir
        if processed_dataset_dir is None:
            processed_dataset_dir = cfg.get("processed_dataset_dir")
        if processed_dataset_dir is None:
            processed_dataset_dir = default_processed_dataset_dir(args, cfg)

        write_run_status(
            args.output_dir,
            status="running",
            stage="building_model",
            extra={
                "processed_dataset_dir": processed_dataset_dir,
                "preprocessing_num_proc": preprocessing_num_proc,
            },
        )

        model_name = cfg["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_config = GPT2Config.from_pretrained(model_name)
        if hasattr(model_config, "n_positions"):
            model_config.n_positions = cfg["block_size"]
        if hasattr(model_config, "n_ctx"):
            model_config.n_ctx = cfg["block_size"]

        # Determine variants before model creation
        norm_variant = args.norm_variant if args.norm_variant is not None else cfg.get("norm_variant", "layernorm")
        attn_variant = args.attn_variant if args.attn_variant is not None else cfg.get("attn_variant", "softmax")
        activation_variant = args.activation_variant if args.activation_variant is not None else cfg.get("activation_variant", "gelu")

        # Update config activation before model creation so it's saved correctly
        if activation_variant == "leaky_relu":
            model_config.activation_function = "leaky_relu"
        elif activation_variant == "relu":
            model_config.activation_function = "relu"
        # else: keep default (gelu_new or whatever config.activation_function was)

        model = GPT2LMHeadModel(model_config)

        apply_model_variants(model, norm_variant=norm_variant, attn_variant=attn_variant, activation_variant=activation_variant)

        # Fail-fast verification for sparsemax patch
        if attn_variant == "sparsemax":
            print("Verifying sparsemax patch is active...")
            global _sparsemax_call_count
            _sparsemax_call_count = 0

            device = next(model.parameters()).device
            dummy = torch.randint(0, model.config.vocab_size, (1, 16), device=device)
            with torch.no_grad():
                model(dummy)

            assert _sparsemax_call_count > 0, (
                f"Sparsemax patch is not being used! Expected >0 calls, got {_sparsemax_call_count}. "
                "The forward method patch may not be invoked correctly."
            )
            print(f"✓ Sparsemax patch verified: {_sparsemax_call_count} calls during dummy forward")

        model_num_params = count_params(model)
        print(f"Model params: {model_num_params:,}")
        if int(os.environ.get("RANK", "0")) == 0:
            with open(os.path.join(args.output_dir, "model_info.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "model_name": model_name,
                        "num_parameters": model_num_params,
                        "norm_variant": norm_variant,
                        "attn_variant": attn_variant,
                        "activation_variant": activation_variant,
                    },
                    handle,
                    indent=2,
                )
        if cfg.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()

        dataset_name = cfg["dataset_name"]
        rank, world_size = get_distributed_context()
        is_main_process = rank == 0
        if args.streaming:
            raise ValueError("Streaming mode is not supported with Trainer in this script.")
        else:
            if is_processed_dataset_ready(processed_dataset_dir):
                write_run_status(
                    args.output_dir,
                    status="running",
                    stage="loading_processed_dataset",
                    extra={"processed_dataset_dir": processed_dataset_dir},
                )
                print(f"Loading preprocessed dataset from: {processed_dataset_dir}")
                lm_datasets = load_from_disk(processed_dataset_dir)
            else:
                if world_size > 1 and not is_main_process:
                    write_run_status(
                        args.output_dir,
                        status="running",
                        stage="waiting_for_processed_dataset",
                        extra={"processed_dataset_dir": processed_dataset_dir, "rank": rank},
                    )
                    print(
                        f"Rank {rank} waiting for rank 0 to preprocess dataset at: {processed_dataset_dir}"
                    )
                    while not is_processed_dataset_ready(processed_dataset_dir):
                        time.sleep(10)
                    lm_datasets = load_from_disk(processed_dataset_dir)
                else:
                    write_run_status(
                        args.output_dir,
                        status="running",
                        stage="preprocessing_dataset",
                        extra={"processed_dataset_dir": processed_dataset_dir, "rank": rank},
                    )
                    if os.path.isdir(processed_dataset_dir):
                        print(f"Removing incomplete processed dataset dir: {processed_dataset_dir}")
                        shutil.rmtree(processed_dataset_dir)
                    raw_train = load_dataset(dataset_name, split="train[:-1%]")
                    raw_eval = load_dataset(dataset_name, split="train[-1%:]")
                    if args.max_train_samples is not None:
                        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
                    if args.max_eval_samples is not None:
                        raw_eval = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))

                    raw_datasets = DatasetDict({"train": raw_train, "validation": raw_eval})
                    lm_datasets = tokenize_and_group(
                        raw_datasets,
                        tokenizer,
                        cfg["block_size"],
                        preprocessing_num_proc,
                    )
                    os.makedirs(os.path.dirname(processed_dataset_dir), exist_ok=True)
                    lm_datasets.save_to_disk(processed_dataset_dir)
                    mark_processed_dataset_ready(processed_dataset_dir)
                    print(f"Saved preprocessed dataset to: {processed_dataset_dir}")

        data_collator = default_data_collator

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=cfg["train_batch_size_per_device"],
            per_device_eval_batch_size=cfg["eval_batch_size_per_device"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            adam_beta1=cfg["adam_beta1"],
            adam_beta2=cfg["adam_beta2"],
            adam_epsilon=cfg["adam_epsilon"],
            max_steps=args.max_steps if args.max_steps is not None else cfg["max_steps"],
            warmup_steps=cfg["warmup_steps"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            logging_steps=cfg["logging_steps"],
            save_total_limit=cfg["save_total_limit"],
            dataloader_num_workers=cfg["dataloader_num_workers"],
            bf16=cfg["bf16"],
            fp16=cfg["fp16"],
            torch_compile=cfg.get("torch_compile", False),
            report_to=cfg["report_to"],
        )

        early_stop_eval_loss = args.early_stop_eval_loss
        if early_stop_eval_loss is None:
            early_stop_eval_loss = cfg.get("early_stop_eval_loss")

        catastrophic_train_loss_threshold = args.catastrophic_train_loss_threshold
        if catastrophic_train_loss_threshold is None:
            catastrophic_train_loss_threshold = cfg.get("catastrophic_train_loss_threshold")

        catastrophic_eval_loss_threshold = args.catastrophic_eval_loss_threshold
        if catastrophic_eval_loss_threshold is None:
            catastrophic_eval_loss_threshold = cfg.get("catastrophic_eval_loss_threshold")

        stop_on_inf_grad_norm = args.stop_on_inf_grad_norm
        if stop_on_inf_grad_norm is None:
            stop_on_inf_grad_norm = bool(cfg.get("stop_on_inf_grad_norm", False))

        catastrophic_guard_min_step = args.catastrophic_guard_min_step
        if catastrophic_guard_min_step is None:
            catastrophic_guard_min_step = int(cfg.get("catastrophic_guard_min_step", 0))

        catastrophic_train_increase_delta = args.catastrophic_train_increase_delta
        if catastrophic_train_increase_delta is None:
            catastrophic_train_increase_delta = cfg.get("catastrophic_train_increase_delta")

        catastrophic_eval_increase_delta = args.catastrophic_eval_increase_delta
        if catastrophic_eval_increase_delta is None:
            catastrophic_eval_increase_delta = cfg.get("catastrophic_eval_increase_delta")

        catastrophic_increase_patience = args.catastrophic_increase_patience
        if catastrophic_increase_patience is None:
            catastrophic_increase_patience = int(cfg.get("catastrophic_increase_patience", 2))

        wikitext_eval_every_n_evals = args.wikitext_eval_every_n_evals
        if args.use_wikitext_as_dev and wikitext_eval_every_n_evals <= 0:
            wikitext_eval_every_n_evals = 1
        if args.target_wikitext_ppl is not None and wikitext_eval_every_n_evals <= 0:
            wikitext_eval_every_n_evals = 1

        if (args.use_wikitext_as_dev or args.target_wikitext_ppl is not None) and args.early_stop_eval_loss is None:
            early_stop_eval_loss = None

        callbacks = []
        if early_stop_eval_loss is not None:
            callbacks.append(EvalLossThresholdStopCallback(target_eval_loss=float(early_stop_eval_loss)))
        if (
            catastrophic_train_loss_threshold is not None
            or catastrophic_eval_loss_threshold is not None
            or stop_on_inf_grad_norm
            or catastrophic_train_increase_delta is not None
            or catastrophic_eval_increase_delta is not None
        ):
            callbacks.append(
                CatastrophicDivergenceStopCallback(
                    train_loss_threshold=(None if catastrophic_train_loss_threshold is None else float(catastrophic_train_loss_threshold)),
                    eval_loss_threshold=(None if catastrophic_eval_loss_threshold is None else float(catastrophic_eval_loss_threshold)),
                    stop_on_inf_grad_norm=bool(stop_on_inf_grad_norm),
                    min_step=int(catastrophic_guard_min_step),
                    train_increase_delta=(None if catastrophic_train_increase_delta is None else float(catastrophic_train_increase_delta)),
                    eval_increase_delta=(None if catastrophic_eval_increase_delta is None else float(catastrophic_eval_increase_delta)),
                    increase_patience=int(catastrophic_increase_patience),
                )
            )
        if wikitext_eval_every_n_evals > 0:
            callbacks.append(
                WikiTextEvalCallback(
                    tokenizer=tokenizer,
                    output_dir=args.output_dir,
                    split=args.wikitext_split,
                    block_size=args.wikitext_block_size,
                    stride=args.wikitext_stride,
                    max_samples=args.wikitext_max_samples,
                    eval_every_n_evals=wikitext_eval_every_n_evals,
                    target_ppl=args.target_wikitext_ppl,
                )
            )

        # Add diagnostics callback for verifiable norm variants
        callbacks.append(
            VerifiableNormDiagnosticsCallback(
                output_dir=args.output_dir,
                enabled=(norm_variant in ["verifiable_pwl_norm_v1", "verifiable_pwl_norm_v2", "verifiable_pwl_norm_v3", "signed_l1_band_norm"]),
            )
        )

        # Create optimizer with proper weight decay exclusions for norm and gate parameters
        optimizer = create_optimizer_with_weight_decay_exclusions(
            model=model,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            optimizers=(optimizer, None),  # Use custom optimizer, let Trainer create scheduler
        )

        resume_checkpoint = args.resume_from_checkpoint
        if resume_checkpoint is None and not args.disable_auto_resume:
            resume_checkpoint = find_latest_checkpoint(args.output_dir)
            if resume_checkpoint is not None:
                print(f"Auto-resuming from checkpoint: {resume_checkpoint}")

        if args.reset_optimizer_on_resume and resume_checkpoint is not None:
            reset_marker_path = os.path.join(args.output_dir, "resume_reset_checkpoint_ready.json")
            if rank == 0:
                if os.path.isfile(reset_marker_path):
                    os.remove(reset_marker_path)
                write_run_status(
                    args.output_dir,
                    status="running",
                    stage="preparing_resume_reset_checkpoint",
                    extra={"source_resume_checkpoint": resume_checkpoint},
                )
                reset_checkpoint, removed_files = prepare_resume_checkpoint_without_optimizer(
                    resume_checkpoint,
                    output_dir=args.output_dir,
                )
                print(f"Prepared reset-resume checkpoint: {reset_checkpoint}")
                print(f"Removed state files: {len(removed_files)}")
                with open(reset_marker_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "resume_checkpoint": reset_checkpoint,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                        handle,
                        indent=2,
                    )
                resume_checkpoint = reset_checkpoint
            else:
                while not os.path.isfile(reset_marker_path):
                    time.sleep(2)
                with open(reset_marker_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                resume_checkpoint = payload["resume_checkpoint"]
                print(f"Rank {rank} using reset-resume checkpoint: {resume_checkpoint}")

        write_run_status(
            args.output_dir,
            status="running",
            stage="training",
            extra={
                "resume_from_checkpoint": resume_checkpoint,
                "max_steps": training_args.max_steps,
            },
        )

        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model()
        if rank == 0:
            tokenizer.save_pretrained(args.output_dir)

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        write_run_status(args.output_dir, status="running", stage="final_eval")
        eval_metrics = trainer.evaluate()
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        final_wikitext_metrics = None
        if args.evaluate_wikitext_at_end or args.target_wikitext_ppl is not None:
            if rank == 0:
                wikitext_input_ids = load_wikitext_input_ids(
                    tokenizer,
                    split=args.wikitext_split,
                    max_samples=args.wikitext_max_samples,
                )
                wt_loss, wt_ppl, wt_seq_len = evaluate_causal_lm_perplexity(
                    trainer.model,
                    input_ids=wikitext_input_ids,
                    block_size=args.wikitext_block_size,
                    stride=args.wikitext_stride,
                )
                final_wikitext_metrics = {
                    "dataset": "wikitext-103-raw-v1",
                    "split": args.wikitext_split,
                    "loss": wt_loss,
                    "perplexity": wt_ppl,
                    "seq_len": wt_seq_len,
                    "max_samples": args.wikitext_max_samples,
                }
                with open(os.path.join(args.output_dir, "wikitext_eval_final.json"), "w", encoding="utf-8") as handle:
                    json.dump(final_wikitext_metrics, handle, indent=2)
                print(
                    f"Final WikiText eval: loss={wt_loss:.4f}, perplexity={wt_ppl:.4f}"
                )

        write_run_status(
            args.output_dir,
            status="completed",
            stage="done",
            extra={
                "final_train_loss": metrics.get("train_loss"),
                "final_eval_loss": eval_metrics.get("eval_loss"),
                "final_eval_perplexity": eval_metrics.get("perplexity"),
                "final_wikitext_perplexity": None if final_wikitext_metrics is None else final_wikitext_metrics["perplexity"],
            },
        )
    except KeyboardInterrupt:
        write_run_status(args.output_dir, status="interrupted", stage="interrupted")
        raise
    except Exception as error:
        write_run_status(
            args.output_dir,
            status="failed",
            stage="failed",
            extra={"error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
