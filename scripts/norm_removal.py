"""Baroni-style LayerNorm attenuation followed by exact affine folding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class AttenuatedLayerNorm(nn.Module):
    """Interpolate from LayerNorm to a fixed-variance affine map.

    The fixed standard deviation is calibrated with an EMA while attenuation is
    zero, then frozen when the transition begins. At attenuation one the map is
    affine (centering is linear) and can be folded exactly into the consumer.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.register_buffer("fixed_std", torch.tensor(1.0))
        self.register_buffer("attenuation", torch.tensor(0.0))
        self.register_buffer("calibrating", torch.tensor(True))

    @classmethod
    def from_layer_norm(
        cls,
        layer_norm: nn.LayerNorm,
        *,
        momentum: float = 0.9,
    ) -> "AttenuatedLayerNorm":
        module = cls(
            layer_norm.normalized_shape[0],
            eps=layer_norm.eps,
            momentum=momentum,
        ).to(device=layer_norm.weight.device, dtype=layer_norm.weight.dtype)
        with torch.no_grad():
            module.weight.copy_(layer_norm.weight)
            if layer_norm.bias is not None:
                module.bias.copy_(layer_norm.bias)
        return module

    def set_attenuation(self, value: float) -> None:
        value = min(1.0, max(0.0, float(value)))
        if value > 0:
            self.calibrating.fill_(False)
        self.attenuation.fill_(value)

    @torch.no_grad()
    def _update_fixed_std(self, hidden_states: torch.Tensor) -> None:
        observed = (
            hidden_states.detach().float().var(dim=-1, unbiased=False) + self.eps
        ).sqrt().mean()
        self.fixed_std.mul_(self.momentum).add_(observed * (1.0 - self.momentum))

    def affine_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        centered = hidden_states - hidden_states.mean(dim=-1, keepdim=True)
        return centered / self.fixed_std.to(hidden_states.dtype) * self.weight + self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.training and bool(self.calibrating.item()):
            self._update_fixed_std(hidden_states)
        real = F.layer_norm(
            hidden_states,
            (self.hidden_size,),
            self.weight,
            self.bias,
            self.eps,
        )
        alpha = self.attenuation.to(hidden_states.dtype)
        if alpha.item() == 0:
            return real
        affine = self.affine_forward(hidden_states)
        return real * (1 - alpha) + affine * alpha


@dataclass
class NormScheduleEntry:
    name: str
    module: AttenuatedLayerNorm


def install_attenuated_layernorms(
    model,
    *,
    momentum: float = 0.9,
) -> List[NormScheduleEntry]:
    """Replace GPT-2 LayerNorms and return the sequential removal order."""
    entries: List[NormScheduleEntry] = []
    for layer, block in enumerate(model.transformer.h):
        if not isinstance(block.ln_1, nn.LayerNorm) or not isinstance(block.ln_2, nn.LayerNorm):
            raise TypeError("LayerNorm attenuation requires a standard-LN checkpoint")
        block.ln_1 = AttenuatedLayerNorm.from_layer_norm(
            block.ln_1, momentum=momentum
        )
        block.ln_2 = AttenuatedLayerNorm.from_layer_norm(
            block.ln_2, momentum=momentum
        )

    if not isinstance(model.transformer.ln_f, nn.LayerNorm):
        raise TypeError("Final normalization must be standard LayerNorm")
    model.transformer.ln_f = AttenuatedLayerNorm.from_layer_norm(
        model.transformer.ln_f, momentum=momentum
    )

    # Match the successful large-model order: MLP norms, attention norms, final.
    for layer, block in enumerate(model.transformer.h):
        entries.append(NormScheduleEntry(f"h.{layer}.ln_2", block.ln_2))
    for layer, block in enumerate(model.transformer.h):
        entries.append(NormScheduleEntry(f"h.{layer}.ln_1", block.ln_1))
    entries.append(NormScheduleEntry("ln_f", model.transformer.ln_f))
    return entries


def update_attenuation_schedule(
    entries: Iterable[NormScheduleEntry],
    step: int,
    *,
    calibration_steps: int,
    transition_steps: int,
    gap_steps: int,
) -> dict[str, float]:
    """Sequentially attenuate each norm and return the current schedule state."""
    if transition_steps < 1 or gap_steps < 0:
        raise ValueError("transition_steps must be positive and gap_steps nonnegative")
    state = {}
    for index, entry in enumerate(entries):
        start = calibration_steps + index * gap_steps
        alpha = (step - start) / transition_steps
        entry.module.set_attenuation(alpha)
        state[entry.name] = float(entry.module.attenuation.item())
    return state


def _affine_matrix(
    module: AttenuatedLayerNorm,
    *,
    compute_dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if float(module.attenuation.item()) != 1.0:
        raise ValueError("Cannot fold a LayerNorm before attenuation reaches one")
    if compute_dtype is None:
        compute_dtype = module.weight.dtype
    scale = module.weight.detach().to(compute_dtype) / module.fixed_std.detach().to(
        compute_dtype
    )
    hidden_size = scale.numel()
    matrix = torch.diag(scale) - torch.ones(
        hidden_size, 1, device=scale.device, dtype=compute_dtype
    ) @ scale.view(1, -1) / hidden_size
    return matrix, module.bias.detach().to(compute_dtype)


@torch.no_grad()
def _fold_into_conv1d(
    module: AttenuatedLayerNorm,
    projection,
    *,
    compute_dtype: torch.dtype | None = None,
) -> None:
    matrix, bias = _affine_matrix(module, compute_dtype=compute_dtype)
    old_weight = projection.weight.detach().to(matrix.dtype)
    old_bias = projection.bias.detach().to(matrix.dtype)
    projection.weight.copy_((matrix @ old_weight).to(projection.weight.dtype))
    projection.bias.copy_((bias @ old_weight + old_bias).to(projection.bias.dtype))


@torch.no_grad()
def fold_attenuated_layernorms(
    model,
    *,
    compute_dtype: torch.dtype | None = None,
) -> None:
    """Fold every fully attenuated norm, leaving only identity modules."""
    for block in model.transformer.h:
        _fold_into_conv1d(
            block.ln_1, block.attn.c_attn, compute_dtype=compute_dtype
        )
        block.ln_1 = nn.Identity()
        _fold_into_conv1d(block.ln_2, block.mlp.c_fc, compute_dtype=compute_dtype)
        block.ln_2 = nn.Identity()

    final_norm = model.transformer.ln_f
    matrix, bias = _affine_matrix(final_norm, compute_dtype=compute_dtype)
    old_linear = model.lm_head
    old_weight_as_conv = old_linear.weight.detach().transpose(0, 1).to(matrix.dtype)
    old_bias = (
        old_linear.bias.detach().to(matrix.dtype)
        if old_linear.bias is not None
        else torch.zeros(
            old_linear.out_features,
            device=old_linear.weight.device,
            dtype=matrix.dtype,
        )
    )
    replacement = nn.Linear(
        old_linear.in_features,
        old_linear.out_features,
        bias=True,
        device=old_linear.weight.device,
        dtype=old_linear.weight.dtype,
    )
    replacement.weight.copy_(
        (matrix @ old_weight_as_conv)
        .transpose(0, 1)
        .to(replacement.weight.dtype)
    )
    replacement.bias.copy_(
        (bias @ old_weight_as_conv + old_bias).to(replacement.bias.dtype)
    )
    model.lm_head = replacement
    model.transformer.ln_f = nn.Identity()
    model.config.tie_word_embeddings = False
