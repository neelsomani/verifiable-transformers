"""PyTorch modules that replace neural Q/K attention with frozen programs."""

from __future__ import annotations

import json
import os
from types import MethodType
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from .dsl import AttentionProgram


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    dtype = logits.dtype
    z = logits.float()
    z = torch.where(torch.isfinite(z), z, torch.full_like(z, -1e4))
    z = z - z.max(dim=dim, keepdim=True).values
    sorted_z = torch.sort(z, dim=dim, descending=True).values
    cumsum = sorted_z.cumsum(dim)
    size = z.size(dim)
    ranks = torch.arange(1, size + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = size
    ranks = ranks.view(shape)
    support = 1 + ranks * sorted_z > cumsum
    count = support.long().sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(dim, count - 1) - 1.0) / count.to(z.dtype)
    return torch.clamp(z - tau, min=0.0).to(dtype)


class ProgramAttentionHead(nn.Module):
    """A standalone program head with trainable V and O and no Q/K."""

    def __init__(self, d_model: int, head_dim: int, program: AttentionProgram):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.program = program
        self.value = nn.Linear(d_model, head_dim, bias=True)
        self.output = nn.Linear(head_dim, d_model, bias=True)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        weights = self.program.weights(input_ids, dtype=hidden_states.dtype)
        values = self.value(hidden_states)
        mixed = torch.matmul(weights, values)
        return self.output(mixed)


class ProgrammedAttention(nn.Module):
    """GPT-2 attention with physical Q/K projections only for neural heads."""

    def __init__(
        self,
        original,
        programs: Mapping[int, AttentionProgram],
        *,
        attention_variant: str,
    ):
        super().__init__()
        extending = isinstance(original, ProgrammedAttention)
        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.split_size = original.split_size
        self.scale_attn_weights = original.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = original.scale_attn_by_inverse_layer_idx
        self.layer_idx = original.layer_idx
        self.attention_variant = attention_variant
        existing_programs = dict(original.programs) if extending else {}
        for head, program in programs.items():
            if head in existing_programs and existing_programs[head] != program:
                raise ValueError(f"Refusing to replace existing program head {head}")
        self.programs = {**existing_programs, **dict(programs)}
        self.neural_heads = tuple(
            head for head in range(self.num_heads) if head not in self.programs
        )
        if any(head < 0 or head >= self.num_heads for head in self.programs):
            raise ValueError("Program head index is outside the attention module")

        # Buffers/dropout/projection retain GPT-2 behavior. W_O remains trainable.
        self.register_buffer("bias", original.bias.detach().clone(), persistent=False)
        self.register_buffer(
            "masked_bias", original.masked_bias.detach().clone(), persistent=False
        )
        self.attn_dropout = original.attn_dropout
        self.resid_dropout = original.resid_dropout
        self.c_proj = original.c_proj

        d_model = self.embed_dim
        reference_weight = (
            original.value_proj.weight if extending else original.c_attn.weight
        )
        factory_kwargs = {
            "device": reference_weight.device,
            "dtype": reference_weight.dtype,
        }
        if extending:
            self.value_proj = original.value_proj
        else:
            self.value_proj = nn.Linear(
                d_model, d_model, bias=True, **factory_kwargs
            )
            with torch.no_grad():
                self.value_proj.weight.copy_(
                    original.c_attn.weight[:, 2 * d_model :].transpose(0, 1)
                )
                self.value_proj.bias.copy_(original.c_attn.bias[2 * d_model :])

        neural_width = len(self.neural_heads) * self.head_dim
        if neural_width:
            self.query_proj = nn.Linear(
                d_model, neural_width, bias=True, **factory_kwargs
            )
            self.key_proj = nn.Linear(
                d_model, neural_width, bias=True, **factory_kwargs
            )
            with torch.no_grad():
                if extending:
                    old_head_indices = {
                        head: index for index, head in enumerate(original.neural_heads)
                    }
                    for new_index, head in enumerate(self.neural_heads):
                        old_index = old_head_indices[head]
                        new_slice = slice(
                            new_index * self.head_dim, (new_index + 1) * self.head_dim
                        )
                        old_slice = slice(
                            old_index * self.head_dim, (old_index + 1) * self.head_dim
                        )
                        self.query_proj.weight[new_slice].copy_(
                            original.query_proj.weight[old_slice]
                        )
                        self.query_proj.bias[new_slice].copy_(
                            original.query_proj.bias[old_slice]
                        )
                        self.key_proj.weight[new_slice].copy_(
                            original.key_proj.weight[old_slice]
                        )
                        self.key_proj.bias[new_slice].copy_(
                            original.key_proj.bias[old_slice]
                        )
                else:
                    query_columns = []
                    key_columns = []
                    for head in self.neural_heads:
                        start = head * self.head_dim
                        stop = start + self.head_dim
                        query_columns.extend(range(start, stop))
                        key_columns.extend(range(d_model + start, d_model + stop))
                    self.query_proj.weight.copy_(
                        original.c_attn.weight[:, query_columns].transpose(0, 1)
                    )
                    self.query_proj.bias.copy_(original.c_attn.bias[query_columns])
                    self.key_proj.weight.copy_(
                        original.c_attn.weight[:, key_columns].transpose(0, 1)
                    )
                    self.key_proj.bias.copy_(original.c_attn.bias[key_columns])
        else:
            self.query_proj = None
            self.key_proj = None

        self._context_input_ids: Optional[torch.Tensor] = None

    def set_context(self, input_ids: torch.Tensor) -> None:
        self._context_input_ids = input_ids

    def _neural_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            scores = scores / torch.full(
                [], value.size(-1) ** 0.5, dtype=scores.dtype, device=scores.device
            )
        if self.scale_attn_by_inverse_layer_idx:
            scores = scores / float(self.layer_idx + 1)
        query_length, key_length = query.size(-2), key.size(-2)
        causal = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.full([], -1e4, dtype=scores.dtype, device=scores.device)
        scores = torch.where(causal, scores, mask_value)
        if attention_mask is not None:
            scores = scores + attention_mask
        if self.attention_variant == "sparsemax":
            weights = _sparsemax(scores, dim=-1)
        else:
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        return self.attn_dropout(weights)

    def _program_weights(
        self,
        program: AttentionProgram,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._context_input_ids is None:
            raise RuntimeError("ProgrammedAttention requires input_ids context")
        weights = program.weights(self._context_input_ids, dtype=value.dtype)
        if attention_mask is not None:
            valid = (attention_mask[:, 0, 0, :] == 0).to(value.dtype)
            weights = weights * valid.unsqueeze(1)
            total = weights.sum(dim=-1, keepdim=True)
            weights = torch.where(total > 0, weights / total.clamp_min(1e-12), weights)
        return self.attn_dropout(weights).unsqueeze(1)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("Program heads do not support cross-attention")
        if layer_past is not None or use_cache:
            raise NotImplementedError("Program heads currently require use_cache=False")

        batch, length, _ = hidden_states.shape
        values = self.value_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        neural_by_head: Dict[int, torch.Tensor] = {}
        neural_weights_by_head: Dict[int, torch.Tensor] = {}
        if self.neural_heads:
            query = self.query_proj(hidden_states).view(
                batch, length, len(self.neural_heads), self.head_dim
            ).transpose(1, 2)
            key = self.key_proj(hidden_states).view(
                batch, length, len(self.neural_heads), self.head_dim
            ).transpose(1, 2)
            neural_values = values[:, self.neural_heads, :, :]
            weights = self._neural_weights(
                query, key, neural_values, attention_mask
            )
            mixtures = torch.matmul(weights, neural_values)
            for index, head in enumerate(self.neural_heads):
                neural_by_head[head] = mixtures[:, index, :, :]
                neural_weights_by_head[head] = weights[:, index : index + 1, :, :]

        mixtures = []
        all_weights = []
        for head in range(self.num_heads):
            if head in self.programs:
                weights = self._program_weights(
                    self.programs[head], values[:, head, :, :], attention_mask
                )
                mixtures.append(torch.matmul(weights, values[:, head : head + 1])[:, 0])
                all_weights.append(weights)
            else:
                mixtures.append(neural_by_head[head])
                all_weights.append(neural_weights_by_head[head])

        output = torch.cat(mixtures, dim=-1)
        output = self.resid_dropout(self.c_proj(output))
        result = (output, None)
        if output_attentions:
            result += (torch.cat(all_weights, dim=1),)
        return result


def install_program_heads(
    model,
    programs: Mapping[Tuple[int, int], AttentionProgram],
    *,
    attention_variant: Optional[str] = None,
) -> None:
    """Replace selected GPT-2 heads and arrange token-context propagation."""
    model.config.use_cache = False
    by_layer: Dict[int, Dict[int, AttentionProgram]] = {}
    for (layer, head), program in programs.items():
        by_layer.setdefault(layer, {})[head] = program

    for layer, layer_programs in by_layer.items():
        original = model.transformer.h[layer].attn
        variant = attention_variant or getattr(
            original, "_verifiable_attention_variant", "softmax"
        )
        model.transformer.h[layer].attn = ProgrammedAttention(
            original,
            layer_programs,
            attention_variant=variant,
        )

    if not hasattr(model, "_program_original_forward"):
        model._program_original_forward = model.forward

        def forward_with_program_context(self, input_ids=None, *args, **kwargs):
            if input_ids is None:
                raise ValueError("Program heads require input_ids, not inputs_embeds alone")
            for block in self.transformer.h:
                if isinstance(block.attn, ProgrammedAttention):
                    block.attn.set_context(input_ids)
            return self._program_original_forward(input_ids=input_ids, *args, **kwargs)

        model.forward = MethodType(forward_with_program_context, model)


def load_programs(path: str) -> Dict[Tuple[int, int], AttentionProgram]:
    """Load a ``programs.json`` mapping keyed by ``<layer>.<head>``."""
    with open(path) as handle:
        raw = json.load(handle)
    return {
        tuple(int(part) for part in key.split(".")): AttentionProgram.from_dict(value)
        for key, value in raw.items()
    }


def save_programs(
    programs: Mapping[Tuple[int, int], AttentionProgram], path: str
) -> None:
    """Save program heads without serializing executable Python objects."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        f"{layer}.{head}": program.to_dict()
        for (layer, head), program in sorted(programs.items())
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
