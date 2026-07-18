"""Linear SMT encoding for attention heads with frozen program weights."""

from __future__ import annotations

from typing import Any, Sequence

from z3 import ArithRef, Sum

from scripts.programs.dsl import AttentionProgram
from scripts.smt.encoders import z3_real


def encode_program_attention_head(
    hidden_states: Sequence[Sequence[ArithRef]],
    input_ids: Sequence[int],
    program: AttentionProgram,
    value_weight: Sequence[Sequence[float]],
    value_bias: Sequence[float],
    output_weight: Sequence[Sequence[float]],
    output_bias: Sequence[float],
) -> list[list[ArithRef]]:
    """Encode a frozen program head using linear arithmetic only.

    Matrix orientation follows ``torch.nn.Linear``: ``[out, in]``. The program
    is evaluated on concrete token IDs first, making every attention weight a
    rational SMT constant. There are no Q/K variables and no products between
    symbolic expressions.
    """
    seq_len = len(hidden_states)
    if seq_len != len(input_ids):
        raise ValueError("hidden_states and input_ids must have equal length")
    if not hidden_states:
        return []
    d_model = len(hidden_states[0])
    head_dim = len(value_weight)
    if len(value_bias) != head_dim:
        raise ValueError("value bias dimension mismatch")
    if any(len(row) != d_model for row in value_weight):
        raise ValueError("value weight dimension mismatch")
    if len(output_weight) != len(output_bias):
        raise ValueError("output bias dimension mismatch")
    if any(len(row) != head_dim for row in output_weight):
        raise ValueError("output weight dimension mismatch")

    values = []
    for position in range(seq_len):
        values.append(
            [
                Sum(
                    [
                        z3_real(value_weight[coord][source])
                        * hidden_states[position][source]
                        for source in range(d_model)
                    ]
                )
                + z3_real(value_bias[coord])
                for coord in range(head_dim)
            ]
        )

    rational_weights = program.rational_weights(input_ids)
    output = []
    for query_position in range(seq_len):
        mixture = [
            Sum(
                [
                    z3_real(rational_weights[query_position][key_position])
                    * values[key_position][coord]
                    for key_position in range(seq_len)
                ]
            )
            for coord in range(head_dim)
        ]
        output.append(
            [
                Sum(
                    [
                        z3_real(output_weight[coord][head_coord])
                        * mixture[head_coord]
                        for head_coord in range(head_dim)
                    ]
                )
                + z3_real(output_bias[coord])
                for coord in range(len(output_weight))
            ]
        )
    return output
