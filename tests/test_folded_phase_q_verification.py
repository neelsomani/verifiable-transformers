import torch

from scripts.gpt2.folded_phase_q_verification import (
    CANDIDATES,
    anchor_sequences,
    decision,
    q,
    qstr,
)


def test_q_preserves_exact_fp32_binary_value():
    value = torch.tensor(0.1, dtype=torch.float32).item()
    numerator, denominator = value.as_integer_ratio()
    assert qstr(q(value)) == f"{numerator}/{denominator}"


def test_preserved_smoke_generator_order_is_exact():
    assert anchor_sequences() == [
        ([10, 6, 10], "single"),
        ([10, 1, 10], "double"),
        ([10, 6, 11], "single"),
        ([10, 1, 11], "double"),
        ([11, 6, 10], "single"),
        ([11, 1, 10], "double"),
        ([11, 6, 11], "single"),
        ([11, 1, 11], "double"),
    ]


def test_projected_tie_break_matches_registered_candidate_order():
    assert decision({CANDIDATES[0]: q(0), CANDIDATES[1]: q(0)}) == CANDIDATES[0]
    assert decision({CANDIDATES[0]: q(0), CANDIDATES[1]: q(1)}) == CANDIDATES[1]
