import torch

from scripts.circuits.forward import _build_residual
from scripts.circuits.graph import CircuitGraph
from scripts.gpt2.readside_calibration import (
    compare_paired_rows,
    compare_semantics_to_audit,
)


class _Conv1D:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, values):
        return values @ self.weight + self.bias


class _Attention:
    head_dim = 2

    def __init__(self):
        self.c_proj = _Conv1D(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 0.0, 1.0, -1.0],
                    [0.5, 1.0, -2.0, 0.0],
                    [1.5, -1.0, 0.0, 2.0],
                ]
            ),
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
        )
        self.resid_dropout = torch.nn.Identity()


class _Block:
    def __init__(self):
        self.attn = _Attention()


class _Transformer:
    def __init__(self):
        self.h = [_Block()]


class _Model:
    def __init__(self):
        self.transformer = _Transformer()


def _fixture():
    graph = CircuitGraph(n_layers=1, n_heads=2, per_head=True)
    template = torch.zeros(1, 1, 4)
    outputs = {
        "attn_0_h_0": torch.tensor([[[2.0, 3.0]]]),
        "attn_0_h_1": torch.tensor([[[5.0, 7.0]]]),
    }
    return _Model(), graph, template, outputs


def test_scalar_readside_gain_changes_only_selected_reader_contribution():
    model, graph, template, outputs = _fixture()
    edges = {("attn_0_h_0", "logits"), ("attn_0_h_1", "logits")}
    baseline = _build_residual(model, outputs, "logits", edges, graph, template)
    calibrated = _build_residual(
        model,
        outputs,
        "logits",
        edges,
        graph,
        template,
        {("attn_0_h_0", "logits"): torch.tensor(2.0)},
    )
    expected_delta = outputs["attn_0_h_0"] @ model.transformer.h[0].attn.c_proj.weight[:2]
    torch.testing.assert_close(calibrated - baseline, expected_delta)


def test_diagonal_readside_gain_is_post_WO_and_channelwise():
    model, graph, template, outputs = _fixture()
    edges = {("attn_0_h_0", "logits")}
    gain = torch.tensor([2.0, 1.0, 0.5, 0.0])
    baseline = _build_residual(model, outputs, "logits", edges, graph, template)
    calibrated = _build_residual(
        model,
        outputs,
        "logits",
        edges,
        graph,
        template,
        {("attn_0_h_0", "logits"): gain},
    )
    contribution = outputs["attn_0_h_0"] @ model.transformer.h[0].attn.c_proj.weight[:2]
    torch.testing.assert_close(calibrated - baseline, contribution * (gain - 1.0))


def test_selected_edge_zero_makes_calibration_delta_vanish():
    model, graph, template, outputs = _fixture()
    edges = {("attn_0_h_1", "logits")}
    baseline = _build_residual(model, outputs, "logits", edges, graph, template)
    calibrated = _build_residual(
        model,
        outputs,
        "logits",
        edges,
        graph,
        template,
        {("attn_0_h_0", "logits"): torch.tensor(99.0)},
    )
    torch.testing.assert_close(calibrated, baseline, rtol=0, atol=0)


def test_unselected_reader_is_unchanged():
    model, graph, template, outputs = _fixture()
    edges = {("attn_0_h_0", "mlp_0")}
    baseline = _build_residual(model, outputs, "mlp_0", edges, graph, template)
    calibrated = _build_residual(
        model,
        outputs,
        "mlp_0",
        edges,
        graph,
        template,
        {("attn_0_h_0", "logits"): torch.tensor(3.0)},
    )
    torch.testing.assert_close(calibrated, baseline, rtol=0, atol=0)


def _row(example_id="x", decision=0, logits=(1.0, 0.0), margin=1.0):
    return {
        "example_id": example_id,
        "target": 0,
        "decision": decision,
        "candidate_logits": list(logits),
        "candidate_margin": margin,
    }


def test_paired_baseline_compares_current_logits_and_margins_at_epsilon():
    comparison = compare_paired_rows(
        [_row()],
        [_row(logits=(1.0 + 9e-6, 0.0), margin=1.0 + 9e-6)],
        1e-5,
    )
    assert comparison["pass"]
    failed = compare_paired_rows(
        [_row()],
        [_row(logits=(1.0 + 2e-5, 0.0), margin=1.0 + 2e-5)],
        1e-5,
    )
    assert not failed["pass"]


def test_old_audit_is_semantic_only_after_amendment():
    current = [_row(logits=(100.0, -100.0), margin=200.0)]
    old_audit = [_row()]
    comparison = compare_semantics_to_audit(current, old_audit)
    assert comparison["pass"]
    assert comparison["numerical_comparison_to_old_audit"] == (
        "not_performed_by_amendment"
    )
