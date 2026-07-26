from scripts.circuits import CircuitGraph
from scripts.gpt2.audit_preheal_necessity import (
    internal_circuit_nodes,
    intervention_edge_sets,
)


def test_whole_circuit_node_zero_is_stronger_than_selected_edge_zero():
    graph = CircuitGraph(n_layers=2, n_heads=2, per_head=True)
    selected = {
        ("emb", "mlp_0"),
        ("mlp_0", "attn_1_h_0"),
        ("attn_1_h_0", "logits"),
    }
    edge_sets = intervention_edge_sets(graph, selected)
    edge_zero_removed = graph.all_edges - edge_sets[
        "whole_selected_circuit_edge_zero"
    ]
    node_zero_removed = graph.all_edges - edge_sets[
        "whole_selected_circuit_node_zero"
    ]
    assert edge_zero_removed == selected
    assert {
        edge for edge in edge_zero_removed if edge[0] != "emb"
    } < node_zero_removed
    assert ("emb", "mlp_0") not in node_zero_removed
    assert all(
        edge[0] not in {"mlp_0", "attn_1_h_0"}
        for edge in edge_sets["whole_selected_circuit_node_zero"]
    )


def test_head_lesions_remove_all_outgoing_edges_before_wo_contribution():
    graph = CircuitGraph(n_layers=10, n_heads=12, per_head=True)
    selected = {("attn_7_h_11", "logits"), ("attn_9_h_0", "logits")}
    edge_sets = intervention_edge_sets(graph, selected)
    assert all(
        edge[0] != "attn_7_h_11"
        for edge in edge_sets["zero_attn_7_h_11"]
    )
    assert all(
        edge[0] not in {"attn_7_h_11", "attn_9_h_0"}
        for edge in edge_sets["zero_both_program_heads"]
    )
    assert internal_circuit_nodes(selected) == ["attn_7_h_11", "attn_9_h_0"]
