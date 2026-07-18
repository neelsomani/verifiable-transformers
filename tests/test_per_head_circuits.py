import torch
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.circuits import (
    CircuitGraph,
    controlled_forward,
    controlled_forward_block,
    expand_block_edges,
)
from scripts.small.config import SmallVerifiableConfig
from scripts.small.train import create_small_model
from scripts.small.extract import cleanup_graph


def make_model(n_heads: int = 2) -> GPT2LMHeadModel:
    torch.manual_seed(7)
    config = GPT2Config(
        vocab_size=23,
        n_positions=8,
        n_embd=12,
        n_layer=2,
        n_head=n_heads,
        n_inner=24,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    return GPT2LMHeadModel(config).eval()


def test_all_heads_reproduce_native_gpt2():
    model = make_model()
    graph = CircuitGraph(n_layers=2, n_heads=2, per_head=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        native = model(input_ids, attention_mask=attention_mask).logits
        controlled = controlled_forward(
            model,
            input_ids,
            graph.all_edges,
            graph,
            attention_mask=attention_mask,
        )

    torch.testing.assert_close(controlled, native, atol=2e-6, rtol=2e-5)


def test_union_of_heads_reproduces_legacy_block_circuit():
    model = make_model()
    block_graph = CircuitGraph(n_layers=2, per_head=False)
    head_graph = CircuitGraph(n_layers=2, n_heads=2, per_head=True)

    # A nontrivial legacy subgraph: several residual edges are deleted while
    # both attention blocks remain connected to the output.
    block_edges = block_graph.all_edges - {
        ("emb", "mlp_0"),
        ("mlp_0", "attn_1"),
        ("attn_0", "logits"),
        ("mlp_1", "logits"),
    }
    head_edges = expand_block_edges(block_edges, n_heads=2)
    input_ids = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])

    with torch.no_grad():
        legacy = controlled_forward_block(
            model, input_ids, block_edges, block_graph
        )
        per_head = controlled_forward(
            model, input_ids, head_edges, head_graph
        )

    torch.testing.assert_close(per_head, legacy, atol=2e-6, rtol=2e-5)


def test_deleting_head_masks_it_before_output_projection():
    model = make_model()
    graph = CircuitGraph(n_layers=2, n_heads=2, per_head=True)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    # Delete head 1 only on the layer-0 -> layer-0-MLP connection.
    edges = graph.all_edges - {("attn_0_h_1", "mlp_0")}
    with torch.no_grad():
        _, nodes = controlled_forward(
            model,
            input_ids,
            graph.all_edges,
            graph,
            return_node_outputs=True,
        )
        actual = controlled_forward(model, input_ids, edges, graph)

        # Build the same layer-0 MLP input explicitly: embeddings plus W_O of
        # head 0 concatenated with a zeroed head 1, including c_proj bias once.
        attention = model.transformer.h[0].attn
        zero = torch.zeros_like(nodes["attn_0_h_1"])
        projected = attention.c_proj(torch.cat([nodes["attn_0_h_0"], zero], dim=-1))
        mlp0 = model.transformer.h[0].mlp(
            model.transformer.h[0].ln_2(nodes["emb"] + projected)
        )

        # Route the manually reconstructed MLP through the otherwise identical
        # graph by checking its cached value.
        _, ablated_nodes = controlled_forward(
            model,
            input_ids,
            edges,
            graph,
            return_node_outputs=True,
        )

    torch.testing.assert_close(ablated_nodes["mlp_0"], mlp0, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(actual).all()


def test_all_heads_reproduce_sparsemax_model():
    config = SmallVerifiableConfig(
        d_model=12,
        n_layers=2,
        n_heads=2,
        d_mlp=24,
        norm_variant="layer_norm",
        attn_variant="sparsemax",
        activation_variant="leaky_relu",
    )
    model = create_small_model(config).eval()
    graph = CircuitGraph(
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        per_head=True,
    )
    input_ids = torch.tensor([[1, 2, 5, 9, 6, 7], [1, 3, 8, 11, 5, 6]])

    with torch.no_grad():
        native = model(input_ids).logits
        controlled = controlled_forward(model, input_ids, graph.all_edges, graph)

    torch.testing.assert_close(controlled, native, atol=2e-6, rtol=2e-5)


def test_cleanup_preserves_bias_only_paths_to_logits():
    graph = CircuitGraph(n_layers=1, n_heads=2, per_head=True)
    edges = {
        ("attn_0_h_0", "mlp_0"),
        ("mlp_0", "logits"),
        ("emb", "attn_0_h_1"),
    }
    # The first path has no embedding input but can still carry c_attn/MLP
    # biases. The second edge is dead because its child cannot reach logits.
    assert cleanup_graph(edges, graph) == {
        ("attn_0_h_0", "mlp_0"),
        ("mlp_0", "logits"),
    }
