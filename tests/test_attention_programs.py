from fractions import Fraction
import json
import sys

import torch
from z3 import Real, Solver, sat
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.programs import (
    AttentionProgram,
    Condition,
    CommandProgramProposer,
    ProgramAttentionHead,
    ProgrammedAttention,
    Rule,
    SynthesisHarness,
    install_program_heads,
)
from scripts.smt.program_head import encode_program_attention_head


def fixed_position_program(position: int) -> AttentionProgram:
    return AttentionProgram(
        rules=(
            Rule(
                Fraction(1),
                (Condition("key_position", "==", position),),
            ),
        ),
        name=f"fixed_{position}",
    )


def test_dsl_uses_exact_rational_normalization_and_round_trips():
    program = AttentionProgram(
        rules=(
            Rule(Fraction(2), (Condition("relative_distance", "==", 0),)),
            Rule(Fraction(1), (Condition("key_token", "==", 5),)),
        ),
        default_weight=Fraction(1),
        name="self_plus_token",
    )
    weights = program.rational_weights([5, 7, 5])
    assert weights[1] == [Fraction(2, 5), Fraction(3, 5), Fraction(0)]
    assert all(sum(row) == 1 for row in weights)
    assert AttentionProgram.from_dict(program.to_dict()) == program

    input_ids = torch.tensor([[5, 7, 5], [7, 5, 7]])
    vectorized = program.weights(input_ids, dtype=torch.float64)
    rational = torch.tensor(
        [
            [[float(value) for value in row] for row in program.rational_weights(tokens)]
            for tokens in input_ids.tolist()
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(vectorized, rational, atol=0.0, rtol=0.0)


def test_synthesis_ranks_iou_but_accepts_only_exact_map():
    input_ids = torch.tensor([[1, 2, 5, 9, 6, 7], [1, 3, 8, 11, 5, 6]])
    target = fixed_position_program(3).weights(input_ids)
    result = SynthesisHarness().synthesize(input_ids, target)

    assert result.accepted
    assert result.score.exact_attention_agreement
    assert result.score.support_iou == 1.0
    assert result.program.rational_weights(input_ids[0].tolist()) == (
        fixed_position_program(3).rational_weights(input_ids[0].tolist())
    )


def test_projected_acceptance_is_separate_from_iou_ranking():
    input_ids = torch.tensor([[1, 2, 3]])
    target = torch.full((1, 3, 3), 1 / 3)
    target = torch.tril(target)
    target = target / target.sum(dim=-1, keepdim=True)
    result = SynthesisHarness().synthesize(
        input_ids,
        target,
        projected_evaluator=lambda program: 1.0,
    )
    assert result.accepted
    assert result.acceptance_reason == "exact projected agreement"


def test_command_proposer_exposes_restricted_lm_loop_without_executing_output():
    payload = fixed_position_program(1).to_dict()
    command = [
        sys.executable,
        "-c",
        "import json,sys; sys.stdin.read(); print(json.dumps(json.loads(sys.argv[1])))",
        json.dumps(payload),
    ]
    proposer = CommandProgramProposer(command, timeout_seconds=10)
    proposals = list(proposer("restricted prompt"))
    assert proposals == [payload]

    input_ids = torch.tensor([[1, 2, 3]])
    target = fixed_position_program(1).weights(input_ids)
    result = SynthesisHarness(proposer=proposer, proposer_rounds=1).synthesize(
        input_ids, target
    )
    assert result.accepted
    assert result.score.exact_attention_agreement


def test_synthesis_mask_excludes_padding_rows_and_tokens():
    input_ids = torch.tensor([[1, 2, 0, 0], [1, 3, 4, 0]])
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    target = fixed_position_program(1).weights(input_ids)
    valid = attention_mask.bool().unsqueeze(-1) & attention_mask.bool().unsqueeze(1)
    target = torch.where(valid, target, torch.full_like(target, 0.123))
    result = SynthesisHarness().synthesize(
        input_ids,
        target,
        attention_mask=attention_mask,
    )
    assert result.accepted
    assert result.score.exact_attention_agreement


def test_vectorized_dsl_matches_rational_interpreter_for_thresholds_and_membership():
    program = AttentionProgram(
        rules=(
            Rule(Fraction(2), (Condition("query_token", "in", (1, 4)),)),
            Rule(Fraction(3), (Condition("absolute_distance", "<=", 2),)),
            Rule(Fraction(1), (Condition("key_position", ">=", 1),)),
            Rule(Fraction(1), (Condition("token_match", "!=", True),)),
        ),
        default_weight=Fraction(1, 2),
    )
    input_ids = torch.tensor([[1, 4, 7, 1], [9, 4, 4, 2]])
    expected = torch.tensor(
        [
            [[float(value) for value in row] for row in program.rational_weights(tokens)]
            for tokens in input_ids.tolist()
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        program.weights(input_ids, dtype=torch.float64),
        expected,
        atol=0.0,
        rtol=0.0,
    )


def test_program_head_has_trainable_v_o_and_no_q_k_parameters():
    head = ProgramAttentionHead(6, 3, fixed_position_program(1))
    names = dict(head.named_parameters())
    assert set(names) == {
        "value.weight",
        "value.bias",
        "output.weight",
        "output.bias",
    }
    assert all(parameter.requires_grad for parameter in names.values())


def test_installed_fully_programmed_attention_physically_deletes_qk():
    config = GPT2Config(
        vocab_size=17,
        n_positions=6,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config).eval()
    program = fixed_position_program(1)
    install_program_heads(model, {(0, 0): program, (0, 1): program})
    attention = model.transformer.h[0].attn

    assert isinstance(attention, ProgrammedAttention)
    assert attention.query_proj is None
    assert attention.key_proj is None
    assert not any("query" in name or "key" in name for name, _ in attention.named_parameters())
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]])).logits
    assert logits.shape == (1, 4, 17)
    assert torch.isfinite(logits).all()


def test_install_program_heads_extends_an_existing_programmed_layer():
    config = GPT2Config(
        vocab_size=17,
        n_positions=6,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config).eval()
    first = fixed_position_program(1)
    second = fixed_position_program(2)
    install_program_heads(model, {(0, 0): first})
    value_before = model.transformer.h[0].attn.value_proj.weight.detach().clone()

    # Supplying the existing mapping again is intentional: synthesis/healing
    # consumes a combined program file while the checkpoint already contains
    # the earlier program.
    install_program_heads(model, {(0, 0): first, (0, 1): second})
    attention = model.transformer.h[0].attn

    assert isinstance(attention, ProgrammedAttention)
    assert attention.programs == {0: first, 1: second}
    assert attention.query_proj is None
    assert attention.key_proj is None
    torch.testing.assert_close(attention.value_proj.weight, value_before)
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]])).logits
    assert logits.shape == (1, 4, 17)
    assert torch.isfinite(logits).all()


def test_smt_program_head_matches_pytorch():
    torch.manual_seed(11)
    module = ProgramAttentionHead(4, 2, fixed_position_program(1)).double().eval()
    input_ids = torch.tensor([[1, 5, 7]])
    hidden = torch.randn(1, 3, 4, dtype=torch.double)
    with torch.no_grad():
        expected = module(hidden, input_ids)[0]

    symbols = [[Real(f"x_{position}_{coord}") for coord in range(4)] for position in range(3)]
    encoded = encode_program_attention_head(
        symbols,
        input_ids[0].tolist(),
        module.program,
        module.value.weight.detach().tolist(),
        module.value.bias.detach().tolist(),
        module.output.weight.detach().tolist(),
        module.output.bias.detach().tolist(),
    )
    solver = Solver()
    for position in range(3):
        for coord in range(4):
            solver.add(symbols[position][coord] == str(hidden[0, position, coord].item()))
    assert solver.check() == sat
    smt_model = solver.model()

    def decimal(expr) -> float:
        value = smt_model.eval(expr, model_completion=True)
        return float(value.numerator_as_long()) / float(value.denominator_as_long())

    actual = torch.tensor(
        [[decimal(encoded[position][coord]) for coord in range(4)] for position in range(3)],
        dtype=torch.double,
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_gpt2_loader_round_trips_norm_free_program_checkpoint(tmp_path):
    from scripts.circuits import CircuitGraph, controlled_forward
    from scripts.gpt2.extract import load_model_with_variants
    from scripts.gpt2.train import apply_model_variants
    from scripts.programs import save_programs

    torch.manual_seed(23)
    config = GPT2Config(
        vocab_size=19,
        n_positions=6,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        tie_word_embeddings=False,
        activation_function="leaky_relu",
    )
    model = GPT2LMHeadModel(config)
    apply_model_variants(model, "none", "sparsemax", "leaky_relu")
    programs = {(0, 0): fixed_position_program(1)}
    install_program_heads(model, programs, attention_variant="sparsemax")
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids).logits

    model.save_pretrained(tmp_path)
    save_programs(programs, tmp_path / "programs.json")
    with open(tmp_path / "model_info.json", "w") as handle:
        json.dump(
            {
                "model_name": "tiny-test",
                "norm_variant": "none",
                "attn_variant": "sparsemax",
                "activation_variant": "leaky_relu",
            },
            handle,
        )
    loaded = load_model_with_variants(str(tmp_path), "cpu")
    with torch.no_grad():
        actual = loaded(input_ids).logits
        graph = CircuitGraph(n_layers=1, n_heads=2, per_head=True)
        controlled = controlled_forward(
            loaded,
            input_ids,
            graph.all_edges,
            graph,
            attention_mask=torch.ones_like(input_ids),
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(controlled, actual, atol=1e-6, rtol=1e-6)
