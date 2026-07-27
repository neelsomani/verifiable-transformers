from types import SimpleNamespace

import torch

from scripts.gpt2 import heal_programs


def test_program_wrapper_cannot_disable_gradient_accumulation_scaling():
    trainer = SimpleNamespace(model_accepts_loss_kwargs=True)
    heal_programs.enforce_mean_loss_accumulation(trainer)
    assert trainer.model_accepts_loss_kwargs is False


def test_early_gate_aborts_when_full_agreement_moves_wrong_direction(
    monkeypatch, tmp_path
):
    callback = heal_programs.HealingGateCallback(
        domains={"quote_close": {}},
        reference={"quote_close": torch.tensor([0, 0, 0, 0])},
        batch_size=4,
        required_agreement=1.0,
        perplexity_budget=30.0,
        output_dir=str(tmp_path),
        early_abort_min_agreement=0.95,
    )
    monkeypatch.setattr(
        heal_programs,
        "decisions_for_model",
        lambda *_args, **_kwargs: {
            "quote_close": torch.tensor([0, 1, 1, 1])
        },
    )
    control = SimpleNamespace(should_training_stop=False)
    callback.on_evaluate(
        SimpleNamespace(),
        SimpleNamespace(global_step=10, is_world_process_zero=True),
        control,
        metrics={"eval_loss": 3.2},
        model=object(),
    )
    assert control.should_training_stop is True
    assert callback.history[-1]["early_abort_triggered"] is True


def test_full_behavior_loss_weight_is_opt_in():
    source = (
        __import__("inspect")
        .getsource(heal_programs.AblationAwareProgramTrainer.compute_loss)
    )
    assert '"ablation_aware_full_loss_weight"' in source
    assert "full_loss_weight * full_behavior_loss" in source


def test_only_world_process_zero_reports_failed_gate():
    source = __import__("inspect").getsource(heal_programs.main)
    assert (
        'if trainer.is_world_process_zero() and not result["success"]'
        in source
    )


def test_opposite_reference_bypass_penalty_requires_argmax_change():
    trainer = object.__new__(heal_programs.AblationAwareProgramTrainer)
    trainer.healing_config = {"bypass_target_mode": "opposite_reference"}
    logits = torch.tensor([[[4.0, -4.0]]])
    attention_mask = torch.ones(1, 1, dtype=torch.long)
    loss = trainer._bypass_penalty(
        logits, attention_mask, [0, 1], torch.tensor([0])
    )
    assert loss.item() > 7.0


def test_unknown_bypass_target_mode_is_rejected():
    trainer = object.__new__(heal_programs.AblationAwareProgramTrainer)
    trainer.healing_config = {"bypass_target_mode": "invalid"}
    with __import__("pytest").raises(ValueError, match="Unknown"):
        trainer._bypass_penalty(
            torch.zeros(1, 1, 2),
            torch.ones(1, 1, dtype=torch.long),
            [0, 1],
            torch.tensor([0]),
        )
