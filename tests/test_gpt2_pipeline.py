import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.gpt2.cluster_preflight import checkpoint_ready, processed_dataset_ready
from scripts.gpt2.extract import load_model_with_variants
from scripts.gpt2.remove_layernorm import (
    NormRemovalScheduleCallback,
    fold_and_measure_in_fp32,
    load_trained_removal_checkpoint,
    validate_attenuation_endpoint,
    validate_bandnorm_eval_loss_gate,
)
from scripts.gpt2.select_base_model import select
from scripts.norm_removal import install_attenuated_layernorms


def make_checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "model_info.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"stub")


def test_phase_c_base_selection_enforces_preregistered_gate(tmp_path):
    norm_free = tmp_path / "norm-free"
    bandnorm = tmp_path / "bandnorm"
    make_checkpoint(norm_free)
    make_checkpoint(bandnorm)
    metrics = {
        "status": "passed",
        "post_fold_eval_loss": 3.3179,
        "bandnorm_eval_loss_gate": 3.318,
        "removal_loss_delta": 0.1210,
        "decision": "norm_free",
    }
    result = select(metrics, str(norm_free), str(bandnorm))
    assert result["decision"] == "norm_free"
    assert result["selected_model"] == str(norm_free)

    metrics["post_fold_eval_loss"] = 3.318
    metrics["decision"] = "bandnorm"
    result = select(metrics, str(norm_free), str(bandnorm))
    assert result["decision"] == "bandnorm"
    assert result["selected_model"] == str(bandnorm)


def test_phase_c_base_selection_rejects_goalpost_shift(tmp_path):
    norm_free = tmp_path / "norm-free"
    bandnorm = tmp_path / "bandnorm"
    make_checkpoint(norm_free)
    make_checkpoint(bandnorm)
    metrics = {
        "status": "passed",
        "post_fold_eval_loss": 3.325,
        "bandnorm_eval_loss_gate": 3.318,
        "decision": "norm_free",
    }
    with pytest.raises(RuntimeError, match="contradicts"):
        select(metrics, str(norm_free), str(bandnorm))


def test_removal_rejects_invocation_time_gate_shift():
    cfg = {"bandnorm_eval_loss_gate": 3.318}
    assert validate_bandnorm_eval_loss_gate(cfg, None) == 3.318
    assert validate_bandnorm_eval_loss_gate(cfg, 3.318) == 3.318
    with pytest.raises(ValueError, match="contradicts"):
        validate_bandnorm_eval_loss_gate(cfg, 3.330)


def test_loader_restores_safetensors_omitted_tied_lm_head(tmp_path):
    config = GPT2Config(
        vocab_size=19,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
        tie_word_embeddings=True,
    )
    source = GPT2LMHeadModel(config)
    source.save_pretrained(tmp_path, safe_serialization=True)
    with open(tmp_path / "model_info.json", "w") as handle:
        json.dump(
            {
                "norm_variant": "layernorm",
                "attn_variant": "softmax",
                "activation_variant": "gelu",
            },
            handle,
        )

    state_dict = load_file(tmp_path / "model.safetensors")
    assert "transformer.wte.weight" in state_dict
    assert "lm_head.weight" not in state_dict

    loaded = load_model_with_variants(str(tmp_path), "cpu")
    assert loaded.lm_head.weight is loaded.transformer.wte.weight
    torch.testing.assert_close(
        loaded.transformer.wte.weight, source.transformer.wte.weight
    )


def test_completed_removal_checkpoint_can_be_folded_without_retraining(tmp_path):
    config = GPT2Config(
        vocab_size=19,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
        tie_word_embeddings=True,
    )
    source = GPT2LMHeadModel(config).eval()
    source_entries = install_attenuated_layernorms(source)
    with torch.no_grad():
        for index, entry in enumerate(source_entries):
            entry.module.fixed_std.fill_(0.75 + 0.1 * index)
            entry.module.set_attenuation(1.0)

    checkpoint = tmp_path / "checkpoint-5000"
    source.save_pretrained(checkpoint, safe_serialization=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 5000})
    )

    recovered = GPT2LMHeadModel(config).float().eval()
    recovered_entries = install_attenuated_layernorms(recovered)
    recovered_step = load_trained_removal_checkpoint(
        recovered, str(checkpoint), required_step=5000
    )
    endpoint = validate_attenuation_endpoint(recovered_entries)

    assert recovered_step == 5000
    assert recovered.lm_head.weight is recovered.transformer.wte.weight
    assert all(state["attenuation"] == 1.0 for state in endpoint.values())
    assert all(not state["calibrating"] for state in endpoint.values())
    for expected, actual in zip(source_entries, recovered_entries):
        torch.testing.assert_close(
            actual.module.fixed_std, expected.module.fixed_std
        )

    fold_metrics = fold_and_measure_in_fp32(
        recovered, torch.tensor([[1, 2, 3, 4]])
    )
    assert fold_metrics["max_abs_diff"] < 1e-5
    assert fold_metrics["relative_l2_error"] < 1e-5
    assert fold_metrics["top1_agreement"] == 1.0


def test_removal_recovery_rejects_incomplete_schedule(tmp_path):
    config = GPT2Config(
        vocab_size=19,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
    )
    model = GPT2LMHeadModel(config)
    entries = install_attenuated_layernorms(model)
    with pytest.raises(RuntimeError, match="not at the fixed-std endpoint"):
        validate_attenuation_endpoint(entries)

    checkpoint = tmp_path / "checkpoint-4999"
    model.save_pretrained(checkpoint, safe_serialization=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 4999})
    )
    with pytest.raises(ValueError, match="required at least 5000"):
        load_trained_removal_checkpoint(model, str(checkpoint), required_step=5000)


def test_recovery_callback_does_not_rewind_at_trainer_step_zero(tmp_path):
    config = GPT2Config(
        vocab_size=19,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
    )
    model = GPT2LMHeadModel(config)
    entries = install_attenuated_layernorms(model)
    with torch.no_grad():
        for entry in entries:
            entry.module.fixed_std.fill_(0.9)
            entry.module.set_attenuation(1.0)

    callback = NormRemovalScheduleCallback(
        entries,
        {
            "calibration_steps": 500,
            "transition_steps": 500,
            "gap_steps": 100,
        },
        str(tmp_path),
        fixed_step=5000,
    )
    callback.on_log(
        None,
        SimpleNamespace(global_step=0, is_world_process_zero=False),
        SimpleNamespace(),
        logs={"eval_loss": 3.2},
    )

    endpoint = validate_attenuation_endpoint(entries)
    assert all(state["attenuation"] == 1.0 for state in endpoint.values())
    assert callback.history[-1]["step"] == 5000
    assert callback.history[-1]["trainer_step"] == 0


def test_cluster_preflight_requires_variant_metadata_and_complete_dataset(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    make_checkpoint(checkpoint)
    assert checkpoint_ready(str(checkpoint))
    (checkpoint / "model_info.json").unlink()
    assert not checkpoint_ready(str(checkpoint))

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    assert not processed_dataset_ready(str(dataset))
    (dataset / "dataset_dict.json").write_text("{}")
    (dataset / "train").mkdir()
    (dataset / "validation").mkdir()
    assert processed_dataset_ready(str(dataset))
