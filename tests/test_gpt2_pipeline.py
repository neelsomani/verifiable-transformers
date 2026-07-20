import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.gpt2.cluster_preflight import checkpoint_ready, processed_dataset_ready
from scripts.gpt2.extract import load_model_with_variants
from scripts.gpt2.remove_layernorm import validate_bandnorm_eval_loss_gate
from scripts.gpt2.select_base_model import select


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
