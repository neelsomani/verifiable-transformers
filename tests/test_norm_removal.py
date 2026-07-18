import torch

from scripts.norm_removal import (
    fold_attenuated_layernorms,
    install_attenuated_layernorms,
)
from scripts.small.config import SmallVerifiableConfig
from scripts.small.train import create_small_model, initialize_from_checkpoint


def test_fully_attenuated_norms_fold_exactly():
    torch.manual_seed(17)
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
    entries = install_attenuated_layernorms(model)
    with torch.no_grad():
        for index, entry in enumerate(entries):
            entry.module.fixed_std.fill_(0.7 + index * 0.1)
            entry.module.set_attenuation(1.0)

    input_ids = torch.tensor([[1, 2, 5, 9, 6, 7], [1, 3, 8, 11, 5, 6]])
    with torch.no_grad():
        before = model(input_ids).logits
        fold_attenuated_layernorms(model)
        after = model(input_ids).logits

    torch.testing.assert_close(after, before, atol=3e-6, rtol=3e-5)
    assert all(isinstance(block.ln_1, torch.nn.Identity) for block in model.transformer.h)
    assert all(isinstance(block.ln_2, torch.nn.Identity) for block in model.transformer.h)
    assert isinstance(model.transformer.ln_f, torch.nn.Identity)


def test_norm_free_config_has_no_normalization_parameters():
    config = SmallVerifiableConfig(norm_variant="none")
    model = create_small_model(config)
    norm_parameter_names = [
        name for name, _ in model.named_parameters() if ".ln_" in name or ".ln_f" in name
    ]
    assert norm_parameter_names == []
    assert model.lm_head.bias is not None


def test_layernorm_checkpoint_initializes_matched_bandnorm_control(tmp_path):
    common = dict(
        d_model=12,
        n_layers=1,
        n_heads=2,
        d_mlp=24,
        attn_variant="sparsemax",
        activation_variant="leaky_relu",
    )
    source = create_small_model(
        SmallVerifiableConfig(norm_variant="layer_norm", **common)
    )
    with torch.no_grad():
        source.transformer.h[0].ln_1.weight.fill_(1.25)
        source.transformer.h[0].ln_1.bias.fill_(-0.5)
    source.save_pretrained(tmp_path)

    target = create_small_model(
        SmallVerifiableConfig(norm_variant="signed_l1_band_norm", **common)
    )
    initialize_from_checkpoint(target, str(tmp_path))
    torch.testing.assert_close(
        target.transformer.h[0].ln_1.gamma,
        source.transformer.h[0].ln_1.weight,
    )
    torch.testing.assert_close(
        target.transformer.h[0].ln_1.beta,
        source.transformer.h[0].ln_1.bias,
    )
    torch.testing.assert_close(
        target.transformer.h[0].attn.c_attn.weight,
        source.transformer.h[0].attn.c_attn.weight,
    )
