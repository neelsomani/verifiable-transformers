"""
Model configuration for the small verifiable Transformer.

This module defines the configuration for a minimal SMT-representable Transformer.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SmallVerifiableConfig:
    """
    Configuration for small verifiable Transformer.

    Uses SMT-friendly components:
    - Signed L1 BandNorm (not LayerNorm)
    - Sparsemax attention (not softmax)
    - LeakyReLU activation (not GELU)
    """

    # Vocabulary and sequence
    vocab_size: int = 32
    max_seq_len: int = 6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: Optional[int] = None

    # Model architecture
    d_model: int = 16
    n_layers: int = 2
    n_heads: int = 1
    d_mlp: int = 64

    # Embedding configuration
    tie_embeddings: bool = False

    # Normalization
    norm_variant: str = "signed_l1_band_norm"
    norm_l1_low_per_dim: float = 0.55
    norm_l1_high_per_dim: float = 1.05

    # Attention
    attn_variant: str = "sparsemax"

    # Activation function
    activation_variant: str = "leaky_relu"
    leaky_relu_negative_slope: float = 0.01

    # Bias terms
    use_bias: bool = True

    # Dropout (set to 0 for deterministic behavior)
    dropout: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_mlp > 0, "d_mlp must be positive"

        # Validate variant choices
        valid_norms = ["signed_l1_band_norm", "layer_norm", "none", "verifiable_pwl_v1", "verifiable_pwl_v2"]
        assert self.norm_variant in valid_norms, \
            f"norm_variant must be one of {valid_norms}, got {self.norm_variant}"

        valid_attns = ["sparsemax", "softmax"]
        assert self.attn_variant in valid_attns, \
            f"attn_variant must be one of {valid_attns}, got {self.attn_variant}"

        valid_activations = ["leaky_relu", "relu", "gelu"]
        assert self.activation_variant in valid_activations, \
            f"activation_variant must be one of {valid_activations}, got {self.activation_variant}"

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Load configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config() -> SmallVerifiableConfig:
    """
    Get default configuration for small verifiable Transformer.

    This configuration is designed to be:
    - Small enough for SMT encoding
    - Large enough to learn the two syntax tasks
    - Uses only SMT-representable components
    """
    return SmallVerifiableConfig(
        vocab_size=32,
        max_seq_len=6,
        d_model=16,
        n_layers=2,
        n_heads=1,
        d_mlp=64,
        norm_variant="signed_l1_band_norm",
        attn_variant="sparsemax",
        activation_variant="leaky_relu",
        tie_embeddings=False,
        use_bias=True,
        dropout=0.0,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )


def get_tiny_config() -> SmallVerifiableConfig:
    """
    Get an even smaller configuration for testing SMT encoding.

    If d_model=16, n_layers=2 is too large for SMT, use this.
    """
    return SmallVerifiableConfig(
        vocab_size=32,
        max_seq_len=6,
        d_model=8,
        n_layers=1,
        n_heads=1,
        d_mlp=32,
        norm_variant="signed_l1_band_norm",
        attn_variant="sparsemax",
        activation_variant="leaky_relu",
        tie_embeddings=False,
        use_bias=True,
        dropout=0.0,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )


if __name__ == "__main__":
    # Print default configuration
    config = get_default_config()
    print("Default Small Verifiable Transformer Configuration:")
    print("=" * 60)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print(f"Parameter estimate:")
    print(f"  Embedding: {config.vocab_size * config.d_model:,}")
    print(f"  Per layer: ~{4 * config.d_model * config.d_model + 2 * config.d_model * config.d_mlp:,}")
    print(f"  Total layers: {config.n_layers}")
    total_params = (
        config.vocab_size * config.d_model +  # embedding
        config.n_layers * (
            4 * config.d_model * config.d_model +  # attention QKV + output
            2 * config.d_model * config.d_mlp  # MLP up + down
        ) +
        config.d_model +  # final norm
        config.vocab_size * config.d_model  # LM head
    )
    print(f"  Approximate total: {total_params:,} parameters")

    # Test save/load
    print("\n" + "=" * 60)
    print("Testing save/load...")
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "config.json")
        config.save(filepath)
        loaded_config = SmallVerifiableConfig.load(filepath)
        assert config == loaded_config
        print("✓ Save/load successful")
