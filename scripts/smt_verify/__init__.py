"""SMT-based formal verification for verifiable transformers."""

from .encoders import (
    encode_leaky_relu,
    encode_signed_l1_band_norm,
    encode_sparsemax,
    encode_multihead_attention_sparsemax,
    encode_mlp,
    encode_nonnegative_l1_projection,
    encode_additive_lift,
)
from .circuit import encode_circuit_forward
from .properties import (
    verify_functional_equivalence,
    verify_content_invariance,
    verify_edge_necessity,
    verify_token_renaming_equivariance,
    verify_structural_constraint,
    verify_continuous_robustness,
)
from .bounded_domain import (
    generate_bounded_sequences,
    generate_quote_close_sequences,
    generate_bracket_type_sequences,
    generate_induction_sequences,
    enumerate_small_domain,
)
from .model_weights import load_model_weights
from .helpers import (
    parse_circuit_edges,
    get_norm_params,
    get_candidate_tokens,
    get_bandnorm_params,
)

__all__ = [
    "encode_leaky_relu",
    "encode_signed_l1_band_norm",
    "encode_sparsemax",
    "encode_multihead_attention_sparsemax",
    "encode_mlp",
    "encode_nonnegative_l1_projection",
    "encode_additive_lift",
    "encode_circuit_forward",
    "verify_functional_equivalence",
    "verify_content_invariance",
    "verify_edge_necessity",
    "verify_token_renaming_equivariance",
    "verify_structural_constraint",
    "verify_continuous_robustness",
    "generate_bounded_sequences",
    "generate_quote_close_sequences",
    "generate_bracket_type_sequences",
    "generate_induction_sequences",
    "enumerate_small_domain",
    "load_model_weights",
    "parse_circuit_edges",
    "get_norm_params",
    "get_candidate_tokens",
    "get_bandnorm_params",
]
