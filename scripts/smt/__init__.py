"""Core SMT encoders and verification utilities.

Reusable components for SMT-based formal verification of Transformers.
"""

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
from .domain import (
    generate_bounded_sequences,
    generate_quote_close_sequences,
    generate_bracket_type_sequences,
    generate_induction_sequences,
    enumerate_small_domain,
)
from .utils import (
    parse_circuit_edges,
    get_norm_params,
    get_candidate_tokens,
    get_small_candidate_tokens,
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
    "parse_circuit_edges",
    "get_norm_params",
    "get_candidate_tokens",
    "get_small_candidate_tokens",
    "get_bandnorm_params",
]
