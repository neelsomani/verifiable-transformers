"""Small end-to-end verifiable Transformer.

Minimal Transformer for formal SMT verification on three symbolic tasks:
- quote_close: Match opening quotes
- bracket_type: Match opening brackets
- add_mod_5: Addition modulo 5
"""

from . import vocab
from .vocab import (
    VOCAB_SIZE,
    BOS,
    PAD,
    TASK_QUOTE,
    TASK_BRACKET,
    TASK_ADD,
    token_to_str,
    tokens_to_str,
)
from .dataset import SmallVerifiableDataset, get_eval_dataset, collate_fn
from .config import SmallVerifiableConfig, get_default_config, get_tiny_config

__all__ = [
    "vocab",
    "VOCAB_SIZE",
    "BOS",
    "PAD",
    "TASK_QUOTE",
    "TASK_BRACKET",
    "TASK_ADD",
    "token_to_str",
    "tokens_to_str",
    "SmallVerifiableDataset",
    "get_eval_dataset",
    "collate_fn",
    "SmallVerifiableConfig",
    "get_default_config",
    "get_tiny_config",
]
