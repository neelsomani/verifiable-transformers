"""Shared circuit graph and controlled-forward primitives."""

from .graph import CircuitGraph, expand_block_edges
from .forward import controlled_forward, controlled_forward_block

__all__ = [
    "CircuitGraph",
    "controlled_forward",
    "controlled_forward_block",
    "expand_block_edges",
]
