"""Residual-stream circuit graphs with attention-head granularity."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set, Tuple


Edge = Tuple[str, str]
_HEAD_RE = re.compile(r"^attn_(\d+)_h_(\d+)$")
_BLOCK_RE = re.compile(r"^attn_(\d+)$")


class CircuitGraph:
    """A residual-stream DAG whose attention nodes are individual heads.

    Attention nodes hold the head result *before* ``W_O``. When their output is
    consumed, the controlled forward pass groups the retained heads, zeros the
    deleted heads, and applies the source layer's ``W_O`` once. This gives edge
    ablation the same semantics as masking heads immediately before ``W_O``.
    """

    def __init__(self, n_layers: int, n_heads: int = 1, per_head: bool = True):
        if n_layers < 1 or n_heads < 1:
            raise ValueError("n_layers and n_heads must both be positive")
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.per_head = bool(per_head)
        self.nodes: List[str] = []
        self.all_edges: Set[Edge] = set()
        self.incoming_edges: Dict[str, List[Edge]] = {}
        self._build()

    @property
    def edges(self) -> Set[Edge]:
        """Compatibility alias used by the small-model extractor."""
        return self.all_edges

    def get_edges(self) -> Set[Edge]:
        return set(self.all_edges)

    def remove_edge(self, edge: Edge) -> None:
        if edge not in self.all_edges:
            return
        self.all_edges.remove(edge)
        self.incoming_edges[edge[1]].remove(edge)

    def attention_nodes(self, layer: int) -> List[str]:
        if self.per_head:
            return [f"attn_{layer}_h_{head}" for head in range(self.n_heads)]
        return [f"attn_{layer}"]

    def _add_edge(self, source: str, target: str) -> None:
        edge = (source, target)
        self.all_edges.add(edge)
        self.incoming_edges[target].append(edge)

    def _build(self) -> None:
        self.nodes = ["emb"]
        for layer in range(self.n_layers):
            self.nodes.extend(self.attention_nodes(layer))
            self.nodes.append(f"mlp_{layer}")
        self.nodes.append("logits")
        self.incoming_edges = {node: [] for node in self.nodes}

        previous: List[str] = ["emb"]
        for layer in range(self.n_layers):
            heads = self.attention_nodes(layer)
            mlp = f"mlp_{layer}"
            for head in heads:
                for parent in previous:
                    self._add_edge(parent, head)
            for parent in previous + heads:
                self._add_edge(parent, mlp)
            previous.extend(heads)
            previous.append(mlp)

        for parent in previous:
            self._add_edge(parent, "logits")

        for child in self.incoming_edges:
            self.incoming_edges[child].sort()

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "granularity": "head" if self.per_head else "block",
            "nodes": list(self.nodes),
            "edges": [
                {"source": source, "target": target}
                for source, target in sorted(self.all_edges)
            ],
        }


def parse_head_node(node: str) -> tuple[int, int] | None:
    match = _HEAD_RE.match(node)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def expand_block_edges(
    edges: Iterable[Edge], n_heads: int
) -> Set[Edge]:
    """Expand legacy ``attn_i`` endpoints to all corresponding head endpoints.

    An attention-to-attention edge expands to the Cartesian product of source
    and target heads. Consequently, expanding every edge of a block circuit is
    exactly its union-of-heads regression circuit.
    """
    if n_heads < 1:
        raise ValueError("n_heads must be positive")

    def expand(node: str) -> List[str]:
        match = _BLOCK_RE.match(node)
        if match is None:
            return [node]
        layer = int(match.group(1))
        return [f"attn_{layer}_h_{head}" for head in range(n_heads)]

    expanded: Set[Edge] = set()
    for source, target in edges:
        for source_head in expand(source):
            for target_head in expand(target):
                expanded.add((source_head, target_head))
    return expanded
