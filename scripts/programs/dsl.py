"""A deliberately restricted, auditable DSL for attention weight matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Sequence

import torch


FEATURES = {
    "query_token",
    "key_token",
    "query_position",
    "key_position",
    "relative_distance",  # query position - key position
    "absolute_distance",
    "token_match",  # query token == key token
}
OPERATORS = {"==", "!=", "<", "<=", ">", ">=", "in"}


def as_fraction(value: int | float | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


@dataclass(frozen=True)
class Condition:
    """One comparison over token identity or (relative) position."""

    feature: str
    operator: str
    value: int | tuple[int, ...] | bool

    def __post_init__(self) -> None:
        if self.feature not in FEATURES:
            raise ValueError(f"Unsupported DSL feature: {self.feature}")
        if self.operator not in OPERATORS:
            raise ValueError(f"Unsupported DSL operator: {self.operator}")
        if self.operator == "in" and not isinstance(self.value, tuple):
            object.__setattr__(self, "value", tuple(self.value))

    def evaluate(
        self,
        query_token: int,
        key_token: int,
        query_position: int,
        key_position: int,
    ) -> bool:
        values = {
            "query_token": query_token,
            "key_token": key_token,
            "query_position": query_position,
            "key_position": key_position,
            "relative_distance": query_position - key_position,
            "absolute_distance": abs(query_position - key_position),
            "token_match": query_token == key_token,
        }
        left = values[self.feature]
        right = self.value
        if self.operator == "==":
            return left == right
        if self.operator == "!=":
            return left != right
        if self.operator == "<":
            return left < right
        if self.operator == "<=":
            return left <= right
        if self.operator == ">":
            return left > right
        if self.operator == ">=":
            return left >= right
        return left in right

    def to_dict(self) -> dict:
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"feature": self.feature, "operator": self.operator, "value": value}

    @classmethod
    def from_dict(cls, value: dict) -> "Condition":
        condition_value = value["value"]
        if isinstance(condition_value, list):
            condition_value = tuple(condition_value)
        return cls(value["feature"], value["operator"], condition_value)


@dataclass(frozen=True)
class Rule:
    """Add a nonnegative rational weight when every condition is true."""

    weight: Fraction
    conditions: tuple[Condition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", as_fraction(self.weight))
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if self.weight < 0:
            raise ValueError("Attention-program weights must be nonnegative")

    def matches(self, qt: int, kt: int, qp: int, kp: int) -> bool:
        return all(c.evaluate(qt, kt, qp, kp) for c in self.conditions)

    def to_dict(self) -> dict:
        return {
            "weight": str(self.weight),
            "conditions": [condition.to_dict() for condition in self.conditions],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "Rule":
        return cls(
            weight=as_fraction(value["weight"]),
            conditions=tuple(Condition.from_dict(c) for c in value["conditions"]),
        )


@dataclass(frozen=True)
class AttentionProgram:
    """Causal additive rules followed by exact row normalization.

    The only data available to a program are concrete token IDs and positions.
    Each branch contributes a rational constant. For a fixed input, every
    normalized attention weight is therefore a rational constant as well.
    """

    rules: tuple[Rule, ...]
    default_weight: Fraction = Fraction(0)
    name: str = "attention_program"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "default_weight", as_fraction(self.default_weight))
        if self.default_weight < 0:
            raise ValueError("default_weight must be nonnegative")
        if not self.rules and self.default_weight == 0:
            raise ValueError("A program must have a rule or positive default weight")

    def rational_weights(self, tokens: Sequence[int]) -> list[list[Fraction]]:
        length = len(tokens)
        rows: list[list[Fraction]] = []
        for query_position in range(length):
            scores = [Fraction(0) for _ in range(length)]
            for key_position in range(query_position + 1):
                score = self.default_weight
                for rule in self.rules:
                    if rule.matches(
                        int(tokens[query_position]),
                        int(tokens[key_position]),
                        query_position,
                        key_position,
                    ):
                        score += rule.weight
                scores[key_position] = score

            total = sum(scores, Fraction(0))
            if total == 0:
                # A deterministic self fallback keeps every row normalized.
                scores[query_position] = Fraction(1)
                total = Fraction(1)
            rows.append([score / total for score in scores])
        return rows

    def weights(
        self,
        input_ids: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        output_dtype = dtype or torch.float32
        batch, length = input_ids.shape
        device = input_ids.device
        query_token = input_ids[:, :, None]
        key_token = input_ids[:, None, :]
        positions = torch.arange(length, device=device)
        query_position = positions.view(1, length, 1)
        key_position = positions.view(1, 1, length)
        causal = key_position <= query_position
        scores = torch.full(
            (batch, length, length),
            float(self.default_weight),
            dtype=output_dtype,
            device=device,
        )
        scores = scores * causal.to(output_dtype)

        features = {
            "query_token": query_token,
            "key_token": key_token,
            "query_position": query_position,
            "key_position": key_position,
            "relative_distance": query_position - key_position,
            "absolute_distance": (query_position - key_position).abs(),
            "token_match": query_token == key_token,
        }

        def compare(left: torch.Tensor, condition: Condition) -> torch.Tensor:
            right = condition.value
            if condition.operator == "in":
                values = torch.tensor(tuple(right), device=device, dtype=left.dtype)
                return (left.unsqueeze(-1) == values).any(dim=-1)
            if condition.operator == "==":
                return left == right
            if condition.operator == "!=":
                return left != right
            if condition.operator == "<":
                return left < right
            if condition.operator == "<=":
                return left <= right
            if condition.operator == ">":
                return left > right
            return left >= right

        for rule in self.rules:
            matches = torch.ones(
                (batch, length, length), dtype=torch.bool, device=device
            )
            for condition in rule.conditions:
                condition_match = compare(features[condition.feature], condition)
                matches = matches & condition_match.expand(batch, length, length)
            scores = scores + (
                float(rule.weight) * (matches & causal).to(output_dtype)
            )

        totals = scores.sum(dim=-1, keepdim=True)
        needs_fallback = totals == 0
        if needs_fallback.any():
            identity = torch.eye(length, dtype=output_dtype, device=device)
            scores = torch.where(
                needs_fallback.expand_as(scores),
                identity.unsqueeze(0).expand(batch, -1, -1),
                scores,
            )
            totals = scores.sum(dim=-1, keepdim=True)
        return scores / totals

    def to_dict(self) -> dict:
        return {
            "dsl_version": 1,
            "name": self.name,
            "default_weight": str(self.default_weight),
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "AttentionProgram":
        if value.get("dsl_version", 1) != 1:
            raise ValueError(f"Unsupported DSL version: {value['dsl_version']}")
        return cls(
            rules=tuple(Rule.from_dict(rule) for rule in value["rules"]),
            default_weight=as_fraction(value.get("default_weight", "0")),
            name=value.get("name", "attention_program"),
        )


def program_from_conditions(
    conditions: Iterable[Condition],
    *,
    weight: int | str | Fraction = 1,
    name: str,
) -> AttentionProgram:
    return AttentionProgram(
        rules=(Rule(as_fraction(weight), tuple(conditions)),),
        name=name,
    )
