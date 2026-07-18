"""Assertion-source attribution shared by the SMT encoders and properties."""

from __future__ import annotations

from typing import Any


ASSERTION_CATEGORIES = (
    "norm",
    "attention",
    "mlp",
    "embedding/residual",
    "decision",
)


def new_assertion_profile() -> dict[str, Any]:
    return {
        "assertions": {category: 0 for category in ASSERTION_CATEGORIES},
        "norm_instances": 0,
    }


def record_solver_delta(profile, category: str, solver, before: int) -> None:
    if profile is None:
        return
    if category not in ASSERTION_CATEGORIES:
        raise ValueError(f"Unknown assertion category: {category}")
    delta = len(solver.assertions()) - before
    if delta < 0:
        raise RuntimeError("Solver assertion count decreased during attribution")
    profile["assertions"][category] += delta


def add_profiled_assertion(profile, category: str, solver, *constraints) -> None:
    before = len(solver.assertions())
    solver.add(*constraints)
    record_solver_delta(profile, category, solver, before)


def increment_norm_instances(profile, count: int = 1) -> None:
    if profile is not None:
        profile["norm_instances"] += count


def assertion_total(profile) -> int:
    return sum(profile["assertions"].values())
