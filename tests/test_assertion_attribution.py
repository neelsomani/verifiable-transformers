from z3 import Bool, Solver

from scripts.smt.attribution import (
    ASSERTION_CATEGORIES,
    add_profiled_assertion,
    assertion_total,
    increment_norm_instances,
    new_assertion_profile,
)


def test_assertion_profile_has_stable_categories_and_exact_total():
    profile = new_assertion_profile()
    solver = Solver()
    add_profiled_assertion(profile, "norm", solver, Bool("norm_a"), Bool("norm_b"))
    add_profiled_assertion(profile, "decision", solver, Bool("decision"))
    increment_norm_instances(profile, 2)

    assert tuple(profile["assertions"]) == ASSERTION_CATEGORIES
    assert profile["assertions"] == {
        "norm": 2,
        "attention": 0,
        "mlp": 0,
        "embedding/residual": 0,
        "decision": 1,
    }
    assert assertion_total(profile) == len(solver.assertions()) == 3
    assert profile["norm_instances"] == 2
