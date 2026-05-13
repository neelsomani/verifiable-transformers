"""SMT encoders for verifiable transformer components."""

from z3 import *
from fractions import Fraction
import math
from typing import List, Tuple


def z3_real(value) -> ArithRef:
    """Convert Python numeric constants to Z3 rationals without float parsing."""
    if isinstance(value, ArithRef):
        return value
    if isinstance(value, bool):
        return RealVal(int(value))
    if isinstance(value, int):
        return RealVal(value)

    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"Non-finite SMT constant: {value!r}")

    fraction = Fraction(str(value_float))
    return Q(fraction.numerator, fraction.denominator)


def encode_leaky_relu(x: ArithRef, alpha: float = 0.01) -> ArithRef:
    """Encode LeakyReLU as piecewise-linear SMT constraint.

    LeakyReLU(x) = x if x >= 0 else alpha * x

    Args:
        x: Input Z3 variable
        alpha: Negative slope (default 0.01)

    Returns:
        Z3 expression for LeakyReLU(x)
    """
    return If(x >= 0, x, z3_real(alpha) * x)


def encode_leaky_relu_with_trace(
    x: ArithRef,
    is_nonnegative: bool,
    solver: Solver,
    ctx_prefix: str,
    alpha: float = 0.01,
) -> ArithRef:
    """Encode LeakyReLU on a certified sign branch."""
    if is_nonnegative:
        solver.add(x >= 0)
        return x

    solver.add(x < 0)
    return z3_real(alpha) * x


def encode_nonnegative_l1_projection(
    y: List[ArithRef],
    radius: float,
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode nonnegative L1 ball projection using top-k thresholding.

    If sum(y) <= radius: return y
    Else: return max(y - tau, 0) where tau is threshold

    Args:
        y: Input vector (nonnegative)
        radius: L1 ball radius
        solver: Z3 solver
        ctx_prefix: Prefix for auxiliary variables

    Returns:
        Projected vector
    """
    n = len(y)
    mass = Sum(y)

    # Threshold variable
    tau = Real(f"{ctx_prefix}_tau")

    # Projected values: max(y - tau, 0)
    proj = [If(y[i] > tau, y[i] - tau, RealVal(0)) for i in range(n)]

    # Constraint: projected sum equals radius (if mass > radius)
    proj_mass = Sum(proj)

    # Add constraint based on whether projection is needed
    # If mass <= radius: tau = 0 (no projection)
    # If mass > radius: proj_mass = radius
    radius = z3_real(radius)
    solver.add(If(mass <= radius, tau == 0, proj_mass == radius))

    # Ensure tau is non-negative
    solver.add(tau >= 0)

    return proj


def encode_additive_lift(
    y: List[ArithRef],
    target: float,
    fallback_mask: List[float],
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode additive lift for low L1 mass.

    If sum(y) < target, add mass uniformly to active coordinates.

    Args:
        y: Input vector (nonnegative)
        target: Target L1 mass
        fallback_mask: Fallback active mask (used if no entries > 0)
        solver: Z3 solver
        ctx_prefix: Prefix for auxiliary variables

    Returns:
        Lifted vector
    """
    n = len(y)
    mass = Sum(y)

    # Check if any coordinate is active
    any_active = Or([y[i] > 0 for i in range(n)])

    # Active mask: if any y[i] > 0 globally, use y[i] > 0; otherwise use fallback
    active = [If(any_active,
                 If(y[i] > 0, RealVal(1), RealVal(0)),
                 z3_real(fallback_mask[i]))
              for i in range(n)]
    active_count = Sum(active)

    # Delta per active coordinate
    delta_var = Real(f"{ctx_prefix}_delta")
    target = z3_real(target)
    solver.add(delta_var == (target - mass) / active_count)

    # Lifted values
    lifted = [y[i] + active[i] * delta_var for i in range(n)]

    # Apply lift only if mass < target
    return [If(mass < target, lifted[i], y[i]) for i in range(n)]


def encode_signed_l1_band_norm(
    x: List[ArithRef],
    gamma: List[float],
    beta: List[float],
    half_low: float,
    half_high: float,
    pos_fallback: List[float],
    neg_fallback: List[float],
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode Signed L1 BandNorm exactly as implemented in train_experiment.py.

    Algorithm:
    1. Center: c = x - mean(x)
    2. Split: p = ReLU(c), n = ReLU(-c)
    3. High-mass projection: L1 ball projection if mass > half_high
    4. Low-mass lift: additive lift if mass < half_low
    5. Recombine: z = p - n
    6. Recenter: z = z - mean(z)
    7. Affine: output = gamma * z + beta

    Args:
        x: Input vector
        gamma: Affine scale parameters
        beta: Affine bias parameters
        half_low: Minimum L1 mass target (per sign)
        half_high: Maximum L1 mass (ball radius, per sign)
        pos_fallback: Fallback mask for positive mass
        neg_fallback: Fallback mask for negative mass
        solver: Z3 solver
        ctx_prefix: Prefix for auxiliary variables

    Returns:
        Normalized vector
    """
    d = len(x)

    # Step 1: Center
    mean_x = Sum(x) / d
    c = [x[i] - mean_x for i in range(d)]

    # Step 2: Split into positive and negative
    p = [If(c[i] > 0, c[i], RealVal(0)) for i in range(d)]
    n = [If(c[i] < 0, -c[i], RealVal(0)) for i in range(d)]

    # Step 3: High-mass projection
    p_projected = encode_nonnegative_l1_projection(p, half_high, solver, f"{ctx_prefix}_p_proj")
    n_projected = encode_nonnegative_l1_projection(n, half_high, solver, f"{ctx_prefix}_n_proj")

    # Step 4: Low-mass additive lift
    p_normalized = encode_additive_lift(p_projected, half_low, pos_fallback, solver, f"{ctx_prefix}_p_lift")
    n_normalized = encode_additive_lift(n_projected, half_low, neg_fallback, solver, f"{ctx_prefix}_n_lift")

    # Step 5: Recombine
    z = [p_normalized[i] - n_normalized[i] for i in range(d)]

    # Step 6: Recenter (optional exact recentering)
    mean_z = Sum(z) / d
    z_recentered = [z[i] - mean_z for i in range(d)]

    # Step 7: Affine transform
    output = [z3_real(gamma[i]) * z_recentered[i] + z3_real(beta[i]) for i in range(d)]

    return output


def encode_sparsemax(
    logits: List[ArithRef],
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode sparsemax (alpha-entmax with alpha=2) using SMT.

    Sparsemax is projection onto probability simplex.
    Uses the threshold characterization:
    sparsemax(z)_i = max(z_i - tau, 0)
    where tau is chosen so sum(sparsemax(z)) = 1

    Args:
        logits: Input logits
        solver: Z3 solver
        ctx_prefix: Prefix for auxiliary variables

    Returns:
        Sparsemax output (probability distribution)
    """
    n = len(logits)

    # Threshold variable
    tau = Real(f"{ctx_prefix}_sparsemax_tau")

    # Sparsemax output
    output = []
    for i, z in enumerate(logits):
        out_i = If(z > tau, z - tau, RealVal(0))
        output.append(out_i)

    # Constraint: sum to 1 (probability simplex)
    solver.add(Sum(output) == 1)

    # Constraint: all non-negative (implied by max(z - tau, 0))
    for out_i in output:
        solver.add(out_i >= 0)

    return output


def encode_sparsemax_with_support(
    logits: List[ArithRef],
    support: List[int],
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode sparsemax on a certified support branch."""
    n = len(logits)
    support = sorted(support)
    if not support:
        raise ValueError(f"Empty sparsemax support for {ctx_prefix}")

    support_set = set(support)
    k = len(support)
    tau = (Sum([logits[i] for i in support]) - z3_real(1.0)) / k

    output = []
    for i in range(n):
        if i in support_set:
            out_i = logits[i] - tau
            solver.add(logits[i] > tau)
        else:
            out_i = RealVal(0)
            solver.add(logits[i] <= tau)
        output.append(out_i)

    solver.add(Sum(output) == z3_real(1.0))
    return output


def encode_nonnegative_l1_projection_with_trace(
    y: List[ArithRef],
    radius: float,
    trace: dict,
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode nonnegative L1 projection on a certified branch."""
    radius_z3 = z3_real(radius)
    mass = Sum(y)

    if not trace["needed"]:
        solver.add(mass <= radius_z3)
        return y

    support = sorted(trace["support"])
    if not support:
        raise ValueError(f"Empty projection support for {ctx_prefix}")

    support_set = set(support)
    tau = (Sum([y[i] for i in support]) - radius_z3) / len(support)
    solver.add(mass > radius_z3)

    projected = []
    for i in range(len(y)):
        if i in support_set:
            solver.add(y[i] > tau)
            projected.append(y[i] - tau)
        else:
            solver.add(y[i] <= tau)
            projected.append(RealVal(0))

    return projected


def encode_additive_lift_with_trace(
    y: List[ArithRef],
    target: float,
    fallback_mask: List[float],
    trace: dict,
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode additive lift on a certified branch."""
    target_z3 = z3_real(target)
    mass = Sum(y)

    if not trace["needed"]:
        solver.add(mass >= target_z3)
        return y

    active = sorted(trace["active"])
    if not active:
        raise ValueError(f"Empty additive-lift active set for {ctx_prefix}")

    active_set = set(active)
    solver.add(mass < target_z3)

    if trace.get("fallback", False):
        for i in range(len(y)):
            solver.add(y[i] <= 0)
            expected = 1.0 if i in active_set else 0.0
            if float(fallback_mask[i]) != expected:
                raise ValueError(f"Fallback mask mismatch for {ctx_prefix}[{i}]")
    else:
        for i in range(len(y)):
            if i in active_set:
                solver.add(y[i] > 0)
            else:
                solver.add(y[i] <= 0)

    delta = (target_z3 - mass) / len(active)
    return [y[i] + delta if i in active_set else y[i] for i in range(len(y))]


def projection_guard_conditions(y: List[ArithRef], radius: float, trace: dict) -> List[BoolRef]:
    radius_z3 = z3_real(radius)
    mass = Sum(y)
    conditions = []

    if not trace["needed"]:
        conditions.append(mass <= radius_z3)
        return conditions

    support = sorted(trace["support"])
    support_set = set(support)
    tau = (Sum([y[i] for i in support]) - radius_z3) / len(support)
    conditions.append(mass > radius_z3)
    for i in range(len(y)):
        conditions.append(y[i] > tau if i in support_set else y[i] <= tau)
    return conditions


def lift_guard_conditions(y: List[ArithRef], target: float, trace: dict) -> List[BoolRef]:
    target_z3 = z3_real(target)
    mass = Sum(y)
    conditions = []

    if not trace["needed"]:
        conditions.append(mass >= target_z3)
        return conditions

    active = set(trace["active"])
    conditions.append(mass < target_z3)
    if trace.get("fallback", False):
        conditions.extend([coord <= 0 for coord in y])
    else:
        for i in range(len(y)):
            conditions.append(y[i] > 0 if i in active else y[i] <= 0)
    return conditions


def signed_l1_band_norm_guard_conditions(
    x: List[ArithRef],
    half_low: float,
    half_high: float,
    trace: dict,
) -> List[BoolRef]:
    """Return branch guard conditions for a traced Signed L1 BandNorm cell."""
    d = len(x)
    mean_x = Sum(x) / d
    c = [x[i] - mean_x for i in range(d)]

    p = []
    n = []
    conditions = []
    for i, sign in enumerate(trace["signs"]):
        if sign > 0:
            conditions.append(c[i] > 0)
            p.append(c[i])
            n.append(RealVal(0))
        elif sign < 0:
            conditions.append(c[i] < 0)
            p.append(RealVal(0))
            n.append(-c[i])
        else:
            conditions.append(c[i] == 0)
            p.append(RealVal(0))
            n.append(RealVal(0))

    conditions.extend(projection_guard_conditions(p, half_high, trace["pos_projection"]))
    conditions.extend(projection_guard_conditions(n, half_high, trace["neg_projection"]))

    p_projected = encode_nonnegative_l1_projection_with_trace(
        p, half_high, trace["pos_projection"], Solver(), "guard_p_proj"
    )
    n_projected = encode_nonnegative_l1_projection_with_trace(
        n, half_high, trace["neg_projection"], Solver(), "guard_n_proj"
    )
    conditions.extend(lift_guard_conditions(p_projected, half_low, trace["pos_lift"]))
    conditions.extend(lift_guard_conditions(n_projected, half_low, trace["neg_lift"]))

    return conditions


def encode_signed_l1_band_norm_with_trace(
    x: List[ArithRef],
    gamma: List[float],
    beta: List[float],
    half_low: float,
    half_high: float,
    pos_fallback: List[float],
    neg_fallback: List[float],
    trace: dict,
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode Signed L1 BandNorm on certified sign/projection/lift branches."""
    d = len(x)

    mean_x = Sum(x) / d
    c = [x[i] - mean_x for i in range(d)]

    p = []
    n = []
    signs = trace["signs"]
    for i, sign in enumerate(signs):
        if sign > 0:
            solver.add(c[i] > 0)
            p.append(c[i])
            n.append(RealVal(0))
        elif sign < 0:
            solver.add(c[i] < 0)
            p.append(RealVal(0))
            n.append(-c[i])
        else:
            solver.add(c[i] == 0)
            p.append(RealVal(0))
            n.append(RealVal(0))

    p_projected = encode_nonnegative_l1_projection_with_trace(
        p, half_high, trace["pos_projection"], solver, f"{ctx_prefix}_p_proj"
    )
    n_projected = encode_nonnegative_l1_projection_with_trace(
        n, half_high, trace["neg_projection"], solver, f"{ctx_prefix}_n_proj"
    )

    p_normalized = encode_additive_lift_with_trace(
        p_projected, half_low, pos_fallback, trace["pos_lift"], solver, f"{ctx_prefix}_p_lift"
    )
    n_normalized = encode_additive_lift_with_trace(
        n_projected, half_low, neg_fallback, trace["neg_lift"], solver, f"{ctx_prefix}_n_lift"
    )

    z = [p_normalized[i] - n_normalized[i] for i in range(d)]
    mean_z = Sum(z) / d
    z_recentered = [z[i] - mean_z for i in range(d)]

    return [z3_real(gamma[i]) * z_recentered[i] + z3_real(beta[i]) for i in range(d)]


def encode_multihead_attention_sparsemax(
    query: List[ArithRef],
    keys: List[List[ArithRef]],
    values: List[List[ArithRef]],
    n_heads: int,
    solver: Solver,
    ctx_prefix: str,
    trace: dict = None,
) -> List[ArithRef]:
    """Encode multi-head attention with sparsemax weighting.

    Args:
        query: Query vector [d_model]
        keys: Key vectors [seq_len, d_model]
        values: Value vectors [seq_len, d_model]
        n_heads: Number of attention heads
        solver: Z3 solver
        ctx_prefix: Prefix for auxiliary variables

    Returns:
        Attention output [d_model]
    """
    seq_len = len(keys)
    d_model = len(query)
    head_dim = d_model // n_heads

    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

    # Scale factor for attention scores
    scale = z3_real(1.0 / (head_dim ** 0.5))

    head_outputs = []

    for h in range(n_heads):
        # Extract head slice
        head_start = h * head_dim
        head_end = (h + 1) * head_dim

        q_h = query[head_start:head_end]
        k_h = [keys[i][head_start:head_end] for i in range(seq_len)]
        v_h = [values[i][head_start:head_end] for i in range(seq_len)]

        # Compute attention scores: score_i = (query · key_i) / sqrt(head_dim)
        scores = []
        for i in range(seq_len):
            score = Sum([q_h[j] * k_h[i][j] for j in range(head_dim)]) * scale
            scores.append(score)

        # Apply sparsemax to get attention weights
        sparsemax_ctx = f"{ctx_prefix}_h{h}"
        if trace is not None:
            weights = encode_sparsemax_with_support(
                scores,
                trace["sparsemax"][sparsemax_ctx],
                solver,
                sparsemax_ctx,
            )
        else:
            weights = encode_sparsemax(scores, solver, sparsemax_ctx)

        # Weighted sum of values
        output_h = []
        for j in range(head_dim):
            out_j = Sum([weights[i] * v_h[i][j] for i in range(seq_len)])
            output_h.append(out_j)

        head_outputs.extend(output_h)

    return head_outputs


def encode_mlp(
    x: List[ArithRef],
    W_up: List[List[float]],
    b_up: List[float],
    W_down: List[List[float]],
    b_down: List[float],
) -> List[ArithRef]:
    """Encode MLP: x -> LeakyReLU(W_up @ x + b_up) -> W_down @ ... + b_down.

    Args:
        x: Input vector [d_model]
        W_up: Up-projection weight [d_ff, d_model]
        b_up: Up-projection bias [d_ff]
        W_down: Down-projection weight [d_model, d_ff]
        b_down: Down-projection bias [d_model]

    Returns:
        MLP output [d_model]
    """
    d_model = len(x)
    d_ff = len(b_up)

    # Up-projection + activation
    hidden = []
    for i in range(d_ff):
        h_i = Sum([z3_real(W_up[i][j]) * x[j] for j in range(d_model)]) + z3_real(b_up[i])
        hidden.append(encode_leaky_relu(h_i))

    # Down-projection
    output = []
    for i in range(d_model):
        out_i = Sum([z3_real(W_down[i][j]) * hidden[j] for j in range(d_ff)]) + z3_real(b_down[i])
        output.append(out_i)

    return output


def encode_mlp_with_trace(
    x: List[ArithRef],
    W_up: List[List[float]],
    b_up: List[float],
    W_down: List[List[float]],
    b_down: List[float],
    trace: dict,
    solver: Solver,
    ctx_prefix: str,
) -> List[ArithRef]:
    """Encode MLP using certified LeakyReLU sign branches."""
    d_model = len(x)
    d_ff = len(b_up)

    hidden = []
    for i in range(d_ff):
        h_i = Sum([z3_real(W_up[i][j]) * x[j] for j in range(d_model)]) + z3_real(b_up[i])
        is_nonnegative = trace[f"{ctx_prefix}_hidden_{i}"]
        hidden.append(encode_leaky_relu_with_trace(h_i, is_nonnegative, solver, f"{ctx_prefix}_hidden_{i}"))

    output = []
    for i in range(d_model):
        out_i = Sum([z3_real(W_down[i][j]) * hidden[j] for j in range(d_ff)]) + z3_real(b_down[i])
        output.append(out_i)

    return output
