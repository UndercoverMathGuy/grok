r"""P1 (sharp constant), first open case n = 2, N = 4: shared algebra.

Setting (inherited, see GOPT_STATE.md / GLOBAL_OPT4.md sec 4.1).
Four atoms l = 1..4 carry complex amplitudes c_l; atom l has two unit-modulus
"slot phases" (a_l, b_l) in the torus T^2.  The junk-cancellation system is

    (KL)   sum_l c_l = A,   sum_l c_l a_l = 0,   sum_l c_l b_l = 0,
           sum_l c_l a_l b_l = 0.

Four complex equations, four complex unknowns: c is the unique Cramer solution
for a generic configuration.  The open claim (S1 at n = 2, N = 4) is

    sum_l |c_l|^r  >=  3^{1-r} |A|^r,      r = q/nu in (0, 1].

THE REFORMULATION USED HERE (derived and verified in this file).
Write D_(l) for the 3x3 determinant of the complementary triple,

    D_(l) := det [[1,1,1],[a_m,a_n,a_o],[b_m,b_n,b_o]],   {m<n<o} = {1..4} \ {l},

("the signed area of the triangle the other three atoms make in C^2") and put

    e_l := (-1)^(l+1) D_(l).

Then (Prop A, verified numerically below, proved in P1A.md):

    |c_l| = |A| |D_(l)| / |det M|,       M = [[1],[a],[b],[ab]] (4x4),
    sum_l e_l = 0,  sum_l e_l a_l = 0,  sum_l e_l b_l = 0,
    det M  = -conj( sum_l e_l a_l b_l ) * prod_l (a_l b_l)     (so the moduli agree),

i.e. e is the unique affine dependency of the four points (a_l, b_l) in C^2, and
the claim becomes the purely geometric statement

    (P1-4)   sum_l |D_(l)|^r  >=  3^{1-r} | sum_l e_l a_l b_l |^r .

Plain reading: take four points of the torus T^2 = {|a|=|b|=1}; they are four
points of the affine plane C^2, so they carry a unique (up to scale) linear
dependency e with sum e = 0 and sum e * point = 0; its coefficients are, up to
sign, the areas of the four complementary triangles.  The number
sum_l e_l a_l b_l measures how badly the quadratic monomial a*b fails to be
affine on those four points.  The claim: the r-th power sum of the four areas is
at least 3^{1-r} times the r-th power of that failure.

Equality manifold (2-dimensional in the 6 gauge-fixed angles, 8 components):
three of the points form the "block" -- a rotated copy of the cube roots of unity
appearing identically in BOTH slots -- and the fourth point is anywhere.

All float64 unless stated.  Local CPU only.
"""

from __future__ import annotations

import itertools
import numpy as np

# ----------------------------------------------------------------------------
# configuration <-> atoms
# ----------------------------------------------------------------------------

TRIPLES = [tuple(t) for t in itertools.combinations(range(4), 3)]  # complements


def points(alpha: np.ndarray, beta: np.ndarray):
    """(a_l, b_l) = (e^{i alpha_l}, e^{i beta_l})."""
    return np.exp(1j * np.asarray(alpha)), np.exp(1j * np.asarray(beta))


def bigM(a, b):
    """The 4x4 Cramer matrix, rows = the multi-affine monomials 1, a, b, ab."""
    return np.array([np.ones(4, dtype=complex), a, b, a * b])


def tri_dets(a, b):
    """D_(l), l = 0..3: determinant of the complementary triple."""
    out = np.empty(4, dtype=complex)
    for l in range(4):
        m, n, o = TRIPLES[3 - l] if False else tuple(j for j in range(4) if j != l)
        out[l] = (
            (a[n] - a[m]) * (b[o] - b[m]) - (a[o] - a[m]) * (b[n] - b[m])
        )
    return out


def dependency(a, b):
    """e_l = (-1)^l D_(l)  (0-indexed sign: (-1)^l, matching (-1)^{l+1} 1-indexed)."""
    D = tri_dets(a, b)
    sg = np.array([1.0, -1.0, 1.0, -1.0])
    return sg * D, D


def atoms_cramer(a, b, A=1.0 + 0j):
    """The unique c solving (KL) (raises if the configuration is degenerate)."""
    M = bigM(a, b)
    rhs = np.array([A, 0, 0, 0], dtype=complex)
    return np.linalg.solve(M, rhs)


def atoms_from_dependency(a, b, A=1.0 + 0j):
    """c_l via the reformulation: c_l = -A * conj(e_l) * conj(a_l b_l) / conj(S) * (-1),
    S := sum_l e_l a_l b_l.  (Same c as atoms_cramer, see verify_identities.)"""
    e, _ = dependency(a, b)
    S = np.sum(e * a * b)
    return A * np.conj(e) * np.conj(a * b) / np.conj(S)


# ----------------------------------------------------------------------------
# objective
# ----------------------------------------------------------------------------


def objective(alpha, beta, r):
    """sum_l |c_l|^r with A = 1, computed in the stable determinant form."""
    a, b = points(alpha, beta)
    e, D = dependency(a, b)
    S = np.abs(np.sum(e * a * b))
    if S == 0.0:
        return np.inf
    return float(np.sum(np.abs(D) ** r) / S**r)


def objective_vec(x, r):
    """x = (alpha_2, alpha_3, alpha_4, beta_2, beta_3, beta_4), gauge alpha_1 = beta_1 = 0."""
    alpha = np.concatenate(([0.0], x[:3]))
    beta = np.concatenate(([0.0], x[3:]))
    return objective(alpha, beta, r)


def target(r):
    return 3.0 ** (1.0 - r)


# ----------------------------------------------------------------------------
# the equality configuration
# ----------------------------------------------------------------------------


def block_config(theta_a=0.0, theta_b=0.0, free=3):
    """Three atoms at the block (cube roots of unity, same in both slots), the
    fourth ("free") at (e^{i theta_a}, e^{i theta_b})."""
    alpha = np.zeros(4)
    beta = np.zeros(4)
    idx = [l for l in range(4) if l != free]
    for j, l in enumerate(idx):
        alpha[l] = 2 * np.pi * j / 3
        beta[l] = 2 * np.pi * j / 3
    alpha[free] = theta_a
    beta[free] = theta_b
    return alpha, beta


# ----------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------


def verify_identities(trials=4000, seed=0):
    rng = np.random.default_rng(seed)
    worst = dict(cramer=0.0, dep0=0.0, depa=0.0, depb=0.0, detM=0.0, mods=0.0)
    for _ in range(trials):
        alpha = rng.uniform(0, 2 * np.pi, 4)
        beta = rng.uniform(0, 2 * np.pi, 4)
        a, b = points(alpha, beta)
        M = bigM(a, b)
        if abs(np.linalg.det(M)) < 1e-6:
            continue
        c = atoms_cramer(a, b)
        e, D = dependency(a, b)
        S = np.sum(e * a * b)
        detM = np.linalg.det(M)
        scale = max(1.0, np.max(np.abs(c)))
        worst["dep0"] = max(worst["dep0"], abs(np.sum(e)) / max(1, np.max(np.abs(e))))
        worst["depa"] = max(worst["depa"], abs(np.sum(e * a)) / max(1, np.max(np.abs(e))))
        worst["depb"] = max(worst["depb"], abs(np.sum(e * b)) / max(1, np.max(np.abs(e))))
        # det M = conj(S) * prod(a b)
        worst["detM"] = max(worst["detM"], abs(detM + np.conj(S) * np.prod(a * b)) / max(1e-12, abs(detM)))
        # |c_l| = |D_(l)| / |det M|
        worst["mods"] = max(worst["mods"], np.max(np.abs(np.abs(c) - np.abs(D) / abs(detM))) / scale)
        # c from the dependency form
        c2 = atoms_from_dependency(a, b)
        worst["cramer"] = max(worst["cramer"], np.max(np.abs(c - c2)) / scale)
    return worst


if __name__ == "__main__":
    w = verify_identities()
    print("identity residuals over random configurations (relative):")
    for k, v in w.items():
        print(f"  {k:8s} {v:.3e}")
    for r in (2.0 / 3.0, 5.0 / 6.0, 0.5):
        vals = []
        rng = np.random.default_rng(1)
        for _ in range(6):
            ta, tb = rng.uniform(0, 2 * np.pi, 2)
            alpha, beta = block_config(ta, tb)
            vals.append(objective(alpha, beta, r) / target(r))
        print(f"  block+free ratio at r={r:.4f}: min {min(vals):.12f} max {max(vals):.12f}")
