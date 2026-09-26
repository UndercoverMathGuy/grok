"""Exactness of the share-programme solver (core/programme.py)."""
import itertools
import math

import numpy as np
import pytest
from scipy.optimize import minimize

from core import programme as P

S_GRID = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def naive_vertices(p):
    """Independent brute force: every support S and every tight set T of the same size, no symmetry."""
    h = (p - 1) // 2
    A = np.array([[1 - math.cos(2 * math.pi * j * d / p) for j in range(1, h + 1)] for d in range(1, h + 1)])
    out = {}
    for K in range(1, h + 1):
        for S in itertools.combinations(range(h), K):
            for T in itertools.combinations(range(h), K):
                Mx = A[np.ix_(T, S)]
                if abs(np.linalg.det(Mx)) < 1e-12:
                    continue
                v = np.linalg.solve(Mx, np.ones(K))
                if np.any(v <= 1e-12):
                    continue
                full = np.zeros(h)
                full[list(S)] = v
                if (A @ full).min() < 1 - 1e-9:
                    continue
                out[tuple(np.round(full, 8))] = full
    return np.array(list(out.values()))


@pytest.mark.parametrize("p", [3, 5, 7, 11, 13])
def test_vertex_set_equals_naive_bruteforce(p):
    V = naive_vertices(p)
    reps, sizes, meta = P.load_vertices(p)
    full = P.expand_orbits(reps, p)
    a = {tuple(np.round(v, 7)) for v in V}
    b = {tuple(np.round(v, 7)) for v in full}
    assert a == b, (len(a), len(b))
    assert sizes.sum() == len(full)


@pytest.mark.parametrize("p", [5, 7, 11, 13, 17])
def test_canonical_vs_all_supports_enumeration(p):
    """Canonical-support enumeration + dilation closure == enumeration over ALL supports."""
    V_all, _ = P.enumerate_vertices_raw(p, canonical_only=False)
    reps, sizes, _ = P.load_vertices(p)
    full = P.expand_orbits(reps, p)
    a = {tuple(np.round(v, 7)) for v in V_all}
    b = {tuple(np.round(v, 7)) for v in full}
    assert a == b


@pytest.mark.parametrize("p", [5, 7, 11, 13])
def test_gamma_equals_naive(p):
    V = naive_vertices(p)
    for s in S_GRID:
        c = np.array([(v[v > 0] ** (1 / s)).sum() for v in V])
        g_naive = c.min() ** (-s)
        assert abs(P.gamma(p, s) - g_naive) <= 1e-12 * g_naive


@pytest.mark.parametrize("p", [5, 7])
def test_gamma_vs_dense_grid(p):
    """Independent check: a dense grid over budget shares never beats the exact value and gets close."""
    h = (p - 1) // 2
    A = P.coverage_matrix(p)
    n = 200 if h == 2 else 60
    for s in [1.0, 1.5, 2.0, 2.5]:
        best = 0.0
        for c in itertools.product(range(n + 1), repeat=h - 1):
            if sum(c) > n:
                continue
            w = np.array(list(c) + [n - sum(c)]) / n
            y = w ** s
            best = max(best, (A @ y).min())
        g = P.gamma(p, s)
        assert best <= g * (1 + 1e-12)
        assert best >= g * (1 - 2e-2)


def _multistart(p, s, n_starts, rng):
    """scipy SLSQP: maximise t s.t. A y >= t, sum y^{1/s} <= 1, y >= 0, from random starts."""
    h = (p - 1) // 2
    A = P.coverage_matrix(p)
    best = 0.0
    for _ in range(n_starts):
        w = rng.dirichlet(np.full(h, rng.choice([0.2, 1.0, 5.0])))
        y0 = w ** s
        x0 = np.concatenate([y0, [(A @ y0).min()]])
        cons = [{"type": "ineq", "fun": lambda x: A @ x[:h] - x[h]},
                {"type": "ineq", "fun": lambda x: 1 - (np.clip(x[:h], 0, None) ** (1 / s)).sum()}]
        res = minimize(lambda x: -x[h], x0, method="SLSQP", constraints=cons,
                       bounds=[(0, None)] * h + [(None, None)], options={"maxiter": 500, "ftol": 1e-12})
        y = np.clip(res.x[:h], 0, None)
        val = P.programme_value(y, p, s)  # re-evaluated exactly on the budget
        best = max(best, val)
    return best


@pytest.mark.parametrize("p", [11, 13, 17, 19, 23])
def test_multistart_never_beats_exact(p):
    rng = np.random.default_rng(p)
    for s in [1.0, 1.2, 1.5, 2.0, 2.5]:
        ms = _multistart(p, s, 25, rng)
        g = P.gamma(p, s)
        assert ms <= g * (1 + 1e-9), (p, s, ms, g)


def test_paper_p5_example():
    """approach.tex: cov_1 = (0.69, 1.81, 1.81, 0.69), cov_2 = (1.81, 0.69, 0.69, 1.81) for delta = 1..4;
    s = 1: equal split covers every offset by 1.25 and both frequencies are used;
    s = 2: equal split (1/4, 1/4) covers by 0.625 < 0.69, one frequency is used."""
    p = 5
    cov = lambda j, d: 1 - math.cos(2 * math.pi * j * d / p)
    assert np.allclose([cov(1, d) for d in range(1, 5)], [0.69, 1.81, 1.81, 0.69], atol=0.005)
    assert np.allclose([cov(2, d) for d in range(1, 5)], [1.81, 0.69, 0.69, 1.81], atol=0.005)
    s1 = P.solve(5, 1.0)
    assert abs(s1.gamma - 1.25) < 1e-12 and s1.kstar == 2
    assert np.allclose(s1.y_all, [[0.5, 0.5]])
    assert abs(P.programme_value(np.array([0.25, 0.25]), 5, 2.0) - 0.625) < 1e-12
    s2 = P.solve(5, 2.0)
    assert s2.kstar == 1
    assert abs(s2.gamma - (1 - math.cos(2 * math.pi / 5))) < 1e-12
    assert s2.gamma > 0.625


@pytest.mark.parametrize("p", [5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
def test_dense_endpoint(p):
    """cor:dense: Gamma_1(p) = p/(p-1), unique maximiser y_j = 2/(p-1)."""
    sol = P.solve(p, 1.0)
    assert abs(sol.gamma - p / (p - 1)) < 1e-12
    assert len(sol.y_all) == 1 and np.allclose(sol.y_all[0], 2 / (p - 1))


@pytest.mark.parametrize("p", [11, 13, 17, 23])
def test_dilation_invariance(p):
    rng = np.random.default_rng(0)
    h = (p - 1) // 2
    for s in [1.3, 1.5, 2.0]:
        for _ in range(5):
            y = rng.random(h) * (rng.random(h) < 0.5)
            if not y.any():
                y[0] = 1
            for u in range(1, h + 1):
                assert abs(P.programme_value(P.dilate(y, u, p), p, s) - P.programme_value(y, p, s)) < 1e-12
            S = [int(j) + 1 for j in np.flatnonzero(rng.random(h) < 0.4)] or [1]
            g0 = P.gamma_restricted(p, S, s)[0]
            for u in range(2, h + 1):
                Su = sorted(set(int(P.fold(u * j, p)) for j in S))
                assert abs(P.gamma_restricted(p, Su, s)[0] - g0) < 1e-12 * g0
        # the optimal set is closed under dilation
        sol = P.solve(p, s)
        keys = {tuple(np.round(y, 9)) for y in sol.y_all}
        for y in sol.y_all:
            for u in range(1, h + 1):
                assert tuple(np.round(P.dilate(y, u, p), 9)) in keys


@pytest.mark.parametrize("p", [11, 13, 17])
def test_gamma_restricted_cache_vs_direct(p):
    rng = np.random.default_rng(1)
    h = (p - 1) // 2
    for s in [1.2, 1.5, 2.2]:
        for _ in range(4):
            S = sorted(rng.choice(np.arange(1, h + 1), size=int(rng.integers(1, min(h, 5) + 1)), replace=False).tolist())
            a = P.gamma_restricted(p, S, s, use_cache=True)[0]
            b = P.gamma_restricted(p, S, s, use_cache=False)[0]
            assert abs(a - b) < 1e-11 * a
    assert abs(P.gamma_restricted(p, list(range(1, h + 1)), 1.5)[0] - P.gamma(p, 1.5)) < 1e-12


@pytest.mark.parametrize("p", [13, 23])
def test_switch_points_consistent(p):
    sw = P.switch_points(p, 1.0, 4.0)
    reps, _, _ = P.load_vertices(p)
    pls = sw["plateaus"]
    # inside each plateau, the exact optimum is the plateau's vertex class (up to dilation)
    for pl in pls:
        s_mid = 0.5 * (pl["s_from"] + pl["s_to"])
        sol = P.solve(p, s_mid)
        assert abs(P.programme_value(pl["rep"], p, s_mid) - sol.gamma) < 1e-10 * sol.gamma
    # at each switch point the two neighbouring classes have equal cost (root found)
    for a, b, s_star in zip(pls[:-1], pls[1:], sw["switches"]):
        va = P.programme_value(a["rep"], p, s_star)
        vb = P.programme_value(b["rep"], p, s_star)
        assert abs(va - vb) < 1e-9 * va


@pytest.mark.parametrize("p,s", [(23, 1.5), (29, 2.0), (31, 1.25), (31, 2.0), (37, 1.5), (37, 2.5)])
def test_best_found_matches_exact_and_kcap_valid(p, s):
    bf = P.best_found(p, s, n_starts=400, seed=0)
    sol = P.solve(p, s)
    assert bf["gamma"] <= sol.gamma * (1 + 1e-9)  # never beats the exact value
    assert abs(bf["gamma"] - sol.gamma) <= 1e-9 * sol.gamma, (bf["gamma"], sol.gamma)
    cap = P.kcap(sol.gamma, s)
    if cap["sharp_int"] is not None:
        assert max(sol.kstar_all) <= cap["sharp_int"] <= cap["paper_int"]


def test_capped_exact_matches_cache():
    for p, s in [(23, 2.0), (29, 2.5), (31, 2.0)]:
        cap = P.kcap(P.best_found(p, s, n_starts=100)["gamma"], s)
        reps, sizes = P.exact_capped(p, cap["sharp_int"])
        c = P.cost(reps, s)
        assert abs(c.min() ** (-s) - P.gamma(p, s)) < 1e-12 * P.gamma(p, s)


def test_cache_checksums():
    for p in P.cached_primes():
        reps, sizes, meta = P.load_vertices(p)  # raises on checksum mismatch
        assert meta["n_orbit_reps"] == len(reps)
