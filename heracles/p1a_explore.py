r"""P1A exploration: global minima, hidden symmetries, and the shape of the
objective near its equality set, for S1 at n = 2, N = 4.

Objective (see p1a_core.py):   Phi(alpha, beta; r) = sum_l |c_l|^r  with |A| = 1,
computed as  sum_l |D_(l)|^r / |det M|^r.   Target: 3^{1-r}.

Blocks
  --sym     : test candidate symmetries of Phi (permutations, conjugation, slot
              swap, and the "monomial rotation" (a,b) -> (conj(b), a conj(b))
              derived in P1A.md).
  --global  : multistart global minimisation over the 6 gauge-fixed angles.
  --floor   : R(t) := min Phi subject to min_l |c_l| >= t  -- how much room a
              branch-and-bound has once the small-atom neighbourhood is removed
              by Lemma R.
  --degen   : the degenerate stratum b_l = a_l (common phase): there det M = 0,
              (KL) has a one-parameter solution family, and Phi must be
              minimised over that family too.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
from scipy.optimize import minimize

from p1a_core import (
    atoms_cramer,
    bigM,
    dependency,
    objective,
    objective_vec,
    points,
    target,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1A")


# ----------------------------------------------------------------------------
# symmetries
# ----------------------------------------------------------------------------


def sym_maps():
    """Candidate maps on configurations (a, b) -> (a', b'), as functions."""
    return {
        "identity": lambda a, b: (a, b),
        "conjugate": lambda a, b: (np.conj(a), np.conj(b)),
        "slot_swap": lambda a, b: (b, a),
        "monomial_rot": lambda a, b: (np.conj(b), a * np.conj(b)),
        "monomial_rot2": lambda a, b: (np.conj(a * np.conj(b)), np.conj(b) * (a * np.conj(b))),
        "slot1_conj": lambda a, b: (np.conj(a), b),
        "shift_a": lambda a, b: (a * np.exp(0.7j), b),
    }


def test_symmetries(trials=400, r=2.0 / 3.0, seed=3):
    rng = np.random.default_rng(seed)
    names = list(sym_maps())
    worst = {k: 0.0 for k in names}
    perm_worst = 0.0
    for _ in range(trials):
        alpha = rng.uniform(0, 2 * np.pi, 4)
        beta = rng.uniform(0, 2 * np.pi, 4)
        a, b = points(alpha, beta)
        if abs(np.linalg.det(bigM(a, b))) < 1e-3:
            continue
        base = obj_ab(a, b, r)
        for k, m in sym_maps().items():
            a2, b2 = m(a, b)
            worst[k] = max(worst[k], abs(obj_ab(a2, b2, r) - base) / base)
        for pi in itertools.permutations(range(4)):
            perm_worst = max(perm_worst, abs(obj_ab(a[list(pi)], b[list(pi)], r) - base) / base)
    return worst, perm_worst


def obj_ab(a, b, r):
    e, D = dependency(a, b)
    S = abs(np.sum(e * a * b))
    if S < 1e-300:
        return np.inf
    return float(np.sum(np.abs(D) ** r) / S**r)


def monomial_rot_order(seed=5):
    """How many times must (a,b) -> (conj(b), a conj(b)) be applied to return?"""
    rng = np.random.default_rng(seed)
    a, b = points(rng.uniform(0, 2 * np.pi, 4), rng.uniform(0, 2 * np.pi, 4))
    a0, b0 = a.copy(), b.copy()
    f = sym_maps()["monomial_rot"]
    for k in range(1, 13):
        a, b = f(a, b)
        if np.max(np.abs(a - a0)) < 1e-9 and np.max(np.abs(b - b0)) < 1e-9:
            return k
    return None


# ----------------------------------------------------------------------------
# global minimisation
# ----------------------------------------------------------------------------


def global_min(r, starts=4000, seed=0, polish=True):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    vals = []
    for _ in range(starts):
        x0 = rng.uniform(0, 2 * np.pi, 6)
        res = minimize(objective_vec, x0, args=(r,), method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-10, fatol=1e-13))
        v = res.fun
        if polish:
            res2 = minimize(objective_vec, res.x, args=(r,), method="Nelder-Mead",
                            options=dict(maxiter=6000, xatol=1e-12, fatol=1e-15))
            if res2.fun < v:
                v, res = res2.fun, res2
        vals.append(v)
        if v < best[0]:
            best = (v, res.x.copy())
    return best, np.array(vals)


def atom_profile(x, r):
    alpha = np.concatenate(([0.0], x[:3]))
    beta = np.concatenate(([0.0], x[3:]))
    a, b = points(alpha, beta)
    c = atoms_cramer(a, b)
    return np.sort(np.abs(c))[::-1]


# ----------------------------------------------------------------------------
# the floor R(t): min Phi subject to min_l |c_l| >= t
# ----------------------------------------------------------------------------


def moduli(x):
    alpha = np.concatenate(([0.0], x[:3]))
    beta = np.concatenate(([0.0], x[3:]))
    a, b = points(alpha, beta)
    e, D = dependency(a, b)
    S = abs(np.sum(e * a * b))
    if S < 1e-14:
        return np.full(4, np.nan)
    return np.abs(D) / S


def floor_at(t, r, starts=600, seed=1):
    """min over configurations of sum |c|^r subject to min_l |c_l| >= t."""
    rng = np.random.default_rng(seed)
    cons = [{"type": "ineq", "fun": lambda x, t=t: moduli(x) - t}]
    best = (np.inf, None)
    for _ in range(starts):
        x0 = rng.uniform(0, 2 * np.pi, 6)
        m = moduli(x0)
        if not np.all(np.isfinite(m)):
            continue
        try:
            res = minimize(objective_vec, x0, args=(r,), method="SLSQP",
                           constraints=cons, options=dict(maxiter=400, ftol=1e-12))
        except Exception:
            continue
        if not res.success:
            continue
        m = moduli(res.x)
        if not np.all(np.isfinite(m)) or np.min(m) < t - 1e-7:
            continue
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


def lemma_R_bound(t, r):
    """The proved bound of Lemma R: t^r + 3^{1-r} (1 - 4t)^r  (0 if 4t >= 1)."""
    if 4 * t >= 1:
        return t ** r
    return t ** r + 3.0 ** (1.0 - r) * (1.0 - 4.0 * t) ** r


def lemma_R_threshold(r):
    """Largest t with  t^r + 3^{1-r}(1-4t)^r >= 3^{1-r}  for all smaller t."""
    lo, hi = 1e-9, 0.25
    # g is positive just above 0, then crosses zero once.
    g = lambda t: lemma_R_bound(t, r) - 3.0 ** (1.0 - r)
    # find the crossing above the interior max
    ts = np.linspace(1e-6, 0.2499, 200000)
    gs = np.array([g(t) for t in ts])
    idx = np.where(gs < 0)[0]
    if len(idx) == 0:
        return 0.25
    i = idx[0]
    lo, hi = ts[i - 1], ts[i]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------
# degenerate stratum b_l = a_l  (common phase): 1-parameter solution family
# ----------------------------------------------------------------------------


def degen_min(r, starts=3000, seed=2):
    """min over 4 nodes on the circle and over the solution family of sum|c|^r,
    where sum_l c_l a_l^t = [t=0] for t = 0,1,2."""
    rng = np.random.default_rng(seed)

    def pack(z):
        ang = np.concatenate(([0.0], z[:3]))
        a = np.exp(1j * ang)
        V = np.array([np.ones(4, dtype=complex), a, a * a])
        # particular solution + null direction
        c0, *_ = np.linalg.lstsq(V, np.array([1.0, 0.0, 0.0], dtype=complex), rcond=None)
        # null vector: nu_l = 1/prod_{m!=l}(a_l - a_m)
        nu = np.array([1.0 / np.prod([a[l] - a[m] for m in range(4) if m != l]) for l in range(4)])
        lam = z[3] + 1j * z[4]
        return c0 + lam * nu

    def f(z):
        c = pack(z)
        return float(np.sum(np.abs(c) ** r))

    best = (np.inf, None)
    for _ in range(starts):
        z0 = np.concatenate((rng.uniform(0, 2 * np.pi, 3), rng.normal(0, 0.5, 2)))
        try:
            res = minimize(f, z0, method="Nelder-Mead",
                           options=dict(maxiter=5000, xatol=1e-11, fatol=1e-14))
        except Exception:
            continue
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best, pack(best[1])


# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=["sym", "global", "floor", "degen"])
    ap.add_argument("--starts", type=int, default=1500)
    ap.add_argument("--r", type=float, nargs="*", default=[2.0 / 3.0])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rec = {}

    if "sym" in a.blocks:
        w, pw = test_symmetries()
        print("symmetry test (relative change of Phi; 0 = exact symmetry):")
        for k, v in w.items():
            print(f"  {k:14s} {v:.3e}")
        print(f"  {'permutations':14s} {pw:.3e}")
        print(f"  order of monomial_rot: {monomial_rot_order()}")
        rec["sym"] = {k: float(v) for k, v in w.items()}
        rec["sym"]["permutations"] = float(pw)
        rec["monomial_rot_order"] = monomial_rot_order()

    for r in a.r:
        tg = target(r)
        if "global" in a.blocks:
            (v, x), vals = global_min(r, starts=a.starts, seed=11)
            prof = atom_profile(x, r)
            print(f"\nr = {r:.6f}  target 3^(1-r) = {tg:.12f}")
            print(f"  global min      = {v:.12f}   ratio {v/tg:.12f}")
            print(f"  atom moduli at argmin: {np.array2string(prof, precision=9)}")
            print(f"  fraction of starts within 1e-6 of target: "
                  f"{np.mean(np.abs(vals - tg) < 1e-6):.3f}")
            # distinct local minima
            uq = np.unique(np.round(vals, 8))
            print(f"  distinct local-min values (rounded 1e-8), smallest 8: "
                  f"{np.array2string(uq[:8], precision=9)}")
            rec[f"global_r{r:.6f}"] = dict(minimum=float(v), ratio=float(v / tg),
                                           profile=[float(p) for p in prof],
                                           local_values=[float(u) for u in uq[:20]])

        if "floor" in a.blocks:
            print(f"\n  floor R(t) = min Phi s.t. min_l |c_l| >= t   (r = {r:.6f})")
            t_star = lemma_R_threshold(r)
            print(f"  Lemma R threshold t* = {t_star:.8f}  (proved: S1 holds for min|c| <= t*)")
            rows = []
            for t in [0.0, 0.005, 0.01, t_star, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.2]:
                if t == 0.0:
                    v = tg
                else:
                    v, _ = floor_at(t, r, starts=max(200, a.starts // 4), seed=7)
                lb = lemma_R_bound(t, r)
                print(f"    t = {t:.6f}   R(t) = {v:.9f}   ratio {v/tg:.9f}"
                      f"   (Lemma R bound {lb/tg:.6f})")
                rows.append(dict(t=float(t), R=float(v), ratio=float(v / tg),
                                 lemmaR=float(lb / tg)))
            rec[f"floor_r{r:.6f}"] = dict(t_star=float(t_star), rows=rows)

        if "degen" in a.blocks:
            (v, z), c = degen_min(r, starts=max(400, a.starts))
            print(f"\n  degenerate stratum b = a, r = {r:.6f}: min = {v:.12f}"
                  f"  ratio {v/tg:.12f}")
            print(f"    |c| = {np.array2string(np.sort(np.abs(c))[::-1], precision=9)}")
            print(f"    nodes = {np.array2string(np.concatenate(([0.0], z[:3])), precision=6)}")
            rec[f"degen_r{r:.6f}"] = dict(minimum=float(v), ratio=float(v / tg),
                                          moduli=[float(m) for m in np.abs(c)])

    with open(os.path.join(OUT, os.environ.get("P1A_TAG","explore")+".json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, 'explore.json')}")


if __name__ == "__main__":
    main()
