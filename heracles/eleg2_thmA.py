"""eleg2_thmA.py -- verification of THEOREM A (ELEGANT2.md SS2).

THEOREM A.  N = n+1 atoms (c_l, v_lj), h(u) = sum_l c_l prod_j (1 + u_j v_lj),
            hmin := min over the torus T^n of |h(u)|.  Then for every r in (0,1]

                sum_l |c_l|^r  >=  (n+1)^{1-r} hmin^r ,

            and since |h| >= Re h, also with hmin replaced by (min Re h)_+.

Checks, all fp64 (mpmath cross-check on a sample):
  (V1) the chain identity  log[(n+1)^{n+1} prod|c_l|] = (n+1) H + Delta   (P2 SS2.1)
  (V2) Delta >= 0 (Hadamard)
  (V3) |h| at every kill point >= hmin      (this is what the proof uses)
  (V4) the theorem's ratio, random populations, n = 1..6
  (V5) the theorem's ratio, ADVERSARIAL (Nelder-Mead descent on the ratio)
  (V6) hmin computed two independent ways (coordinate descent vs dense torus grid)
  (V7) equality only at the block: structure of the adversarial minimisers
  (V8) COLLAPSE: if some c_l = 0, or two atoms share a slot phase, then hmin = 0

Run:  uv run --with scipy python eleg2_thmA.py
"""
import json
import os
import sys
import numpy as np
from itertools import permutations, product
from math import log, comb
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import rand_pop, min_re_h, h_at

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)
RS = [0.05, 0.2, 1/3, 0.5, 2/3, 0.75, 0.9, 0.99, 1.0]


# ---------------------------------------------------------------- |h| minimum

def min_abs_h(c, v, starts=30, rng=None, iters=60):
    """min over T^n of |h(u)|, by cyclic coordinate descent.

    For fixed u_{-j}, h = a + b u_j, so min over |u_j| = 1 of |h| is
    | |a| - |b| |, attained at u_j = -(a/|a|)(conj(b)/|b|).
    """
    N, n = v.shape
    if rng is None:
        rng = np.random.default_rng(0)
    best = np.inf
    for s in range(starts):
        if s == 0:
            u = np.ones(n, dtype=complex)
        elif s <= N:
            u = -np.conj(v[(s - 1) % N])
        else:
            u = np.exp(2j * np.pi * rng.random(n))
        val = abs(h_at(c, v, u))
        for _ in range(iters):
            prev = val
            for j in range(n):
                fac = 1.0 + u[None, :] * v
                fac[:, j] = 1.0
                pr = np.prod(fac, axis=1)
                a = np.sum(c * pr)
                b = np.sum(c * v[:, j] * pr)
                if abs(b) < 1e-300 or abs(a) < 1e-300:
                    val = abs(a + b * u[j])
                    continue
                u[j] = -(a / abs(a)) * (np.conj(b) / abs(b))
                val = abs(abs(a) - abs(b))
            if prev - val < 1e-16:
                break
        best = min(best, val)
    return best


def min_abs_h_grid(c, v, G=40):
    N, n = v.shape
    th = 2 * np.pi * np.arange(G) / G
    best = np.inf
    for idx in product(range(G), repeat=n):
        u = np.exp(1j * th[list(idx)])
        best = min(best, abs(h_at(c, v, u)))
    return best


# ---------------------------------------------------------------- chain identity

def chain_parts(c, v):
    N, n = v.shape
    D = n + 1
    assert N == D
    vals = []
    for l0 in range(N):
        rest = [l for l in range(N) if l != l0]
        for sg in permutations(rest):
            u = np.array([-np.conj(v[sg[j], j]) for j in range(n)])
            vals.append(abs(h_at(c, v, u)))
    H = float(np.mean([log(x + 1e-300) for x in vals]))
    logV2 = 0.0
    for j in range(n):
        z = v[:, j]
        s = 0.0
        for a in range(N):
            for b in range(a + 1, N):
                s += 2 * log(abs(z[a] - z[b]) + 1e-300)
        logV2 += s
    Delta = D * log(D) - logV2 / n
    lhs = D * log(D) + float(np.sum(np.log(np.abs(c) + 1e-300)))
    return lhs, H, Delta, vals


# ---------------------------------------------------------------- adversary

def ratio(x, n, r, use_abs, starts, rng):
    D = n + 1
    g, t, b = x[:D], x[D:2 * D], x[2 * D:].reshape(D, n)
    c = np.exp(g) * np.exp(1j * t)
    v = np.exp(1j * b)
    if use_abs:
        mm = min_abs_h(c, v, starts=starts, rng=rng)
    else:
        mm, _ = min_re_h(c, v, 1.0, starts=starts, rng=rng)
        mm = max(mm, 0.0)
    if mm <= 1e-13:
        return 1e6
    return float(np.sum(np.abs(c) ** r) / (D ** (1 - r) * mm ** r))


def adversary(n, r, use_abs, restarts=10, starts=6, seed=0):
    rng = np.random.default_rng(seed)
    D = n + 1
    dim = 2 * D + D * n
    best, bx = np.inf, None
    for t in range(restarts):
        if t % 2 == 0:
            g = -np.log(D) * np.ones(D)
            th = np.zeros(D)
            bb = np.array([[-2 * np.pi * l / D] * n for l in range(D)], dtype=float)
            x0 = np.concatenate([g, th, bb.ravel()])
            x0 = x0 + (0.05 + 0.6 * rng.random()) * rng.normal(size=dim)
        else:
            x0 = np.concatenate([0.4 * rng.normal(size=D), 2 * np.pi * rng.random(D),
                                 2 * np.pi * rng.random(D * n)])
        res = minimize(lambda z: ratio(z, n, r, use_abs, starts, rng), x0,
                       method="Nelder-Mead",
                       options=dict(maxiter=400 + 60 * dim, maxfev=400 + 60 * dim, xatol=1e-9, fatol=1e-12))
        if res.fun < best:
            best, bx = float(res.fun), res.x.copy()
    g, th, bb = bx[:D], bx[D:2 * D], bx[2 * D:].reshape(D, n)
    c = np.exp(g) * np.exp(1j * th)
    v = np.exp(1j * bb)
    return best, c, v


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    rec = {}

    # ---- V1, V2, V3, V4, V6, V8
    w1 = w2 = 0.0
    w3 = np.inf
    w4 = np.inf
    w6 = 0.0
    w8 = 0.0
    for n in range(1, 6):
        D = n + 1
        for kind in ["generic", "junkfree", "nearjf"]:
            for t in range(30 if n <= 3 else (12 if n == 4 else 5)):
                c, v = rand_pop(rng, D, n, kind)
                c = c / max(np.max(np.abs(c)), 1e-30)
                lhs, H, Delta, vals = chain_parts(c, v)
                w1 = max(w1, abs(lhs - (D * H + Delta)))
                w2 = min(w2, Delta) if t else Delta
                hmin = min_abs_h(c, v, starts=18, rng=rng)
                w3 = min(w3, min(vals) - hmin + 1e-15)
                if n <= 2 and t < 6:
                    w6 = max(w6, abs(hmin - min_abs_h_grid(c, v, G=60 if n == 2 else 400)))
                if hmin > 1e-12:
                    for r in RS:
                        w4 = min(w4, float(np.sum(np.abs(c) ** r)
                                           / (D ** (1 - r) * hmin ** r)))
                # V8 collapse
                if t < 5:
                    c0 = c.copy(); c0[0] = 0.0
                    w8 = max(w8, min_abs_h(c0, v, starts=18, rng=rng))
                    v2 = v.copy(); v2[1, :] = v2[0, :]
                    w8 = max(w8, min_abs_h(c, v2, starts=18, rng=rng))
    rec["V1_chain_identity_max_abs_err"] = w1
    rec["V2_min_Delta"] = w2
    rec["V3_min(|h| at kill pts) - hmin  (>=0)"] = w3
    rec["V4_min_ratio_random"] = w4
    rec["V6_hmin_coorddesc_vs_grid_max_err"] = w6
    rec["V8_collapse_max_hmin"] = w8
    for k, vv in rec.items():
        print(f"  {k}: {vv:.3e}" if abs(vv) < 1e-2 or vv > 1e3 else f"  {k}: {vv:.12f}")

    # ---- V5, V7 adversarial
    adv = []
    for n in [1, 2, 3]:
        for r in [1/3, 2/3] if n != 2 else [1/3, 2/3, 0.9]:
            for use_abs in [True, False]:
                b, c, v = adversary(n, r, use_abs, restarts=5, starts=5,
                                    seed=100 * n + int(100 * r) + int(use_abs))
                ac = np.abs(c) / np.abs(c).max()
                ang = (np.angle(v) / (2 * np.pi)) % 1.0
                adv.append(dict(n=n, r=r, use_abs=use_abs, best=b,
                                absc=ac.round(6).tolist(),
                                argv=ang.round(5).tolist()))
                print(f"  ADV n={n} r={r:.3f} {'|h|' if use_abs else 'Re h'}: "
                      f"min ratio = {b:.12f}  |c|/max = {ac.round(4).tolist()}")
    rec["V5_adversarial"] = adv
    with open(os.path.join(OUT, "thmA.json"), "w") as f:
        json.dump(rec, f, indent=1, default=float)
