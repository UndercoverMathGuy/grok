r"""P1A: soundness checks for the certificate in p1a_cert.py.

Two independent things have to be true for the certificate to mean anything.

CHECK 1 (the relaxation is sound).  My interval lower bound G_lower on a box must
never exceed the adversary's G at any point of that box.  I re-implement G here
in plain float64 straight from `ADVERSARY.md` section 6 (independently of
`adv_proof.py`) and compare against `p1a_cert.G_lower` on small boxes around
randomly sampled points.

CHECK 2 (the chain is sound).  The whole point of G is
`sum_l |c_l|^r >= G(beta, tau)` for every (KL) system.  I re-verify that directly
on genuine (KL) systems -- for n = 2 using the exact Cramer solution of
p1a_core.py at N = 4, and for general (n, N) by solving the constraint system
with least squares plus random null-space components.

Run:  uv run --with mpmath python p1a_certcheck.py
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
from mpmath import iv

import p1a_cert as PC
from p1a_core import atoms_cramer, bigM, points

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1A")


# --------------------------------------------------------------- G in float64


def D_h_float(n, h):
    K = {1: 0.5, 2: 3 * np.sqrt(3.0) / 8, 3: 49 / 72}
    if h == 0:
        return 1.0

    def parts(tot, k):
        if k == 1:
            yield (tot,)
            return
        for first in range(1, tot - k + 2):
            for rest in parts(tot - first, k - 1):
                yield (first,) + rest

    best = 0.0
    for d in parts(n, h):
        v = 1.0
        for dd in d:
            v *= K[dd]
        best = max(best, v)
    return best


def peel_rows_float(n):
    rows, tags = [], []
    for i in range(n - 1):
        z = np.zeros(n)
        z[i], z[i + 1] = 1, -1
        rows.append(z)
        tags.append(("ord", i))
    z = np.zeros(n)
    z[n - 1] = 1
    rows.append(z)
    tags.append(("beta", None))
    for h in range(n):
        z = np.zeros(n)
        z[h:] = 1
        rows.append(z)
        tags.append(("tail", h))
    return np.array(rows), tags


def G_float(n, r, beta, tau):
    """G exactly as in ADVERSARY.md section 6, in float64."""
    tgt = (n + 1) ** (1 - r)
    Ds = [D_h_float(n, h) for h in range(n + 1)]
    A, tags = peel_rows_float(n)
    b = np.array([0.0 if k == "ord" else (beta if k == "beta" else Ds[h] - tau - beta)
                  for k, h in tags])
    best = np.inf
    for S in itertools.combinations(range(len(tags)), n):
        M = A[list(S)]
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        y = np.linalg.solve(M, b[list(S)])
        if (A @ y >= b - 1e-11).all() and (y >= -1e-11).all():
            best = min(best, float((np.maximum(y, 0.0) ** r).sum()))
    if not np.isfinite(best):
        return np.inf
    peel = best + beta ** r
    tail = 0.0 if tau <= 0 else max(tau ** r, tau * max(beta, 1e-300) ** (r - 1))
    return tail + max(peel, tgt * max(0.0, 1 - 2 ** n * tau) ** r)


# --------------------------------------------------------------- CHECK 1


def check_relaxation(n, r, trials=4000, seed=5, half=1e-3):
    rng = np.random.default_rng(seed)
    Ds = [PC.D_h(n, h) for h in range(n + 1)]
    tgt = PC.ipow(iv.mpf(n + 1), 1.0 - r)
    worst = np.inf
    nb = 0
    for _ in range(trials):
        beta = rng.uniform(1e-4, 1.0 / (n + 1))
        tau = rng.uniform(1e-3, 0.45)
        g = G_float(n, r, beta, tau)
        if not np.isfinite(g):
            continue
        bl = max(0.0, beta - half)
        bh = beta + half
        tl = max(1e-9, tau - half)
        th = tau + half
        gl = PC.G_lower(n, r, iv.mpf(bl), iv.mpf(bh), iv.mpf(tl), iv.mpf(th), Ds, tgt)
        nb += 1
        worst = min(worst, g - float(gl.b))
    return worst, nb


# --------------------------------------------------------------- CHECK 2


def kl_systems_n2N4(rng, trials):
    """exact 4-atom (KL) systems at n = 2 via Cramer."""
    out = []
    for _ in range(trials):
        al = rng.uniform(0, 2 * np.pi, 4)
        be = rng.uniform(0, 2 * np.pi, 4)
        a, b = points(al, be)
        if abs(np.linalg.det(bigM(a, b))) < 1e-3:
            continue
        out.append(np.abs(atoms_cramer(a, b)))
    return out


def kl_systems_general(rng, n, N, trials):
    """(KL) systems: least-squares solution plus random null-space components."""
    subsets = [[j for j in range(n) if (mask >> j) & 1] for mask in range(1 << n)]
    out = []
    for _ in range(trials):
        v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        U = np.array([np.prod(v[:, T], axis=1) if T else np.ones(N, dtype=complex)
                      for T in subsets])                      # (2^n, N)
        rhs = np.zeros(len(subsets), dtype=complex)
        rhs[0] = 1.0
        c, *_ = np.linalg.lstsq(U, rhs, rcond=None)
        if np.max(np.abs(U @ c - rhs)) > 1e-8:
            continue
        if N > len(subsets):
            ns = np.linalg.svd(U)[2][len(subsets):].conj().T
            c = c + ns @ (rng.normal(0, 0.4, ns.shape[1]) + 1j * rng.normal(0, 0.4, ns.shape[1]))
            if np.max(np.abs(U @ c - rhs)) > 1e-8:
                continue
        out.append(np.abs(c))
    return out


def check_chain(n, r, profiles):
    worst = np.inf
    for x in profiles:
        xs = np.sort(x)[::-1]
        beta = xs[n] if len(xs) > n else 0.0
        tau = float(xs[n + 1:].sum())
        g = G_float(n, r, beta, tau)
        if not np.isfinite(g):
            continue
        worst = min(worst, float((xs ** r).sum()) / g)
    return worst


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(101)
    rec = {}
    print("CHECK 1 -- is the interval relaxation a genuine lower bound?")
    print("  (reported: min over samples of  G_float(point) - upper end of G_lower(box);")
    print("   must be >= 0, and small = the relaxation is tight)")
    for n, q in ((2, 2.0), (3, 2.0), (2, 2.5)):
        r = q / (n + 1)
        w, nb = check_relaxation(n, r)
        print(f"   n = {n}, q = {q}: min (G - G_lower_upper) = {w:+.6e} over {nb} boxes"
              f"   {'OK' if w >= -1e-12 else '<-- UNSOUND'}")
        rec[f"relax_n{n}_q{q}"] = dict(worst=float(w), boxes=int(nb))

    print("\nCHECK 2 -- is  sum_l |c_l|^r >= G(beta, tau)  on genuine (KL) systems?")
    print("  (reported: min over systems of  sum|c|^r / G ; must be >= 1)")
    for n, q, Ns in ((2, 2.0, [4, 5, 6, 7]), (3, 2.0, [4, 5, 6])):
        r = q / (n + 1)
        rows = []
        for N in Ns:
            if n == 2 and N == 4:
                profs = kl_systems_n2N4(rng, 3000)
            else:
                profs = kl_systems_general(rng, n, N, 1500)
            if not profs:
                continue
            w = check_chain(n, r, profs)
            print(f"   n = {n}, q = {q}, N = {N}: min sum|c|^r / G = {w:.9f}"
                  f"   over {len(profs)} systems   {'OK' if w >= 1 - 1e-9 else '<-- VIOLATED'}")
            rows.append(dict(N=N, ratio=float(w), systems=len(profs)))
        rec[f"chain_n{n}_q{q}"] = rows

    with open(os.path.join(OUT, "certcheck.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, 'certcheck.json')}")


if __name__ == "__main__":
    main()
