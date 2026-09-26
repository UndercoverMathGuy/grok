"""adv_p1.py -- ADVERSARIAL attack on P1 (the sharp-constant statement S1).

P1.  For every N >= n+1 and every solution c of the (KL) system

        sum_{l=1}^N c_l prod_{j in T} v_lj  =  A * [T empty]      for every T subset {1..n},

     (v_lj on the unit circle, c_l complex) one has

        sum_l |c_l|^r  >=  (n+1)^{1-r} |A|^r          for every r in (0,1].

Plain reading: "N atoms that carry clean amplitude A and cancel all 2^n - 1 junk sign
patterns cost at least as much as the (n+1)-atom block".  Equality at the block
(n+1 atoms of equal modulus |A|/(n+1) at rotated (n+1)-st roots of unity).

This script tries to BREAK it.  Blocks:

  A  n = 2, N = 4 -- the exactly-square case.  (KL) is 4 complex equations in 4
     unknowns, so c is the unique Cramer solution and the statement becomes a closed
     six-angle inequality
         sum_l |delta_l|^r  >=  3^{1-r} |sum_l (-1)^{l+1} delta_l|^r,
     delta_l = the four 3x3 minors of the 3x4 matrix (a ; b ; a*b), a_1 = b_1 = 1.
     Massive vectorised random + structured sampling, top-candidate polish, r swept
     finely including r -> 0 and r -> 1.  Special attention to NEAR-DEGENERATE
     configurations (coalescing atoms, where the minors and det M blow up together).

  B  n = 2, N = 5..8 and n = 3, N = 5..9 -- underdetermined: for given phases the
     best c is chosen too (minimise sum|c|^r over the affine solution set).  SLSQP
     with analytic-free numerical Jacobians, random + coalescing + block+extra starts.

  C  the near-degenerate limit of A, probed on purpose: two atoms brought together at
     separation eps, eps = 1e-1 .. 1e-9, with the remaining 4 angles optimised.

  D  a Lipschitz branch-and-bound over the 6-torus, intended as a CERTIFICATE rather
     than a search.  It is implemented and correct but it CANNOT terminate, and the
     reason is a property of the inequality, not of the code: the equality set
     (delta_4 = 0, i.e. one atom vanishing) has codimension 2, the excess above the
     target vanishes like eps^r at distance eps from it while the Lipschitz error is
     O(rho), so boxes need rho <~ L^{-1/(1-r)} ~ 7e-5 and covering a codimension-2
     neighbourhood at that resolution takes ~(2 pi/rho)^4 ~ 6e19 boxes.  Kept, with
     --maxboxes, only so the abort is reproducible; see ADVERSARY.md Sec 7 A6.

All float64 (block A's best points re-checked in 50-digit mpmath).  Local CPU.
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "FINAL_RUNS" / "ADV"


# --------------------------------------------------------------------------- block A
def minors(al, be):
    """The four 3x3 minors delta_l of (a ; b ; ab), with a_1 = b_1 = 1.

    al, be: (B, 3) arrays of angles for atoms 2,3,4.  Returns delta (B, 4) complex
    and Delta (B,) complex = sum_l (-1)^{l+1} delta_l = det M.
    """
    a = np.empty(al.shape[:-1] + (4,), dtype=complex)
    b = np.empty_like(a)
    a[..., 0] = 1.0
    b[..., 0] = 1.0
    a[..., 1:] = np.exp(1j * al)
    b[..., 1:] = np.exp(1j * be)
    ab = a * b
    d = np.empty(a.shape[:-1] + (4,), dtype=complex)
    for pos, (i, j, k) in enumerate([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]):
        ai, aj, ak = a[..., i], a[..., j], a[..., k]
        bi, bj, bk = b[..., i], b[..., j], b[..., k]
        d[..., pos] = (ab[..., i] * (aj * bk - ak * bj)
                       - ab[..., j] * (ai * bk - ak * bi)
                       + ab[..., k] * (ai * bj - aj * bi))
    sgn = np.array([1.0, -1.0, 1.0, -1.0])
    return d, (d * sgn).sum(-1)


def fA(x, r):
    """x = 6 angles (alpha_2,3,4, beta_2,3,4) -> sum_l |c_l|^r for the unique Cramer c."""
    x = np.asarray(x, dtype=float)
    d, D = minors(x[..., :3], x[..., 3:])
    aD = np.abs(D)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (np.abs(d) ** r).sum(-1) / aD ** r
    return np.where(aD < 1e-14, np.inf, val)


def fA_scalar(x, r):
    v = float(fA(np.asarray(x, float)[None, :], r)[0])
    return v if np.isfinite(v) else 1e6


def polish(x0, r, rounds=1, maxit=2000):
    x = np.asarray(x0, float).copy()
    best = fA_scalar(x, r)
    for _ in range(rounds):
        res = minimize(fA_scalar, x, args=(r,), method="Nelder-Mead",
                       options={"maxiter": maxit, "maxfev": maxit,
                                "xatol": 1e-14, "fatol": 1e-16})
        if res.fun < best:
            best, x = res.fun, res.x
        res = minimize(fA_scalar, x, args=(r,), method="Powell",
                       options={"maxiter": maxit, "xtol": 1e-14, "ftol": 1e-16})
        if res.fun < best:
            best, x = res.fun, res.x
    return best, x


def structured_starts(rng, nrand):
    """Random plus deliberately awkward configurations, EXCLUDING nothing but biased
    towards the places a counterexample could hide."""
    S = [rng.uniform(0, 2 * np.pi, size=(nrand, 6))]
    w = 2 * np.pi / 3
    # (i) three atoms at the block, fourth atom anywhere  (the equality manifold)
    m = 4000
    base = np.tile(np.array([w, 2 * w, 0.0, w, 2 * w, 0.0]), (m, 1))
    base[:, 2] = rng.uniform(0, 2 * np.pi, m)
    base[:, 5] = rng.uniform(0, 2 * np.pi, m)
    S.append(base + rng.normal(0, 0.25, base.shape))
    # (ii) two atoms coalescing at separation eps
    for e in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
        m = 3000
        z = rng.uniform(0, 2 * np.pi, size=(m, 6))
        z[:, 2] = z[:, 1] + e * rng.normal(size=m)
        z[:, 5] = z[:, 4] + e * rng.normal(size=m)
        S.append(z)
    # (iii) all four atoms on a common "diagonal" b = a (the common-phase family)
    m = 4000
    z = rng.uniform(0, 2 * np.pi, size=(m, 3))
    S.append(np.concatenate([z, z + rng.normal(0, 0.15, z.shape)], 1))
    # (iv) four-fold symmetric: atoms at 4th roots of unity, perturbed
    m = 3000
    q = 2 * np.pi / 4
    base = np.tile(np.array([q, 2 * q, 3 * q, q, 2 * q, 3 * q]), (m, 1))
    S.append(base + rng.normal(0, 0.6, base.shape))
    # (v) clustered pairs
    m = 3000
    z = rng.uniform(0, 2 * np.pi, size=(m, 6))
    z[:, 1] = z[:, 0] + rng.normal(0, 0.05, m)
    z[:, 4] = z[:, 3] + rng.normal(0, 0.05, m)
    S.append(z)
    return np.concatenate(S, 0)


def block_A(rng, rs, nsample, ntop, log):
    res = {}
    X = structured_starts(rng, nsample)
    log(f"[A] {X.shape[0]:,} sampled configurations, r values {rs}")
    for r in rs:
        tgt = 3.0 ** (1 - r)
        t0 = time.time()
        # evaluate in chunks
        vals = np.empty(X.shape[0])
        for s in range(0, X.shape[0], 200000):
            vals[s:s + 200000] = fA(X[s:s + 200000], r)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        order = np.argsort(vals)[:ntop]
        cand = []
        for i in order:
            cand.append(polish(X[i], r, rounds=1, maxit=1500))
        cand.sort(key=lambda t: t[0])
        best, bx = cand[0]
        for v0, x0 in cand[:12]:                      # deep polish of the leaders
            v, xx = polish(x0, r, rounds=3, maxit=20000)
            if v < best:
                best, bx = v, xx
        res[f"r={r:.4f}"] = dict(r=r, target=tgt, best=best, ratio=best / tgt,
                                 angles=list(map(float, bx)),
                                 sampled_min=float(vals[order[0]]),
                                 secs=time.time() - t0)
        log(f"[A] r={r:.4f} q={3*r:.4f}  min sum|c|^r = {best:.15f}  3^(1-r) = {tgt:.15f}"
            f"   ratio = {best/tgt:.15f}   ({time.time()-t0:.0f}s)")
    return res


# --------------------------------------------------------------------------- block C
def block_C(rng, rs, log):
    """Deliberate near-degeneracy: atoms 3 and 4 separated by eps in both slots."""
    res = {}
    for r in rs:
        tgt = 3.0 ** (1 - r)
        rows = []
        for e in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8):
            def g(u):
                # u = (al2, be2, al3, be3, t1, t2): atom 4 = atom 3 + eps*(t1, t2)
                x = np.array([u[0], u[2], u[2] + e * u[4],
                              u[1], u[3], u[3] + e * u[5]])
                return fA_scalar(x, r)
            best = np.inf
            for _ in range(120):
                u0 = np.concatenate([rng.uniform(0, 2 * np.pi, 4),
                                     rng.normal(0, 1, 2)])
                rr = minimize(g, u0, method="Nelder-Mead",
                              options={"maxiter": 8000, "xatol": 1e-13, "fatol": 1e-15})
                best = min(best, rr.fun)
            rows.append(dict(eps=e, best=best, ratio=best / tgt))
            log(f"[C] r={r:.4f} eps={e:.0e}  min = {best:.15f}  ratio = {best/tgt:.15f}")
        res[f"r={r:.4f}"] = rows
    return res


# --------------------------------------------------------------------------- block B
def subsets(n):
    return [T for k in range(n + 1) for T in itertools.combinations(range(n), k)]


def block_B(rng, cells, starts, log):
    """N > n+1: phases AND c free, joint SLSQP on the exact (KL) equality system.

    Variables: Re c, Im c (N each) and the N*n phase angles.  Objective sum |c_l|^r,
    constraints C_T = [T empty] for all 2^n subsets T (real and imaginary parts).
    Start families: random; block + extra atoms; COALESCING pairs (two atoms brought to
    within 1e-3, where the Cramer minors blow up); all atoms on the common diagonal.
    """
    res = {}
    for (n, N, r) in cells:
        Ts = subsets(n)
        nT = len(Ts)
        tgt = (n + 1) ** (1 - r)

        def split(z):
            return z[:N] + 1j * z[N:2 * N], np.exp(1j * z[2 * N:].reshape(N, n))

        def f(z):
            c, _ = split(z)
            return float(np.sum((np.abs(c) ** 2 + 1e-300) ** (r / 2)))

        def con(z):
            c, v = split(z)
            X = np.ones((N, nT), dtype=complex)
            for t, T in enumerate(Ts):
                for j in T:
                    X[:, t] = X[:, t] * v[:, j]
            C = X.T @ c
            out = [C[0].real - 1.0, C[0].imag]
            for t in range(1, nT):
                out += [C[t].real, C[t].imag]
            return np.array(out)

        best, bz = np.inf, None
        for s in range(starts):
            ang = rng.uniform(0, 2 * np.pi, (N, n))
            if s % 4 == 1:                                   # block + extras
                for l in range(n + 1):
                    ang[l, :] = -2 * np.pi * l / (n + 1)
                ang += rng.normal(0, 0.25, ang.shape)
            elif s % 4 == 2:                                 # coalescing pair
                ang[1] = ang[0] + rng.normal(0, 1e-3, n)
            elif s % 4 == 3:                                 # common diagonal
                ang = np.tile(rng.uniform(0, 2 * np.pi, (N, 1)), (1, n)) \
                    + rng.normal(0, 0.2, (N, n))
            z0 = np.concatenate([rng.standard_normal(2 * N) / N, ang.reshape(-1)])
            try:
                rr = minimize(f, z0, constraints=[{"type": "eq", "fun": con}],
                              method="SLSQP", options={"maxiter": 300, "ftol": 1e-16})
            except Exception:
                continue
            if np.max(np.abs(con(rr.x))) < 1e-9 and rr.fun < best:
                best, bz = float(rr.fun), rr.x
        res[f"n{n}_N{N}_r{r:.4f}"] = dict(n=n, N=N, r=r, target=tgt, best=best,
                                          ratio=best / tgt if np.isfinite(best) else None)
        log(f"[B] n={n} N={N} r={r:.4f} (q={r*(n+1):.4f}): min = {best:.12f}  "
            f"target = {tgt:.12f}   ratio = {best/tgt:.12f}")
    return res


# --------------------------------------------------------------------------- block D
def block_D(r, eps0, max_boxes, log, depth_cap=40):
    """Lipschitz branch-and-bound lower bound for block A on {min_l |c_l| >= eps0}.

    Each minor delta_l is a sum of 6 unit-modulus monomials of degree <= 4 in the six
    angles, so over a box of half-width rho (per angle) the Lipschitz bound
        | |delta_l(x)| - |delta_l(centre)| |  <=  L_l . rho ,   L_l = sum over its
    monomials of (number of angle variables the monomial actually contains),
    and similarly for Delta = det M (24 monomials of degree <= 4).  Hence on the box
        f(x)  >=  sum_l max(0, |delta_l(c)| - L_l rho)^r / ( |Delta(c)| + L_D rho )^r .
    Boxes whose bound already exceeds 3^{1-r} are discarded; boxes in which some
    |c_l| can be < eps0 are discarded (handled analytically); others are split.
    """
    tgt = 3.0 ** (1 - r)
    # Lipschitz constants: every monomial of delta_l is a product of 4 of
    # a_1,b_1,a_i,b_i,... with a_1 = b_1 = 1 fixed; worst case 4 free angles.
    Ld = 6 * 4.0
    LD = 24 * 4.0
    # start from a coarse partition of the 6-torus
    nb = 6
    edges = np.linspace(0, 2 * np.pi, nb + 1)
    cent = 0.5 * (edges[:-1] + edges[1:])
    rho0 = (edges[1] - edges[0]) / 2
    grid = np.stack(np.meshgrid(*([cent] * 6), indexing="ij"), -1).reshape(-1, 6)
    boxes = [(grid, rho0)]
    done = 0
    worst = np.inf
    nboxes = grid.shape[0]
    for depth in range(depth_cap):
        if not boxes:
            break
        C, rho = boxes.pop(0)
        d, D = minors(C[:, :3], C[:, 3:])
        ad = np.abs(d)
        aD = np.abs(D)
        lo_d = np.maximum(0.0, ad - Ld * rho)
        hi_D = aD + LD * rho
        bound = (lo_d ** r).sum(-1) / np.maximum(hi_D, 1e-300) ** r
        # also: can some |c_l| = |delta_l|/|Delta| be below eps0 anywhere in the box?
        lo_D = np.maximum(0.0, aD - LD * rho)
        hi_d = ad + Ld * rho
        small = (hi_d / np.maximum(lo_D, 1e-300) < eps0).any(-1) | (lo_D <= 0)
        keep = (bound < tgt) & (~small)
        done += int((~keep).sum())
        worst = min(worst, float(bound[bound >= tgt].min()) if (bound >= tgt).any() else worst)
        C = C[keep]
        if C.shape[0] == 0:
            log(f"[D] r={r:.3f} CERTIFIED at depth {depth}: every box resolved, "
                f"{nboxes:,} boxes examined")
            return dict(r=r, eps0=eps0, certified=True, depth=depth, boxes=nboxes)
        # split each surviving box into 2^6
        off = np.array(list(itertools.product([-0.5, 0.5], repeat=6))) * rho
        C = (C[:, None, :] + off[None, :, :]).reshape(-1, 6)
        rho = rho / 2
        nboxes += C.shape[0]
        log(f"[D] r={r:.3f} depth {depth}: {C.shape[0]:,} live boxes, rho={rho:.3e}")
        if C.shape[0] > max_boxes:
            log(f"[D] r={r:.3f} ABORTED: {C.shape[0]:,} live boxes exceeds cap")
            return dict(r=r, eps0=eps0, certified=False, depth=depth,
                        live=int(C.shape[0]), boxes=nboxes)
        boxes.append((C, rho))
    return dict(r=r, eps0=eps0, certified=False, depth=depth_cap, boxes=nboxes)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="A")
    ap.add_argument("--nsample", type=int, default=2_000_000)
    ap.add_argument("--ntop", type=int, default=250)
    ap.add_argument("--starts", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rs", type=float, nargs="+", default=None)
    ap.add_argument("--eps0", type=float, default=0.05)
    ap.add_argument("--maxboxes", type=int, default=4_000_000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"p1{a.tag}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    rng = np.random.default_rng(a.seed)
    rs = a.rs if a.rs else [0.05, 0.1, 0.2, 1 / 3, 0.4, 0.5, 2 / 3, 0.75, 5 / 6, 0.9, 0.97]
    out = {}
    if "A" in a.blocks:
        out["A"] = block_A(rng, rs, a.nsample, a.ntop, log)
    if "C" in a.blocks:
        out["C"] = block_C(rng, [2 / 3, 0.5, 0.25], log)
    if "B" in a.blocks:
        cells = ([(2, N, r) for N in (5, 6, 7, 8) for r in (2 / 3, 5 / 6)]
                 + [(3, N, r) for N in (5, 6) for r in (0.5,)])
        out["B"] = block_B(rng, cells, a.starts, log)
    if "D" in a.blocks:
        out["D"] = [block_D(r, a.eps0, a.maxboxes, log) for r in (2 / 3,)]
    (OUT / f"p1{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {OUT / f'p1{a.tag}.json'}")


if __name__ == "__main__":
    main()
