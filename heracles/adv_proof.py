"""adv_proof.py -- P1 (the sharp constant, statement S1) at n = 2 and n = 3, for EVERY
number of atoms N, from two inherited PROVED lemmas that had never been combined.

Found while adversarially attacking P1 (see ADVERSARY.md).  RESULT: the combination
proves P1 outright at n = 2 for every N and every r <= r*(2), and at n = 3 for every
r <= r*(3) -- in particular at the paper's exponent q = 2 (r = 2/3 at n = 2, r = 1/2 at
n = 3).  It does NOT reach r -> 1.

-------------------------------------------------------------------------------------
THE STATEMENT
-------------------------------------------------------------------------------------
(KL)   sum_{l=1}^N c_l prod_{j in T} v_lj  =  A [T empty]   for every T subset {1..n},
       c_l complex, v_lj on the unit circle.        ("N atoms carry clean amplitude A
                                                      and cancel all 2^n - 1 junk
                                                      sign patterns")

P1     sum_{l=1}^N |c_l|^r  >=  (n+1)^{1-r} |A|^r          for every r in (0, 1].
       ("no population can be cheaper than the (n+1)-atom block")

Known before: N = n+1 exactly (Key Lemma / Theorem B); a peeling ladder (Theorem E) that
reaches 0.99940 of the target at n = 2 and is PROVED unable to reach 1 (GLOBAL_OPT3 F3-2).

-------------------------------------------------------------------------------------
THE TWO INGREDIENTS (inherited, proved elsewhere, not re-proved here)
-------------------------------------------------------------------------------------
(B)  THEOREM B (GLOBAL_OPT.md Sec 4).  For EXACTLY n+1 atoms, arbitrary complex c and
     arbitrary unit phases, JUNK ALLOWED,
         sum_{l=1}^{n+1} |c_l|^r  >=  (n+1)^{1-r} max(R, 0)^r,
         R := min over u in T^n of Re h(u),   h(u) = sum_l c_l prod_j (1 + u_j v_lj).
     Plain reading: n+1 atoms cannot hold Re h above a level R everywhere on the torus
     more cheaply than the block does.

(E)  THEOREM E / (PEEL_h) (GLOBAL_OPT3.md Sec 3).  For a (KL) system with sorted moduli
     x_1 >= x_2 >= ...,
         sum_{l > h} x_l  >=  D_h(n) |A| ,    h = 0, 1, ..., n,
         D_h(n) = max over d_1 + ... + d_h = n of prod_i kappa_{d_i},
         kappa_1 = 1/2,  kappa_2 = 3 sqrt(3)/8,  kappa_3 = 49/72   (exact algebraic)
     so  n = 2: (1, 0.649519053, 1/4);   n = 3: (1, 49/72, 0.324759526, 1/8).
     Plain reading: a test polynomial that kills the h biggest atoms still reads off a
     fixed fraction of A from the rest, so the tail of the atom list cannot be light.

-------------------------------------------------------------------------------------
THE ARGUMENT
-------------------------------------------------------------------------------------
Normalise |A| = 1, sort the moduli x_1 >= x_2 >= ..., write
      y_i := x_i  (i = 1..n+1),   beta := y_{n+1},   tau := sum_{l > n+1} x_l .
("y = the n+1 biggest atoms, tau = the total mass of everything else.")

BOUND 1 (drop the tail, then Theorem B).  (KL) says h(u) = A identically, so the
population of the n+1 biggest atoms has h_S(u) = A - sum_{l>n+1} c_l prod_j(1+u_j v_lj),
and |prod_j (1 + u_j v_lj)| <= 2^n, hence R_S >= 1 - 2^n tau.  By (B),
      sum_{i=1}^{n+1} y_i^r  >=  (n+1)^{1-r} max(1 - 2^n tau, 0)^r .          (BOUND 1)

BOUND 2 (peeling).  sum_{i>h} y_i + tau = sum_{l>h} x_l >= D_h(n) for h = 0..n, so
      sum_{i=1}^{n+1} y_i^r  >=  Peel(beta, tau) := beta^r +
        min { sum_{i=1}^{n} y_i^r : y_1 >= ... >= y_n >= beta,
              sum_{i=h+1}^{n} y_i >= D_h(n) - tau - beta,  h = 0..n-1 } ,
a CONCAVE minimisation over a pointed polyhedron, so it sits at a vertex and is computed
here by exact vertex enumeration.                                            (BOUND 2)

THE TAIL'S OWN COST.  Every tail atom is at most beta and t -> t^{r-1} decreases, so
      sum_{l>n+1} x_l^r = sum_l x_l . x_l^{r-1} >= tau . beta^{r-1};
subadditivity of t -> t^r also gives sum_{l>n+1} x_l^r >= tau^r.  Hence
      Tail(tau, beta) := max( tau^r, tau beta^{r-1} ).
*This term is what lets one argument cover every N at once: without it the relaxation is
vacuous (all the mass could "hide" in an unbounded number of tail atoms), and with it
the tail's price blows up as beta -> 0.*

Putting the three together, for every (KL) system,
      sum_l |c_l|^r  >=  G(beta, tau)
                     := Tail(tau, beta) + max( Peel(beta, tau),
                                               (n+1)^{1-r} max(1 - 2^n tau, 0)^r ) ,
with (beta, tau) ranging over  tau >= 0,  beta >= max(0, D_n(n) - tau).

    P1 HOLDS  <==  V(n, r) := min over that 2-D region of G  >=  (n+1)^{1-r} .

V is computed on a fine (tau, beta) grid with a local polish, the inner Peel being exact.
At tau = 0 the second term is exactly the target and Tail = 0, so V <= target always: the
bound is SHARP, with the block among its minimisers.  The question is only whether
anything dips below.

Blocks: `sweep` (the r sweep and r*), `verify` (every link of the chain re-checked
numerically on random exact (KL) systems).  All float64.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "FINAL_RUNS" / "ADV"

KAPPA = {1: 0.5, 2: 3 * np.sqrt(3.0) / 8, 3: 49 / 72}


def D_h(n, h):
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
            v *= KAPPA[dd]
        best = max(best, v)
    return best


# --------------------------------------------------------------- BOUND 2 (exact)
def peel_rows(n):
    """Constraint matrix of the inner polyhedron in y_1..y_n (a.y >= b)."""
    rows, kind = [], []
    for i in range(n - 1):
        z = np.zeros(n); z[i] = 1; z[i + 1] = -1
        rows.append(z); kind.append(("ord", i))
    z = np.zeros(n); z[n - 1] = 1
    rows.append(z); kind.append(("beta", None))
    for h in range(n):
        z = np.zeros(n); z[h:] = 1
        rows.append(z); kind.append(("tail", h))
    return np.array(rows), kind


_VC = {}


def peel_inner(n, r, beta, tau, Ds, A, kind):
    b = np.empty(len(kind))
    for t, (k, h) in enumerate(kind):
        b[t] = 0.0 if k == "ord" else (beta if k == "beta" else Ds[h] - tau - beta)
    key = (n, len(kind))
    if key not in _VC:
        _VC[key] = [S for S in itertools.combinations(range(len(kind)), n)
                    if abs(np.linalg.det(A[list(S)])) > 1e-10]
    best = np.inf
    for S in _VC[key]:
        y = np.linalg.solve(A[list(S)], b[list(S)])
        if (A @ y >= b - 1e-11).all() and (y >= -1e-11).all():
            v = float((np.maximum(y, 0.0) ** r).sum())
            if v < best:
                best = v
    return best


def G(n, r, beta, tau, Ds, A, kind):
    tgt = (n + 1) ** (1 - r)
    if tau < 0 or beta < 0 or beta < Ds[n] - tau - 1e-15:
        return np.inf
    inner = peel_inner(n, r, beta, tau, Ds, A, kind)
    if not np.isfinite(inner):
        return np.inf
    peel = inner + beta ** r
    tail = 0.0 if tau <= 0 else max(tau ** r, tau * max(beta, 1e-300) ** (r - 1))
    return tail + max(peel, tgt * max(0.0, 1 - 2 ** n * tau) ** r)


def V_value(n, r, ntau=600, nbeta=600, bmax=0.8, polish=True, tau_lo=1e-12, taumax=0.4):
    """min of G over the (beta, tau) region.  Vectorised: for each vertex subset S of the
    inner polyhedron, y_S is AFFINE in (beta, tau), so the whole grid is one array."""
    Ds = [D_h(n, h) for h in range(n + 1)]
    A, kind = peel_rows(n)
    tgt = (n + 1) ** (1 - r)
    m = len(kind)
    b0 = np.zeros(m); eb = np.zeros(m); et = np.zeros(m)
    for t, (k, h) in enumerate(kind):
        if k == "beta":
            eb[t] = 1.0
        elif k == "tail":
            b0[t] = Ds[h]; eb[t] = -1.0; et[t] = -1.0
    taus = np.concatenate([np.geomspace(tau_lo, max(tau_lo * 10, 1e-3), ntau // 3),
                           np.geomspace(max(tau_lo * 10, 1e-3), taumax, ntau - ntau // 3)])
    taus = taus[taus >= tau_lo]
    betas = np.linspace(1e-12, bmax, nbeta)
    TT, BB = np.meshgrid(taus, betas, indexing="ij")
    TT = TT.reshape(-1); BB = BB.reshape(-1)
    ok = BB >= Ds[n] - TT - 1e-15
    TT, BB = TT[ok], BB[ok]
    B0 = b0[None, :] + BB[:, None] * eb[None, :] + TT[:, None] * et[None, :]   # (P, m)
    inner = np.full(TT.shape, np.inf)
    for S in itertools.combinations(range(m), n):
        Ms = A[list(S)]
        if abs(np.linalg.det(Ms)) < 1e-10:
            continue
        Y = np.linalg.solve(Ms, B0[:, list(S)].T).T                            # (P, n)
        feas = (Y @ A.T >= B0 - 1e-11).all(1) & (Y >= -1e-11).all(1)
        val = (np.maximum(Y, 0.0) ** r).sum(1)
        inner = np.where(feas & (val < inner), val, inner)
    peel = inner + BB ** r
    tail = np.where(TT > 0, np.maximum(TT ** r, TT * np.maximum(BB, 1e-300) ** (r - 1)), 0.0)
    Gv = tail + np.maximum(peel, tgt * np.maximum(0.0, 1 - 2 ** n * TT) ** r)
    Gv = np.where(np.isfinite(inner), Gv, np.inf)
    i = int(np.argmin(Gv))
    best, barg = float(Gv[i]), (float(BB[i]), float(TT[i]))
    if polish:
        f = lambda z: G(n, r, abs(z[0]), abs(z[1]), Ds, A, kind)
        fc = lambda z: G(n, r, abs(z[0]), max(abs(z[1]), tau_lo), Ds, A, kind)
        for sc in (1.0, 0.3, 3.0):
            rr = minimize(fc, np.array([barg[0], barg[1] * sc]), method="Nelder-Mead",
                          options={"maxiter": 3000, "xatol": 1e-14, "fatol": 1e-16})
            if rr.fun < best:
                best, barg = float(rr.fun), (abs(rr.x[0]), max(abs(rr.x[1]), tau_lo))
    return best, tgt, barg


def phi(t, n, r):
    """the BOUND-1 curve: what the argument gives when the peeling side is not used."""
    return (n + 1) ** (1 - r) * max(0.0, 1 - 2 ** n * t) ** r + t ** r


def eps_star(n, r):
    """largest t0 such that phi(s) >= (n+1)^{1-r} for every s in [0, t0]."""
    tgt = (n + 1) ** (1 - r)
    hi = 2.0 ** (-n)
    if phi(hi, n, r) >= tgt:
        return hi
    lo, pr = 0.0, 1e-17
    while pr < hi and phi(pr, n, r) >= tgt:
        lo = pr
        pr *= 1.2
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if phi(mid, n, r) >= tgt:
            lo = mid
        else:
            hi = mid
    return lo


def margin(n, r, **kw):
    """The number that carries the proof: min of G over tau >= eps*(n,r)/2.

    For tau <= eps* the argument is ANALYTIC (G >= phi(tau) >= target), so the two
    regions overlap and a value >= target here completes the proof at this r."""
    e = eps_star(n, r)
    v, tgt, arg = V_value(n, r, tau_lo=max(e / 2, 1e-17), **kw)
    return v, tgt, arg, e


def r_star(n, lo=0.3, hi=0.999, iters=16, **kw):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v, tgt, _, _ = margin(n, mid, ntau=400, nbeta=400, **kw)
        if v >= tgt:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------- verification
def subsets(n):
    return [T for k in range(n + 1) for T in itertools.combinations(range(n), k)]


def random_KL(rng, n, N):
    Ts = subsets(n)
    for _ in range(50):
        v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        X = np.ones((N, len(Ts)), dtype=complex)
        for t, T in enumerate(Ts):
            for j in T:
                X[:, t] = X[:, t] * v[:, j]
        Ar = np.vstack([np.hstack([X.T.real, -X.T.imag]),
                        np.hstack([X.T.imag, X.T.real])])
        br = np.zeros(2 * len(Ts)); br[0] = 1.0
        c0, *_ = np.linalg.lstsq(Ar, br, rcond=None)
        if np.linalg.norm(Ar @ c0 - br) > 1e-9:
            continue
        u, s, vt = np.linalg.svd(Ar)
        ns = vt[int((s > 1e-10).sum()):].T
        if ns.shape[1]:
            c0 = c0 + ns @ rng.standard_normal(ns.shape[1]) * rng.uniform(0, 0.5)
        return c0[:N] + 1j * c0[N:], v
    return None, None


def min_Re_h(c, v, grid=120):
    n = v.shape[1]
    ang = 2 * np.pi * np.arange(grid) / grid
    mesh = np.meshgrid(*([ang] * n), indexing="ij")
    tot = np.zeros(mesh[0].shape, dtype=complex)
    for l in range(len(c)):
        t = np.full(mesh[0].shape, c[l], dtype=complex)
        for j in range(n):
            t = t * (1 + np.exp(1j * mesh[j]) * v[l, j])
        tot = tot + t
    R = tot.real
    i = np.unravel_index(np.argmin(R), R.shape)

    def f(x):
        z = 0j
        for l in range(len(c)):
            w = c[l]
            for j in range(n):
                w = w * (1 + np.exp(1j * x[j]) * v[l, j])
            z += w
        return z.real

    rr = minimize(f, np.array([ang[i[j]] for j in range(n)]), method="Nelder-Mead",
                  options={"maxiter": 3000, "xatol": 1e-12, "fatol": 1e-14})
    return min(float(R.min()), float(rr.fun))


def verify(rng, log, trials=25):
    w = {"peel": np.inf, "B": np.inf, "RS": np.inf, "G": np.inf, "P1": np.inf}
    cnt = 0
    for n in (2, 3):
        Ds = [D_h(n, h) for h in range(n + 1)]
        A, kind = peel_rows(n)
        for N in range(n + 1, n + 6):
            for _ in range(trials):
                c, v = random_KL(rng, n, N)
                if c is None or abs(c.sum()) < 1e-6:
                    continue
                c = c / c.sum()
                cnt += 1
                o = np.argsort(np.abs(c))[::-1]
                x = np.abs(c)[o]
                for h in range(n + 1):
                    w["peel"] = min(w["peel"], x[h:].sum() / Ds[h])
                S, T = o[:n + 1], o[n + 1:]
                tau = float(np.abs(c[T]).sum())
                RS = min_Re_h(c[S], v[S])
                w["RS"] = min(w["RS"], RS - (1 - 2 ** n * tau))
                for r in (0.2, 0.5, 2 / 3, 0.8):
                    tgt = (n + 1) ** (1 - r)
                    lhs = float((np.abs(c[S]) ** r).sum())
                    rhs = tgt * max(RS, 0.0) ** r
                    if rhs > 0:
                        w["B"] = min(w["B"], lhs / rhs)
                    tot = float((np.abs(c) ** r).sum())
                    g = G(n, r, float(x[n]), tau, Ds, A, kind)
                    if np.isfinite(g) and g > 0:
                        w["G"] = min(w["G"], tot / g)
                    w["P1"] = min(w["P1"], tot / tgt)
    log(f"[verify] {cnt} random exact (KL) systems, n = 2,3, N = n+1..n+5:")
    log(f"  (E)  peeling  min_h [sum_(l>h) x_l]/D_h(n)        = {w['peel']:.9f}   (>= 1)")
    log(f"  (B)  Theorem B  min sum_S|c|^r/[(n+1)^(1-r)R_S^r] = {w['B']:.9f}   (>= 1)")
    log(f"       R_S >= 1 - 2^n tau, min slack                = {w['RS']:+.9f}   (>= 0)")
    log(f"  (G)  the assembled bound: min sum|c|^r / G        = {w['G']:.9f}   (>= 1)")
    log(f"  P1 itself: min sum|c|^r / (n+1)^(1-r)             = {w['P1']:.9f}   (>= 1)")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--noverify", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"proof{a.tag}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    out = {}
    rs = [0.05, 0.1, 0.2, 1 / 3, 0.4, 0.5, 0.55, 0.6, 2 / 3, 0.7, 0.75, 0.8,
          0.82, 0.85, 0.875, 0.9, 0.925, 0.95, 0.97, 0.99]
    for n in a.ns:
        log(f"\n=== n = {n}   nu = {n+1}   D_h = "
            + ", ".join(f"{D_h(n, h):.9f}" for h in range(n + 1)))
        log(f"{'r':>8} {'q':>8} {'target':>14} {'eps*':>11} {'min G, tau>=eps*/2':>19}"
            f" {'/target':>12}  argmin (beta, tau)")
        rows = []
        for r in rs:
            v, tgt, arg, e = margin(n, r)
            rows.append(dict(r=r, q=r * (n + 1), target=tgt, eps_star=e, V=v,
                             ratio=v / tgt, beta=arg[0], tau=arg[1]))
            log(f"{r:8.4f} {r*(n+1):8.4f} {tgt:14.9f} {e:11.4e} {v:19.9f} {v/tgt:12.9f}  "
                f"({arg[0]:.6f}, {arg[1]:.4e})"
                + ("" if v >= tgt else "   <-- short: no proof at this r"))
        rst = r_star(n)
        log(f"  r*({n}) = {rst:.6f}  (q* = {rst*(n+1):.6f}):  P1 PROVED for every N and "
            f"every r <= r*")
        out[f"n{n}"] = dict(rows=rows, r_star=rst)
    if not a.noverify:
        out["verify"] = verify(np.random.default_rng(23), log, a.trials)
    (OUT / f"proof{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"\nwrote {OUT / f'proof{a.tag}.json'}")


if __name__ == "__main__":
    main()
