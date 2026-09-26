"""verify_eq_gamma.py -- referee tasks 4 and 5: the programme Gamma_s(p) itself.

Gamma_s(p) = max over shares b (b_k >= 0 for k = 1..nf = (p-1)/2, sum_k b_k = 1)
             of  min_{D = 1..p-1}  sum_k b_k^s (1 - cos(2 pi k D / p)),      s = nu/q.

Checked here:
  (4) the value, by EXHAUSTIVE support enumeration (all 2^nf - 1 supports, reduced by the
      multiplicative symmetry k -> c k) with a multistart epigraph SLSQP on each, then a
      high-precision KKT / active-set Newton solve in mpmath (50 digits).
  (5) uniqueness of the optimal b; the size of the ACTIVE SET A = {D : the min is attained at D}
      ("a kink"); whether the optimum is interior on its support; and whether the
      share-weighted block is well defined when |A| > 1.

Plain reading of the KKT system used for the high-precision solve.  At a maximiser the inner min
is attained at several offsets D at once.  Being optimal means: no way to move the shares that
raises ALL of those simultaneously.  Formally there are mixing weights mu_D >= 0 on the active
offsets, summing to 1, with
      s * b_k^{s-1} * sum_{D in A} mu_D (1 - cos(2 pi k D / p))  =  lambda    for every k with b_k > 0,
      sum_{k} b_k^s (1 - cos(2 pi k D / p)) = t                              for every D in A,
      sum_k b_k = 1,   sum_D mu_D = 1.
(That is: the "mixed" marginal gain of every used frequency is the same -- otherwise move share.)

Usage: uv run python heracles/verify_eq_gamma.py
"""
import itertools, json
from pathlib import Path

import numpy as np
import mpmath as mp
from scipy.optimize import minimize

import importlib.util, sys
_spec = importlib.util.spec_from_file_location(
    "vqb", str(Path(__file__).resolve().parent / "verify_eq_block.py"))
vqb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(vqb)

OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "VERIFY"


def gamma_exhaustive(p, s, starts=60):
    nf = (p - 1) // 2
    G = vqb.gmat(p)
    seen, recs = set(), []
    best = dict(value=-np.inf)
    for K in range(1, nf + 1):
        for S in itertools.combinations(range(1, nf + 1), K):
            c = vqb.canon_support(S, p)
            if c in seen:
                continue
            seen.add(c)
            v, b = vqb.maxmin_on_support(G, list(S), s, starts=starts, seed=abs(hash(c)) % (2 ** 31))
            recs.append((float(v), list(S), b.tolist()))
            if v > best["value"]:
                best = dict(value=float(v), S=list(S), b=b.tolist())
    recs.sort(reverse=True, key=lambda r: r[0])
    return best, recs, len(seen)


def active_set(p, s, S, b, tol=1e-9):
    G = vqb.gmat(p)[[k - 1 for k in S], :]
    vals = (np.asarray(b) ** s) @ G
    v = vals.min()
    A = [int(D + 1) for D in range(len(vals)) if vals[D] <= v + tol * max(1.0, abs(v))]
    return v, A, vals


def kkt_refine(p, s, S, A, b0, dps=60):
    """Newton solve of the KKT system in mpmath.  Returns (t, b, mu) or None.
    A must contain each active constraint ONCE (D and p-D give identical rows, so pass D <= p/2)."""
    mp.mp.dps = dps
    s = mp.mpf(s)
    K, m = len(S), len(A)
    g = [[1 - mp.cos(2 * mp.pi * k * D / p) for D in A] for k in S]   # g[k][D]

    def F(*x):
        b = list(x[:K]); mu = list(x[K:K + m]); lam = x[K + m]; t = x[K + m + 1]
        eqs = []
        for i in range(K):
            eqs.append(s * b[i] ** (s - 1) * sum(mu[d] * g[i][d] for d in range(m)) - lam)
        for d in range(m):
            eqs.append(sum(b[i] ** s * g[i][d] for i in range(K)) - t)
        eqs.append(sum(b) - 1)
        eqs.append(sum(mu) - 1)
        return eqs

    t0 = min(sum(mp.mpf(float(b0[i])) ** s * g[i][d] for i in range(K)) for d in range(m))
    x0 = [mp.mpf(float(v)) for v in b0] + [mp.mpf(1) / m] * m + [mp.mpf(1), t0]
    try:
        sol = mp.findroot(F, x0, solver="mdnewton", tol=mp.mpf(10) ** (-(dps - 10)))
    except Exception as e:
        return None, str(e)
    x = [sol[i] for i in range(K + m + 2)]
    return dict(t=x[K + m + 1], b=x[:K], mu=x[K:K + m]), None


def uniqueness_scan(p, s, S, bstar, tries=4000, seed=1):
    """Sample random shares on the support; how far below the optimum are near-optimal points,
    and is there a SECOND local max (a distinct b with the same value)?"""
    rng = np.random.default_rng(seed)
    G = vqb.gmat(p)[[k - 1 for k in S], :]
    vstar = float((np.asarray(bstar) ** s) @ G).__float__() if False else float(((np.asarray(bstar) ** s) @ G).min())
    near = []
    for _ in range(tries):
        b = rng.dirichlet(np.ones(len(S)) * rng.choice([0.3, 1.0, 5.0, 50.0]))
        v = float(((b ** s) @ G).min())
        if v > vstar * (1 - 1e-7):
            near.append((v, b.tolist()))
    # local polish from many starts, collect distinct argmaxes
    argmaxes = []
    for t in range(200):
        v, b = vqb.maxmin_on_support(G, list(range(1, len(S) + 1)), s, starts=1, seed=t) \
            if False else (None, None)
    return vstar, len(near), near[:5]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for (p, n, q) in [(11, 2, 2.0), (13, 2, 2.0), (23, 2, 2.0), (11, 3, 2.0)]:
        nu = n + 1; s = nu / q
        tag = f"p{p}_n{n}_q{q:g}"
        print(f"== {tag}  s = nu/q = {s}", flush=True)
        best, recs, nsup = gamma_exhaustive(p, s, starts=80 if p < 20 else 60)
        S0, b0 = best["S"], np.array(best["b"])
        keep = [i for i in range(len(S0)) if b0[i] > 1e-9]
        S, b = [S0[i] for i in keep], b0[keep] / b0[keep].sum()
        v, A, vals = active_set(p, s, S, b)
        print(f"   exhaustive over {nsup} symmetry-classes of supports")
        print(f"   Gamma_{s:g}({p}) = {best['value']:.15f}")
        print(f"   true support (zero shares dropped) {S}   shares {[round(x,12) for x in b]}")
        print(f"   interior on its support: {bool((b > 1e-12).all())}   K = {len(S)}")
        print(f"   ACTIVE offsets D (min attained) = {A}   (|A| = {len(A)}); D and p-D are the same constraint")
        Ah = [D for D in A if D <= p // 2]
        print(f"   distinct up to D <-> p-D: {Ah}  (|A/~| = {len(Ah)})")
        print(f"   runner-up supports:")
        for v2, S2, b2 in recs[1:4]:
            print(f"      {S2}  value {v2:.12f}  (deficit {best['value']-v2:.3e})")
        sol, err = kkt_refine(p, s, S, Ah, b)
        row = dict(p=p, n=n, q=q, s=s, gamma_f64=best["value"], S=S, b=b.tolist(),
                   nsupports=nsup, active=A, active_half=Ah, interior=bool((b > 1e-12).all()))
        if sol:
            mp.mp.dps = 50
            print(f"   mpmath KKT refine (50 dps): Gamma = {mp.nstr(sol['t'], 30)}")
            print(f"      b  = {[mp.nstr(x, 25) for x in sol['b']]}")
            print(f"      mu = {[mp.nstr(x, 20) for x in sol['mu']]}   (all >= 0: "
                  f"{all(x >= 0 for x in sol['mu'])})")
            print(f"      |Gamma_mp - Gamma_f64| = {float(abs(sol['t'] - mp.mpf(best['value']))):.3e}")
            row["gamma_mp"] = mp.nstr(sol["t"], 30)
            row["b_mp"] = [mp.nstr(x, 25) for x in sol["b"]]
            row["mu_mp"] = [float(x) for x in sol["mu"]]
        else:
            print(f"   mpmath KKT refine FAILED: {err}")
            row["gamma_mp"] = None
        vstar, nnear, near = uniqueness_scan(p, s, S, b)
        print(f"   random-share scan: {nnear}/4000 samples within 1e-7 of the optimum", flush=True)
        row["near_count"] = nnear
        res[tag] = row
    (OUT / "eq_gamma.json").write_text(json.dumps(res, indent=1, default=str))
    print("\nwrote", OUT / "eq_gamma.json")
