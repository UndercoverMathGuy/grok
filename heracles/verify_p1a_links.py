r"""VERIFY / P1A -- independent numerical stress tests of the three ingredients that
ADVERSARY.md section 6 assembles into G, and of the assembled chain itself.

Nothing here calls adv_proof.py or p1a_cert.py; G is re-implemented from the written
formulas.  All float64 (plus an mpmath spot check where tightness matters).

  L1  Theorem E  (GLOBAL_OPT3):   sum_{l>h} x_l >= D_h(n) |A|,  h = 0..n,
      on genuine (KL) systems, including degenerate ones (repeated phase vectors,
      coalescing phases, zero-modulus atoms, N up to 60).
  L2  Theorem B  (GLOBAL_OPT sec.4): for EXACTLY n+1 atoms, junk allowed,
      sum |c_l|^r >= (n+1)^{1-r} max(R,0)^r,  R = min_{u in T^n} Re h(u),
      h(u) = sum_l c_l prod_j (1 + u_j v_lj).   Tested including REPEATED phase
      vectors, which is the degeneracy the drop step can produce.
  L3  the drop step:  R_S >= |A| - 2^n tau  for S = the top n+1 atoms of a (KL) system.
  L4  the assembled chain  sum_l |c_l|^r >= G(beta, tau).

Run:  uv run --with mpmath --with numpy --with scipy python verify_p1a_links.py
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
rng = np.random.default_rng(20260921)


# ------------------------------------------------------------------ constants


KAP = {1: 0.5, 2: 3 * np.sqrt(3.0) / 8, 3: 49 / 72}


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
    return max(np.prod([KAP[d] for d in dd]) for dd in parts(n, h))


def subsets(n):
    return [tuple(j for j in range(n) if (m >> j) & 1) for m in range(1 << n)]


def Umat(v):
    """rows indexed by T subset {1..n}, columns by atom: prod_{j in T} v_lj."""
    N, n = v.shape
    return np.array([np.prod(v[:, list(T)], axis=1) if T else np.ones(N, complex)
                     for T in subsets(n)])


# ------------------------------------------------------------------ (KL) generators


def kl_random(n, N):
    """random phases, least-norm solution + random null component.  Needs N >= 2^n."""
    v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
    U = Umat(v)
    rhs = np.zeros(1 << n, complex); rhs[0] = 1.0
    c, *_ = np.linalg.lstsq(U, rhs, rcond=None)
    if np.max(np.abs(U @ c - rhs)) > 1e-9:
        return None
    if N > (1 << n):
        ns = np.linalg.svd(U)[2][(1 << n):].conj().T
        s = rng.normal(0, 0.5, (ns.shape[1], 2))
        c = c + ns @ (s[:, 0] + 1j * s[:, 1])
        if np.max(np.abs(U @ c - rhs)) > 1e-9:
            return None
    return v, c


def kl_block(n):
    """the exact (n+1)-atom junk-free block: v_lj = w^l (same in every slot), c_l = 1/(n+1)."""
    w = np.exp(-2j * np.pi / (n + 1))
    v = np.array([[w ** l] * n for l in range(n + 1)], dtype=complex)
    c = np.full(n + 1, 1.0 / (n + 1), dtype=complex)
    return v, c


def kl_split(n, extra, jitter=0.0):
    """block with `extra` atoms obtained by SPLITTING block atoms -- repeated phase
    vectors (jitter = 0) or nearly-repeated ones (jitter > 0).  Genuine (KL)."""
    v, c = kl_block(n)
    v = list(v); c = list(c)
    for _ in range(extra):
        i = int(rng.integers(len(c)))
        f = rng.uniform(0.05, 0.5)
        newv = v[i] * np.exp(1j * jitter * rng.normal(0, 1, n)) if jitter else v[i].copy()
        if jitter:
            # keep (KL) exact: re-solve for the two coefficients is not possible with a
            # moved phase, so only exact duplicates are used when jitter == 0
            newv = v[i].copy()
        v.append(newv); c.append(c[i] * f); c[i] = c[i] * (1 - f)
    return np.array(v), np.array(c)


def kl_zero_atoms(n, extra):
    v, c = kl_block(n)
    v = np.vstack([v, np.exp(1j * rng.uniform(0, 2 * np.pi, (extra, n)))])
    c = np.concatenate([c, np.zeros(extra, complex)])
    return v, c


def kl_coalesce(n, N, eps):
    """random (KL) with two atoms forced to nearly identical phases."""
    out = kl_random(n, N)
    if out is None:
        return None
    v, _ = out
    v[1] = v[0] * np.exp(1j * eps * rng.normal(0, 1, n))
    U = Umat(v)
    rhs = np.zeros(1 << n, complex); rhs[0] = 1.0
    c, *_ = np.linalg.lstsq(U, rhs, rcond=None)
    if np.max(np.abs(U @ c - rhs)) > 1e-8:
        return None
    if N > (1 << n):
        ns = np.linalg.svd(U)[2][(1 << n):].conj().T
        s = rng.normal(0, 0.5, (ns.shape[1], 2))
        c = c + ns @ (s[:, 0] + 1j * s[:, 1])
        if np.max(np.abs(U @ c - rhs)) > 1e-8:
            return None
    return v, c


# ------------------------------------------------------------------ R = min Re h


def Rmin(v, c, n, grid=None, polish=True):
    """min over the torus of Re sum_l c_l prod_j (1+u_j v_lj).  Vectorised grid + polish.
    The grid min is an UPPER bound on the true min, and the true min appears on the RIGHT
    of Theorem B, so over-estimating R makes the test HARDER -- the adversarial direction."""
    from scipy.optimize import minimize
    g = grid or (128 if n == 2 else 48)

    def f(th):
        u = np.exp(1j * np.asarray(th))
        return float(np.real(np.sum(c * np.prod(1 + u[None, :] * v, axis=1))))

    axes = np.linspace(0, 2 * np.pi, g, endpoint=False)
    U = np.exp(1j * np.stack(np.meshgrid(*([axes] * n), indexing="ij"), axis=-1))
    U = U.reshape(-1, n)                                   # (g^n, n)
    prod = np.prod(1.0 + U[:, None, :] * v[None, :, :], axis=2)   # (g^n, N)
    vals = np.real(prod @ c)
    i = int(np.argmin(vals))
    best, barg = float(vals[i]), U[i]
    barg = np.angle(barg)
    if polish:
        for _ in range(3):
            res = minimize(f, np.array(barg) + rng.normal(0, 0.05, n), method="Nelder-Mead",
                           options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000))
            if res.fun < best:
                best, barg = float(res.fun), res.x
    return best


# ------------------------------------------------------------------ G, from the text


def G(n, r, beta, tau):
    tgt = (n + 1) ** (1 - r)
    Ds = [D_h(n, h) for h in range(n + 1)]
    rows, tags = [], []
    for i in range(n - 1):
        z = np.zeros(n); z[i], z[i + 1] = 1, -1
        rows.append(z); tags.append(("ord", 0))
    z = np.zeros(n); z[n - 1] = 1
    rows.append(z); tags.append(("beta", 0))
    for h in range(n):
        z = np.zeros(n); z[h:] = 1
        rows.append(z); tags.append(("tail", h))
    A = np.array(rows)
    b = np.array([0.0 if k == "ord" else (beta if k == "beta" else Ds[h] - beta - tau)
                  for k, h in tags])
    best = np.inf
    for S in itertools.combinations(range(len(rows)), n):
        M = A[list(S)]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        y = np.linalg.solve(M, b[list(S)])
        if (A @ y >= b - 1e-10).all():
            best = min(best, float((np.maximum(y, 0) ** r).sum()))
    peel = (best + beta ** r) if np.isfinite(best) else -np.inf
    tail = 0.0 if tau <= 0 else max(tau ** r, tau * max(beta, 1e-300) ** (r - 1))
    return tail + max(peel, tgt * max(0.0, 1 - 2 ** n * tau) ** r)


# ------------------------------------------------------------------ the tests


def main():
    os.makedirs(OUT, exist_ok=True)
    rec = {}

    # ---------------- L1: Theorem E on genuine (KL) systems
    print("L1  Theorem E:  min over systems and h of  [sum_{l>h} x_l] / D_h(n)   (must be >= 1)")
    for n in (2, 3):
        worst, wh, cnt = np.inf, None, 0
        for N in range(1 << n, (1 << n) + 9):
            for _ in range(400):
                out = kl_random(n, N)
                if out is None:
                    continue
                v, c = out
                x = np.sort(np.abs(c))[::-1]
                cnt += 1
                for h in range(n + 1):
                    s = float(x[h:].sum())
                    val = s / D_h(n, h)
                    if val < worst:
                        worst, wh = val, (N, h)
        # degenerate families
        for fam, gen in (("split", lambda: kl_split(n, int(rng.integers(1, 6)))),
                         ("zeros", lambda: kl_zero_atoms(n, int(rng.integers(1, 40)))),
                         ("coal", lambda: kl_coalesce(n, (1 << n) + 2, 10.0 ** -rng.uniform(1, 6)))):
            for _ in range(600):
                out = gen()
                if out is None:
                    continue
                v, c = out
                x = np.sort(np.abs(c))[::-1]
                cnt += 1
                for h in range(n + 1):
                    val = float(x[h:].sum()) / D_h(n, h)
                    if val < worst:
                        worst, wh = val, (fam, h)
        print(f"   n = {n}: min ratio = {worst:.9f} at {wh}   over {cnt} systems"
              f"   {'OK' if worst >= 1 - 1e-9 else '<<< THEOREM E VIOLATED'}")
        rec[f"E_n{n}"] = dict(min_ratio=float(worst), where=str(wh), systems=cnt)

    # ---------------- L2: Theorem B on n+1 atoms, junk allowed, incl. repeated phases
    print("\nL2  Theorem B (n+1 atoms, junk allowed): min of "
          "sum|c|^r / [(n+1)^{1-r} max(R,0)^r]   (must be >= 1)")
    for n, r in ((2, 2 / 3), (2, 0.5), (2, 0.9), (3, 0.5)):
        worst, wcase, cnt = np.inf, None, 0
        cases = []
        for _ in range(250):
            cases.append(("generic",
                          np.exp(1j * rng.uniform(0, 2 * np.pi, (n + 1, n))),
                          rng.normal(0, 1, n + 1) + 1j * rng.normal(0, 1, n + 1)))
        for _ in range(250):                       # two atoms share a phase vector
            v = np.exp(1j * rng.uniform(0, 2 * np.pi, (n + 1, n)))
            v[1] = v[0]
            cases.append(("repeated", v,
                          rng.normal(0, 1, n + 1) + 1j * rng.normal(0, 1, n + 1)))
        for _ in range(250):                       # near-block
            vb, cb = kl_block(n)
            cases.append(("near-block", vb * np.exp(1j * rng.normal(0, 0.15, (n + 1, n))),
                          cb + rng.normal(0, 0.08, n + 1) + 1j * rng.normal(0, 0.08, n + 1)))
        for tag, v, c in cases:
            R = Rmin(v, c, n)
            if R <= 1e-9:
                continue
            lhs = float((np.abs(c) ** r).sum())
            rhs = (n + 1) ** (1 - r) * R ** r
            cnt += 1
            if lhs / rhs < worst:
                worst, wcase = lhs / rhs, tag
        print(f"   n = {n}, r = {r:.4f}: min = {worst:.9f}  ({wcase})  over {cnt} cases"
              f"   {'OK' if worst >= 1 - 1e-7 else '<<< THEOREM B VIOLATED'}")
        rec[f"B_n{n}_r{r:.3f}"] = dict(min_ratio=float(worst), where=str(wcase), cases=cnt)

    # ---------------- L3: the drop step R_S >= 1 - 2^n tau
    print("\nL3  drop step:  R_S - (1 - 2^n tau)   (must be >= 0)")
    for n in (2, 3):
        worst, cnt = np.inf, 0
        for N in range(1 << n, (1 << n) + 6):
            for _ in range(150):
                out = kl_random(n, N)
                if out is None:
                    continue
                v, c = out
                o = np.argsort(np.abs(c))[::-1]
                v, c = v[o], c[o]
                tau = float(np.abs(c[n + 1:]).sum())
                RS = Rmin(v[:n + 1], c[:n + 1], n)
                cnt += 1
                worst = min(worst, RS - (1 - 2 ** n * tau))
        print(f"   n = {n}: min slack = {worst:+.6e} over {cnt} systems"
              f"   {'OK' if worst >= -1e-9 else '<<< VIOLATED'}")
        rec[f"drop_n{n}"] = dict(min_slack=float(worst), systems=cnt)

    # ---------------- L4: the assembled chain
    print("\nL4  chain:  min of  sum_l|c_l|^r / G(beta, tau)   (must be >= 1)")
    for n, r in ((2, 2 / 3), (3, 0.5)):
        worst, wcase, cnt = np.inf, None, 0
        fams = {
            "random N=2^n..2^n+12": [lambda N=N: kl_random(n, N)
                                     for N in range((1 << n), (1 << n) + 13)],
            "split (repeated phases)": [lambda k=k: kl_split(n, k) for k in range(1, 12)],
            "zero atoms": [lambda k=k: kl_zero_atoms(n, k) for k in (1, 5, 20, 57)],
            "coalescing": [lambda e=e: kl_coalesce(n, (1 << n) + 2, e)
                           for e in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)],
        }
        for fam, gens in fams.items():
            for gen in gens:
                for _ in range(120):
                    out = gen()
                    if out is None:
                        continue
                    v, c = out
                    x = np.sort(np.abs(c))[::-1]
                    if len(x) < n + 2:
                        x = np.concatenate([x, np.zeros(n + 2 - len(x))])
                    beta = float(x[n]); tau = float(x[n + 1:].sum())
                    g = G(n, r, beta, tau)
                    if not np.isfinite(g) or g <= 0:
                        continue
                    cnt += 1
                    ratio = float((x ** r).sum()) / g
                    if ratio < worst:
                        worst, wcase = ratio, (fam, len(x), beta, tau)
        print(f"   n = {n}, r = {r:.4f}: min = {worst:.9f}   over {cnt} systems"
              f"   {'OK' if worst >= 1 - 1e-9 else '<<< CHAIN VIOLATED'}")
        print(f"      worst case: {wcase}")
        rec[f"chain_n{n}_r{r:.3f}"] = dict(min_ratio=float(worst), where=str(wcase),
                                           systems=cnt)

    # ---------------- P1 itself on the same systems (sanity)
    print("\nL5  P1 itself:  min of  sum|c|^r / (n+1)^{1-r}   (must be >= 1)")
    for n, r in ((2, 2 / 3), (2, 0.95), (3, 0.5)):
        worst, cnt = np.inf, 0
        for N in range((1 << n), (1 << n) + 13):
            for _ in range(200):
                out = kl_random(n, N)
                if out is None:
                    continue
                _, c = out
                cnt += 1
                worst = min(worst, float((np.abs(c) ** r).sum()) / (n + 1) ** (1 - r))
        for k in range(1, 12):
            for _ in range(200):
                _, c = kl_split(n, k)
                cnt += 1
                worst = min(worst, float((np.abs(c) ** r).sum()) / (n + 1) ** (1 - r))
        print(f"   n = {n}, r = {r:.4f}: min = {worst:.9f} over {cnt} systems"
              f"   {'OK' if worst >= 1 - 1e-9 else '<<< P1 VIOLATED'}")
        rec[f"P1_n{n}_r{r:.3f}"] = dict(min_ratio=float(worst), systems=cnt)

    with open(os.path.join(OUT, "p1a_links.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT,'p1a_links.json')}")


if __name__ == "__main__":
    main()
