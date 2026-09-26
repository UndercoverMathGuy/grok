"""VERIFY link 4: the peeling constants S_d, kappa_d, D_h(n) -- recomputed from scratch.

Independent re-derivation.  Nothing is imported from gopt3_*.py / p1b_*.py.

Definitions (re-derived here, not copied):
  S_d      := max over w in T^d of | 1 - prod_{j=1..d} (1+w_j)/2 |
              ("how far from 1 the normalised reproducing-kernel product can get on the torus")
  kappa_d  := (1 - 2^{-d}) / S_d
  D_h(n)   := max { prod_i kappa_{d_i} : d_i >= 1, sum_i d_i <= n }   (P1B's corrected version)
  D_h^=(n) := same with sum_i d_i = n                                  (GLOBAL_OPT3's version)

We check (a) the claimed diagonal reduction of S_d by a full d-dimensional global search,
(b) the exact algebraic values S_2 = 2/sqrt3, S_3 = 9/7, kappa_2 = 3 sqrt3/8, kappa_3 = 49/72,
(c) the D_h tables in GLOBAL_OPT3 and P1B.
"""
import itertools, json, os
import numpy as np
from scipy.optimize import minimize
import mpmath as mp

mp.mp.dps = 40
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)


def Sd_diag(d):
    """max over a single angle t of |1 - ((1+e^{it})/2)^d|, high precision."""
    def f(t):
        u = ((1 + mp.e**(1j * t)) / 2) ** d
        return -abs(1 - u)
    best = None
    for t0 in np.linspace(0.01, mp.pi, 400):
        t = mp.mpf(float(t0))
        # golden search locally
        lo, hi = max(mp.mpf(0), t - mp.pi / 400), min(mp.pi, t + mp.pi / 400)
        for _ in range(200):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            if f(m1) < f(m2):
                hi = m2
            else:
                lo = m1
        v = -f((lo + hi) / 2)
        if best is None or v > best:
            best = v
    return best


def Sd_full(d, ntries=4000, seed=0):
    """max over the FULL d-torus of |1 - prod_j (1+w_j)/2| by multistart (float64)."""
    rng = np.random.default_rng(seed)

    def neg(psi):
        w = np.exp(1j * psi)
        return -abs(1 - np.prod(1 + w) / 2 ** d)

    best = 0.0
    for _ in range(ntries):
        x0 = rng.uniform(0, 2 * np.pi, d)
        res = minimize(neg, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
        best = max(best, -res.fun)
    return best


def main():
    rep = {}
    # ---- S_d and kappa_d -------------------------------------------------
    rows = []
    for d in range(1, 7):
        sd_d = Sd_diag(d)
        sd_f = Sd_full(d, ntries=300 if d <= 3 else 800, seed=d)
        kap = (1 - mp.mpf(2) ** (-d)) / sd_d
        rows.append(dict(d=d, S_diag=float(sd_d), S_full=float(sd_f),
                         full_minus_diag=float(sd_f) - float(sd_d), kappa=float(kap)))
    rep["Sd"] = rows
    print("d   S_d(diagonal)        S_d(full torus)      full-diag        kappa_d")
    for r in rows:
        print(f"{r['d']}   {r['S_diag']:.15f}   {r['S_full']:.15f}   {r['full_minus_diag']:+.3e}   {r['kappa']:.15f}")

    # exact algebraic checks
    exact = {
        2: (2 / mp.sqrt(3), 3 * mp.sqrt(3) / 8),
        3: (mp.mpf(9) / 7, mp.mpf(49) / 72),
    }
    ex = {}
    for d, (s_ex, k_ex) in exact.items():
        sd = Sd_diag(d)
        ex[d] = dict(S_err=float(abs(sd - s_ex)), kappa_err=float(abs((1 - mp.mpf(2) ** (-d)) / sd - k_ex)))
        print(f"  exact check d={d}: |S_d - closed form| = {ex[d]['S_err']:.3e}, "
              f"|kappa_d - closed form| = {ex[d]['kappa_err']:.3e}")
    rep["exact_checks"] = ex

    # S_4 claimed closed form: S_4^2 = max_c (1 - 15c^4 + 16c^3 - 2c^2) at c = (6+sqrt21)/15
    c = (6 + mp.sqrt(21)) / 15
    s4sq = 1 - 15 * c ** 4 + 16 * c ** 3 - 2 * c ** 2
    print(f"  GLOBAL_OPT3's S_4 closed form -> S_4 = {float(mp.sqrt(s4sq)):.15f} "
          f"vs computed {rows[3]['S_diag']:.15f}  (diff {float(mp.sqrt(s4sq)) - rows[3]['S_diag']:+.3e})")
    rep["S4_closedform"] = dict(value=float(mp.sqrt(s4sq)), computed=rows[3]["S_diag"])

    # ---- D_h(n), both conventions ---------------------------------------
    kappa = {d: float((1 - mp.mpf(2) ** (-d)) / Sd_diag(d)) for d in range(1, 8)}

    def D(h, n, strict_eq):
        if h == 0:
            return 1.0
        best = -1.0
        # compositions of total <= n (or == n) into h positive parts
        for comp in itertools.product(range(1, n + 1), repeat=h):
            tot = sum(comp)
            if tot > n:
                continue
            if strict_eq and tot != n:
                continue
            best = max(best, float(np.prod([kappa[d] for d in comp])))
        return best

    tab = {}
    print("\n D_h(n):   [<=n  =  P1B corrected]   vs   [=n  =  GLOBAL_OPT3]")
    for n in range(2, 7):
        le = [D(h, n, False) for h in range(0, n + 1)]
        eq = [D(h, n, True) for h in range(0, n + 1)]
        tab[n] = dict(le=le, eq=eq)
        print(f"  n={n}: <=n {['%.9f' % v for v in le]}")
        print(f"        =n  {['%.9f' % v for v in eq]}")
        for h in range(n + 1):
            if abs(le[h] - eq[h]) > 1e-12:
                print(f"        *** differ at h={h}: {le[h]:.9f} vs {eq[h]:.9f}  "
                      f"(P1B's correction improves D_{h}({n}))")
    rep["D"] = tab

    # is only d in {1,2,3} ever needed under the <=n convention?
    print("\n  argmax parts under <=n convention (should never use d>=4 since kappa_3 is the max):")
    kmax = max(kappa[d] for d in range(1, 8))
    print(f"    max_d kappa_d = {kmax:.12f} at d = {max(range(1,8), key=lambda d: kappa[d])}")
    rep["kappa"] = kappa

    with open(os.path.join(OUT, "constants.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("\nwrote", os.path.join(OUT, "constants.json"))


if __name__ == "__main__":
    main()
