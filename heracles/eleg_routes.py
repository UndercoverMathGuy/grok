"""eleg_routes.py -- the settled / failed routes of ELEGANT.md section 5.

blocks:
  mirror   the strictly-UPPER entries of M = sum_l c_l m_l m_l^* are NOT controlled by (KL),
           so the tempting claim "M = A I" (a tight frame) is FALSE.  Explicit n=1 witness
           plus a sweep showing the lower triangle is 0 and the upper one is not.
  logconv  is r |-> log V(r) log-convex / log-concave?  V(r) = min over (KL) of sum|c_l|^r.
           Theorem 1 + the block say V(r) = (n+1)^{1-r} exactly, i.e. log V is AFFINE.
           Measured directly, so the brief's question is answered numerically as well.
  partial  the e_k relaxation with the middle degrees DROPPED: keeping only e_D >= 1, or only
           e_1 >= D, gives values strictly below D.  So the middle degrees are load-bearing.

Run:  uv run --with scipy python eleg_routes.py [--blocks mirror logconv partial]
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize

from eleg_chain import (chain_vectors, elem_sym_all, moment_matrix, sample_full,
                        sample_pertblock, subsets)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------- mirror
def block_mirror(rng):
    res = {}
    # the explicit n = 1, N = 2 witness
    v = np.array([[1.0 + 0j], [1j]])
    c = np.array([(1 - 1j) / 2, (1 + 1j) / 2])
    sets = subsets(1)
    res["witness_KL_residual"] = float(np.max(np.abs(
        moment_matrix(v, sets) @ c - np.array([1.0 + 0j, 0.0 + 0j]))))
    m = chain_vectors(v)
    M = np.einsum("l,lk,lj->kj", c, m, m.conj())
    res["witness_M"] = [[f"{z.real:+.12f}{z.imag:+.12f}j" for z in row] for row in M]
    res["witness_upper"] = float(abs(M[0, 1]))

    lower, upper, upper_over_S = 0.0, 0.0, 0.0
    cnt = 0
    for n in (1, 2, 3):
        for N in range(n + 1, 2 ** n + 3):
            for _ in range(60):
                got = (sample_full(n, N, rng, jitter=0.3) if N >= 2 ** n
                       else sample_pertblock(n, N, rng, jitter=0.3))
                if got is None:
                    continue
                v, c = got
                A = c.sum()
                if abs(A) < 1e-6:
                    continue
                m = chain_vectors(v)
                M = np.einsum("l,lk,lj->kj", c, m, m.conj()) / A
                lower = max(lower, float(np.max(np.abs(np.tril(M, -1)))))
                up = float(np.max(np.abs(np.triu(M, 1))))
                upper = max(upper, up)
                upper_over_S = max(upper_over_S, up / (np.sum(np.abs(c)) / abs(A)))
                cnt += 1
    res.update({"systems": cnt, "max_lower": lower, "max_upper": upper,
                "max_upper_over_sum_moduli": upper_over_S})
    print(f"[mirror] {cnt} systems | max |strictly lower| = {lower:.3e} "
          f"| max |strictly upper| = {upper:.4f} "
          f"| max upper/(sum|c|/|A|) = {upper_over_S:.4f}")
    print(f"[mirror] n=1 witness: KL residual {res['witness_KL_residual']:.2e}, "
          f"M = {res['witness_M']}, |M_01| = {res['witness_upper']:.12f}")
    return res


# --------------------------------------------------------------------- logconv
def V_numeric(n, r, N, rng, starts=250):
    """min sum_l |c_l|^r over (KL) systems with A = 1, free phases and free c."""
    sets = subsets(n)
    best = np.inf
    for _ in range(starts):
        th0 = rng.random(N * n) * 2 * np.pi

        def val(th):
            v = np.exp(1j * th.reshape(N, n))
            M = moment_matrix(v, sets)
            rhs = np.zeros(len(sets), dtype=complex)
            rhs[0] = 1.0
            c, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            if np.max(np.abs(M @ c - rhs)) > 1e-8:
                return 1e3
            return float(np.sum(np.abs(c) ** r))

        out = minimize(val, th0, method="Nelder-Mead",
                       options={"maxiter": 2500, "xatol": 1e-10, "fatol": 1e-13})
        best = min(best, float(out.fun))
    return best


def block_logconv(rng):
    rows = []
    rs = [0.2, 1 / 3, 0.5, 2 / 3, 0.8, 0.95]
    for n in (1, 2):
        N = 2 ** n
        vals = []
        for r in rs:
            V = V_numeric(n, r, N, rng, starts=30 if n == 1 else 60)
            vals.append(V)
            rows.append({"n": n, "r": r, "V": V, "target": (n + 1) ** (1 - r),
                         "ratio": V / (n + 1) ** (1 - r)})
            print(f"[logconv] n={n} r={r:.4f}: V={V:.10f}  "
                  f"target={(n+1)**(1-r):.10f}  ratio={V/(n+1)**(1-r):.10f}")
        # second differences of log V on the (unequally spaced) r grid
        lv = np.log(np.array(vals))
        sec = []
        for i in range(1, len(rs) - 1):
            h0, h1 = rs[i] - rs[i - 1], rs[i + 1] - rs[i]
            sec.append(((lv[i + 1] - lv[i]) / h1 - (lv[i] - lv[i - 1]) / h0) / ((h0 + h1) / 2))
        print(f"[logconv] n={n} second differences of log V: "
              f"{['%.2e' % s for s in sec]}")
        rows.append({"n": n, "second_differences": [float(s) for s in sec]})
    return rows


# --------------------------------------------------------------------- partial
def relax_partial(D, r, N, keep, starts=120, rng=None):
    """min sum x^r s.t. only the constraints e_k >= binom(D,k) for k in `keep`."""
    def obj(z):
        return float(np.sum((z ** 2) ** r))

    def cons(z):
        e = elem_sym_all(z ** 2)
        return np.array([(e[k] if k < len(e) else 0.0) - math.comb(D, k) for k in keep])

    best, bx = np.inf, None
    for _ in range(starts):
        z0 = rng.random(N) * 2.0
        out = minimize(obj, z0, constraints=[{"type": "ineq", "fun": cons}],
                       method="SLSQP", options={"maxiter": 500, "ftol": 1e-14})
        if out.success and np.min(cons(out.x)) > -1e-9 and out.fun < best:
            best, bx = float(out.fun), np.sort(out.x ** 2)[::-1]
    return best, bx


def block_partial(rng):
    rows = []
    for D in (3, 4, 5):
        for r in (0.5, 2 / 3):
            for name, keep in (("all", list(range(1, D + 1))),
                               ("top only (e_D)", [D]),
                               ("trace only (e_1)", [1])):
                N = 2 * D
                best, bx = relax_partial(D, r, N, keep, starts=90, rng=rng)
                rows.append({"D": D, "r": r, "keep": name, "N": N,
                             "min": best, "ratio": best / D,
                             "argmin": None if bx is None else
                             [float(t) for t in np.round(bx, 6)]})
                print(f"[partial] D={D} r={r:.4f} keep={name:16s}: "
                      f"min={best:.9f} ratio={best/D:.9f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=["mirror", "partial", "logconv"])
    a = ap.parse_args()
    rng = np.random.default_rng(31337)
    path = os.path.join(OUT, "routes.json")
    rep = json.load(open(path)) if os.path.exists(path) else {}
    if "mirror" in a.blocks:
        rep["mirror"] = block_mirror(rng)
    if "partial" in a.blocks:
        rep["partial"] = block_partial(rng)
    if "logconv" in a.blocks:
        rep["logconv"] = block_logconv(rng)
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("wrote routes.json")


if __name__ == "__main__":
    main()
