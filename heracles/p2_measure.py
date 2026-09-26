"""P2 measurement: the three candidate objects, as functions of the junk level.

For a population of N atoms with C_empty = 1 and junk energy sum_{T!=0}|C_T|^2 <= eta^2:

  Cl2_r(eta)  = min sum_l |c_l|^r                (the object P2 must minorise)
  Pmin(eta)   = min (n+1)^{n+1} prod_l |c_l|     (the "quadratic product lemma", N=n+1)
  Lkill(eta)  = min mean over the (n+1)! kill points of log|h|   (N=n+1)
  Rmin(eta)   = min over the torus of Re h, minimised over configs (sanity)

Usage:  uv run python p2_measure.py --obj cost --nu 3 --q 2 --starts 40
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize
import p2_core as P


def solve(obj, n, N, r, eta, starts, seed=0, maxiter=400):
    M = P.masks(n)
    rng = np.random.default_rng(seed)
    dim = 2 * N + N * n

    if obj == "cost":
        f = lambda z: (P.cost_r(z[:N] + 1j * z[N:2 * N], r), P.cost_r_grad(z, N, n, r))
    elif obj == "prod":
        def f(z):
            c = z[:N] + 1j * z[N:2 * N]
            a = np.maximum(np.abs(c), 1e-300)
            val = float(np.sum(np.log(a)))
            g = np.zeros(dim)
            g[:N] = c.real / a ** 2
            g[N:2 * N] = c.imag / a ** 2
            return val, g
    elif obj == "lkill":
        def f(z):
            c, th = P.unpack(z, N, n)
            V = np.exp(1j * th)
            U, _ = P.kill_points(V)
            h = P.hval(c, V, U)
            return float(np.mean(np.log(np.maximum(np.abs(h), 1e-300)))), None
    else:
        raise SystemExit("obj?")

    cons = [
        {"type": "eq",
         "fun": lambda z: P.eq_clean(z, N, n, M)[0],
         "jac": lambda z: P.eq_clean(z, N, n, M)[1]},
        {"type": "ineq",
         "fun": lambda z: eta ** 2 - P.junk_energy(z, N, n, M)[0],
         "jac": lambda z: -P.junk_energy(z, N, n, M)[1]},
    ]
    best, bz = np.inf, None
    for s in range(starts):
        if s == 0:
            z0 = P.block_start(n, N, rng)
        elif s < max(2, starts // 3):
            z0 = P.block_start(n, N, rng, noise=0.05 * s)
        else:
            z0 = P.rand_start(n, N, rng)
        # project the start onto C_empty = 1 crudely
        try:
            if obj == "lkill":
                res = minimize(lambda z: f(z)[0], z0, jac=None, constraints=cons,
                               method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12})
            else:
                res = minimize(lambda z: f(z)[0], z0, jac=lambda z: f(z)[1],
                               constraints=cons, method="SLSQP",
                               options={"maxiter": maxiter, "ftol": 1e-12})
        except Exception:
            continue
        z = res.x
        ce, _ = P.eq_clean(z, N, n, M)
        je, _ = P.junk_energy(z, N, n, M)
        if np.abs(ce).max() > 1e-7 or je > eta ** 2 + 1e-9:
            continue
        v = f(z)[0] if obj != "lkill" else f(z)[0]
        if v < best:
            best, bz = v, z.copy()
    return best, bz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="cost", choices=["cost", "prod", "lkill"])
    ap.add_argument("--nu", type=int, default=3)
    ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--Ns", type=int, nargs="*", default=None)
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--levels", type=float, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    n = a.nu - 1
    r = a.q / a.nu
    Ns = a.Ns if a.Ns else ([n + 1] if a.obj != "cost" else [n + 1, n + 2])
    etamax = np.sqrt(2 ** n - 1)
    lv = a.levels if a.levels else list(np.concatenate([
        np.array([0.0, 1e-3, 3e-3, 1e-2, 3e-2]),
        np.linspace(0.05, etamax, 25)]))
    rows = []
    for eta in lv:
        vals = {}
        for N in Ns:
            v, _ = solve(a.obj, n, N, r, eta, a.starts, seed=1234 + int(1e4 * eta))
            vals[N] = v
        v = min(vals.values())
        rows.append({"eta": float(eta), "val": float(v),
                     "perN": {str(k): float(x) for k, x in vals.items()}})
        print(f"eta={eta:.6f}  {a.obj}={v:.12f}", flush=True)
    out = a.out or f"FINAL_RUNS/P2/{a.obj}_nu{a.nu}_q{a.q}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"obj": a.obj, "nu": a.nu, "q": a.q, "r": r, "Ns": Ns, "rows": rows},
              open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
