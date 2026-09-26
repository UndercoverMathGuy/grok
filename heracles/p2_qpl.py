"""P2: the Quadratic Product Lemma (QPL) curve.

    Pi(eta) := min { (n+1)^{n+1} prod_{l=1}^{n+1} |c_l| }
               over N = n+1 atoms with C_empty = 1 and junk energy <= eta^2.

Pi(0) = 1 is the Key Lemma's product step (GLOBAL_OPT (4.2) at R = 1).
The inherited PROVED robustification is Pi(eta) >= (1 - sqrt(2^n-1) eta)^{n+1}, which is
LINEAR in eta.  Second-order perturbation at the block predicts instead

    -log Pi(eta) = c_n eta^2 + O(eta^4),    c_n = (n+1)^2 / (4 n)

(the largest value of  sum_l x_l^2/2 - sum_l y_l^2/2  at unit junk energy, where
c_l = (1 + x_l + i y_l)/(n+1); the embedding phases delta_lj are free currency).
This script measures Pi(eta) globally and tests both statements.
"""
import argparse, json, os
import numpy as np
from scipy.optimize import minimize
import p2_core as P


def pi_min(n, eta, starts, seed=0, maxiter=500):
    N = n + 1
    M = P.masks(n)
    rng = np.random.default_rng(seed)

    def f(z):
        c = z[:N] + 1j * z[N:2 * N]
        a = np.maximum(np.abs(c), 1e-300)
        g = np.zeros(z.size)
        g[:N] = c.real / a ** 2
        g[N:2 * N] = c.imag / a ** 2
        return float(np.sum(np.log(a))), g

    cons = [
        {"type": "eq", "fun": lambda z: P.eq_clean(z, N, n, M)[0],
         "jac": lambda z: P.eq_clean(z, N, n, M)[1]},
        {"type": "ineq", "fun": lambda z: eta ** 2 - P.junk_energy(z, N, n, M)[0],
         "jac": lambda z: -P.junk_energy(z, N, n, M)[1]},
    ]
    best = np.inf
    for s in range(starts):
        z0 = P.block_start(n, N, rng, noise=0.0 if s == 0 else 0.02 * s) \
            if s < starts // 2 else P.rand_start(n, N, rng)
        try:
            res = minimize(lambda z: f(z)[0], z0, jac=lambda z: f(z)[1],
                           constraints=cons, method="SLSQP",
                           options={"maxiter": maxiter, "ftol": 1e-14})
        except Exception:
            continue
        z = res.x
        if np.abs(P.eq_clean(z, N, n, M)[0]).max() > 1e-8:
            continue
        if P.junk_energy(z, N, n, M)[0] > eta ** 2 + 1e-10:
            continue
        best = min(best, f(z)[0])
    return best + (n + 1) * np.log(n + 1)   # = log Pi(eta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--npts", type=int, default=36)
    a = ap.parse_args()
    out = {}
    for n in a.n:
        es = np.sqrt(2 ** n - 1)
        cn = (n + 1) ** 2 / (4.0 * n)
        lv = np.concatenate([np.array([0.0, 3e-3, 1e-2, 3e-2, 0.06]),
                             np.linspace(0.1, es, a.npts)])
        rows = []
        for eta in lv:
            lp = pi_min(n, eta, a.starts, seed=97 + int(1e5 * eta))
            q = (-lp / eta ** 2) if eta > 0 else 0.0
            rows.append({"eta": float(eta), "logPi": float(lp),
                         "deficit_over_eta2": float(q),
                         "lin_proved": float((n + 1) * np.log(max(1 - es * eta, 1e-300))),
                         "quad_pred": float(-cn * eta ** 2)})
            print(f"n={n} eta={eta:.5f} logPi={lp:+.9f}  (-logPi)/eta^2={q:.6f} "
                  f"vs c_n={cn:.6f}", flush=True)
        out[str(n)] = {"c_n_pred": cn, "etastar": float(es), "rows": rows}
    os.makedirs("FINAL_RUNS/P2", exist_ok=True)
    json.dump(out, open("FINAL_RUNS/P2/qpl.json", "w"), indent=1)
    print("wrote FINAL_RUNS/P2/qpl.json")


if __name__ == "__main__":
    main()
