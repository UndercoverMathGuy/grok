"""P3C step 1: INDEPENDENT re-check of P2's chord-criterion refutation at nu = 4.

P3A's K-free sufficient condition for (T) is the "chord criterion"

    rho(eta) := [1 - A(eta)/kappa0] / eta   <=   rho_need := (1 - 1/kappa0)/sqrt(2^n - 1)
    for every eta in (0, sqrt(2^n - 1)],      A(eta) := Cl2_r(eta)^{1/r},
                                              kappa0 := A(0) = (n+1)^{(1-r)/r}.

Geometrically: the certified-amplitude curve A must stay ABOVE the straight
chord joining (0, kappa0) to (eta*, 1).

Cl2_r(eta) is a MINIMUM, so ANY explicit feasible configuration is an upper
bound on it, hence an upper bound on A, hence a LOWER bound on rho.  So a
single exhibited configuration with rho > rho_need refutes the criterion
outright -- no global optimisation needed, only an honest fp64 audit of one
point.  That is what this script produces.

  uv run python p3c_chord.py --cells nu4q2 nu4q3 nu3q2 nu3q2.5 --starts 120
"""
import argparse, json, os
import numpy as np
from scipy.optimize import minimize

import p3c_core as C


def solve_level(n, r, eta, N, starts, rng):
    """min sum_l |c_l|^r  s.t.  C_empty = 1, junk energy <= eta^2, over N atoms."""
    dim = 2 * N + N * n

    def cost(z):
        c = z[:N] + 1j * z[N:2 * N]
        a = np.maximum(np.abs(c), 1e-300)
        g = np.zeros(dim)
        w = r * a ** (r - 2)
        g[:N] = w * c.real
        g[N:2 * N] = w * c.imag
        return float(np.sum(a ** r)), g

    def clean_eq(z):
        c, V = C.unpack(z, N, n)
        Cc = C.coeffs(c, V)[0]
        return np.array([Cc.real - 1.0, Cc.imag])

    def junk_ineq(z):
        c, V = C.unpack(z, N, n)
        Cc = C.coeffs(c, V)
        return eta ** 2 - float(np.sum(np.abs(Cc[1:]) ** 2))

    cons = [{"type": "eq", "fun": clean_eq},
            {"type": "ineq", "fun": junk_ineq}]

    best = (np.inf, None)
    for s in range(starts):
        if s == 0:
            c0, th0 = C.block(n)
            c0 = np.concatenate([c0, np.zeros(N - len(c0), dtype=complex)])
            th0 = np.vstack([th0, rng.uniform(0, 2 * np.pi, (N - th0.shape[0], n))])
            z0 = C.pack(c0, th0)
        elif s % 4 == 1:
            # one dominant atom + small others: the max-junk end
            c0 = np.zeros(N, dtype=complex)
            c0[0] = 1.0
            c0[1:] = 0.05 * (rng.standard_normal(N - 1) + 1j * rng.standard_normal(N - 1))
            z0 = C.pack(c0, rng.uniform(0, 2 * np.pi, (N, n)))
        elif s % 4 == 2:
            # perturbed block
            c0, th0 = C.block(n)
            c0 = np.concatenate([c0, np.zeros(N - len(c0), dtype=complex)])
            th0 = np.vstack([th0, rng.uniform(0, 2 * np.pi, (N - th0.shape[0], n))])
            z0 = C.pack(c0, th0) + 0.4 * rng.standard_normal(dim)
        else:
            c0 = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / N
            z0 = C.pack(c0, rng.uniform(0, 2 * np.pi, (N, n)))
        try:
            res = minimize(lambda z: cost(z)[0], z0, jac=lambda z: cost(z)[1],
                           constraints=cons, method="SLSQP",
                           options={"maxiter": 800, "ftol": 1e-14})
        except Exception:
            continue
        z = res.x
        c, V = C.unpack(z, N, n)
        cl, et, co = C.report(c, V, r)
        # HARD audit: only accept strictly feasible points (tiny tolerances)
        if abs(cl - 1.0) > 1e-9 or et > eta * (1 + 1e-9) + 1e-12:
            continue
        if co < best[0]:
            best = (co, z.copy())
    return best


CELLS = {"nu3q2": (3, 2.0), "nu3q2.5": (3, 2.5), "nu4q2": (4, 2.0),
         "nu4q3": (4, 3.0), "nu5q2": (5, 2.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["nu4q2", "nu4q3", "nu3q2", "nu3q2.5"])
    ap.add_argument("--starts", type=int, default=120)
    ap.add_argument("--levels", type=int, default=25)
    ap.add_argument("--out", default="FINAL_RUNS/P3C/chord_recheck.json")
    a = ap.parse_args()

    out = {}
    for key in a.cells:
        nu, q = CELLS[key]
        n, r = nu - 1, q / nu
        k0, rn = C.kappa0(n, r), C.rho_need(n, r)
        estar = np.sqrt(2 ** n - 1.0)
        rng = np.random.default_rng(20260921)
        levels = np.linspace(estar / a.levels, estar, a.levels)
        rows = []
        for eta in levels:
            bb = (np.inf, None, None)
            for N in (n + 1, n + 2, n + 3):
                co, z = solve_level(n, r, eta, N, a.starts, rng)
                if co < bb[0]:
                    bb = (co, z, N)
            co, z, N = bb
            c, V = C.unpack(z, N, n)
            cl, et, co2 = C.report(c, V, r)
            rho = C.rho_of(eta, co2, n, r)
            rows.append({"eta": float(eta), "N": int(N), "cost": float(co2),
                         "A": float(co2 ** (1 / r)), "rho": float(rho),
                         "eta_audit": float(et), "clean_err": float(abs(cl - 1.0)),
                         "c_re": c.real.tolist(), "c_im": c.imag.tolist(),
                         "theta": np.angle(V).tolist()})
            print(f"{key} eta={eta:.6f} N={N} cost={co2:.12f} A={co2**(1/r):.9f} "
                  f"rho={rho:.9f} (need <= {rn:.9f})", flush=True)
        best = max(rows, key=lambda x: x["rho"])
        out[key] = {"nu": nu, "q": q, "r": r, "n": n, "kappa0": k0,
                    "eta_star": float(estar), "rho_need": rn,
                    "rho_sup": best["rho"], "argmax_eta": best["eta"],
                    "chord_holds": bool(best["rho"] <= rn * (1 + 1e-12)),
                    "rows": rows}
        print(f"== {key}: sup rho = {best['rho']:.9f} at eta={best['eta']:.6f}; "
              f"rho_need={rn:.9f}; chord "
              f"{'HOLDS' if out[key]['chord_holds'] else 'FAILS'}\n", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
