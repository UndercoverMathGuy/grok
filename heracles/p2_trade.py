"""P2: the margin/cost trade, redone with PROVEN minorants of Cl2_r.

Three minorants of Cl2_r(eta) are compared (all tight at eta = 0):

  proven   Theorem P2-1: (C1) sum p >= 1, (C2) prod p >= ((1-delta)/(n+1))^{n+1},
           (C3) p_l >= (1-delta)/2^n,  delta = sqrt(2^n-1) eta.     [unconditional]
  qpl      the same with (C2) replaced by prod p >= (n+1)^{-(n+1)} exp(-c_n eta^2),
           c_n = (n+1)^2/(4n).                                       [conditional on QPL]
  measured the reference curve from p2_measure.py.                   [not a bound]

Two criteria are evaluated on each.  Write A(eta) = Cl2_r(eta)^{1/r},
kappa0 = (n+1)^{(1-r)/r}, and  rho(eta) := [1 - A(eta)/kappa0]/eta.

  (JT)   agent 4 / GLOBAL_OPT4 SS6.1, K frequencies:  (T) holds on the pure class for all
         junk levels up to eta_cap as soon as  rho(eta) <= 1/(2 K sqrt(2^n-1))  there.
  (CH)   P3A SS2.3, K-free:  (T) holds at every junk level and every K as soon as
         rho(eta) <= rho_need := (1 - 1/kappa0)/sqrt(2^n-1)  for all eta in (0, sqrt(2^n-1)].
         Equivalently A must lie above the chord joining (0, kappa0) and (etastar, 1).
"""
import argparse, json, os
import numpy as np
from scipy.optimize import minimize


def prog(n, r, sumfloor, logprodfloor, atomfloor):
    """min sum_{l=1}^{n+1} x_l^r s.t. sum x >= sumfloor, sum log x >= logprodfloor, x >= atomfloor."""
    nb = n + 1
    if logprodfloor <= -700:
        return float(sumfloor) ** r          # only (C1) survives: one atom
    f = lambda x: (float(np.sum(np.exp(r * x))), r * np.exp(r * x))
    cons = [{"type": "ineq", "fun": lambda x: np.sum(np.exp(x)) - sumfloor,
             "jac": lambda x: np.exp(x)},
            {"type": "ineq", "fun": lambda x: np.sum(x) - logprodfloor,
             "jac": lambda x: np.ones_like(x)}]
    lo = np.log(atomfloor) if atomfloor > 0 else -60.0
    bnds = [(lo, 6.0)] * nb
    rng = np.random.default_rng(0)
    starts = [np.full(nb, np.log(sumfloor / nb))]
    for _ in range(50):
        starts.append(np.log(np.maximum(sumfloor * rng.dirichlet(np.full(nb, 0.6)), 1e-9)))
    best = np.inf
    for x0 in starts:
        x0 = np.clip(x0, lo, 6.0)
        try:
            res = minimize(f, x0, jac=True, bounds=bnds, constraints=cons,
                           method="SLSQP", options={"maxiter": 300, "ftol": 1e-14})
        except Exception:
            continue
        x = res.x
        if (np.sum(np.exp(x)) < sumfloor - 1e-9 or np.sum(x) < logprodfloor - 1e-9
                or np.any(np.exp(x) < atomfloor - 1e-9)):
            continue
        best = min(best, float(np.sum(np.exp(r * x))))
    return best


_PI = {}


def _pi_meas(n, eta):
    """log Pi(eta) from the measured product curve (FINAL_RUNS/P2/qpl.json),
    linearly interpolated in eta; -inf once an atom has collapsed."""
    if n not in _PI:
        try:
            d = json.load(open("FINAL_RUNS/P2/qpl.json"))[str(n)]["rows"]
            e = np.array([x["eta"] for x in d])
            lp = np.array([x["logPi"] for x in d])
        except FileNotFoundError:        # fall back on the running log
            e, lp = [], []
            for ln in open("FINAL_RUNS/P2/qpl.log"):
                if ln.startswith(f"n={n} "):
                    e.append(float(ln.split("eta=")[1].split()[0]))
                    lp.append(float(ln.split("logPi=")[1].split()[0]))
            e, lp = np.array(e), np.array(lp)
        bad = np.where(lp < -20)[0]
        if len(bad):
            e, lp = e[:bad[0]], lp[:bad[0]]
        _PI[n] = (e, lp)
    e, lp = _PI[n]
    if eta > e[-1]:
        return -np.inf
    return float(np.interp(eta, e, lp))


def cl2_lower(kind, n, r, eta):
    nb, es = n + 1, np.sqrt(2 ** n - 1)
    delta = es * eta
    atom = max(1 - delta, 0.0) / 2 ** n
    if kind == "proven":
        lpf = nb * np.log(max(1 - delta, 1e-300) / nb)
    elif kind == "qpl":                       # small-eta closed form, c_n = (n+1)^2/(4n)
        cn = nb ** 2 / (4.0 * n)
        lpf = -nb * np.log(nb) - cn * eta ** 2
    elif kind == "meas":                      # the measured product curve
        v = _pi_meas(n, eta)
        lpf = -nb * np.log(nb) + v if np.isfinite(v) else -1e3
        atom = 0.0
    else:
        raise ValueError(kind)
    v = prog(n, r, 1.0, lpf, atom)
    return max(v, 1.0)            # Cl2_r >= (sum|c_l|)^r >= 1 always


def rho_curve(kind, n, r, npts=61):
    es = np.sqrt(2 ** n - 1)
    k0 = (n + 1) ** ((1 - r) / r)
    etas = np.linspace(es / npts, es, npts)
    out = []
    for eta in etas:
        A = cl2_lower(kind, n, r, eta) ** (1.0 / r)
        out.append((float(eta), float(A), float((1 - A / k0) / eta)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["3:2", "3:2.5", "4:2", "4:3"])
    ap.add_argument("--Ks", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--npts", type=int, default=81)
    a = ap.parse_args()
    out = {}
    for cell in a.cells:
        nu, q = cell.split(":")
        nu, q = int(nu), float(q)
        n, r = nu - 1, q / nu
        es = np.sqrt(2 ** n - 1)
        k0 = (n + 1) ** ((1 - r) / r)
        rho_need = (1 - 1 / k0) / es
        cell_out = {"n": n, "q": q, "r": r, "kappa0": k0, "etastar": float(es),
                    "rho_need_chord": float(rho_need), "c_n": (n + 1) ** 2 / (4.0 * n)}
        print(f"\n=== nu={nu} q={q} (r={r:.4f}, kappa0={k0:.6f}, "
              f"etastar={es:.6f}, rho_need={rho_need:.9f}) ===")
        for kind in ("proven", "qpl", "meas"):
            cur = rho_curve(kind, n, r, a.npts)
            rho_sup = max(x[2] for x in cur)
            cell_out[kind] = {"rho_sup": float(rho_sup), "curve": cur}
            # thresholds
            th = {}
            for K in a.Ks:
                lim = 1.0 / (2 * K * es)
                cap = 0.0
                for eta, A, rh in cur:
                    if rh <= lim:
                        cap = eta
                    else:
                        break
                th[str(K)] = float(cap)
            # chord threshold (P3A criterion, K-free)
            cap = 0.0
            for eta, A, rh in cur:
                if rh <= rho_need:
                    cap = eta
                else:
                    break
            cell_out[kind]["eta_cross_JT"] = th
            cell_out[kind]["eta_cross_chord"] = float(cap)
            print(f"  [{kind:6s}] sup_eta rho = {rho_sup:.6f}  "
                  f"(need <= {rho_need:.6f} for the K-free chord)")
            print(f"           chord-criterion threshold eta <= {cap:.4f}")
            print("           (JT) eta_cross by K: " +
                  "  ".join(f"K{K}:{th[str(K)]:.3f}" for K in a.Ks))
        out[cell] = cell_out
    os.makedirs("FINAL_RUNS/P2", exist_ok=True)
    json.dump(out, open("FINAL_RUNS/P2/trade.json", "w"), indent=1)
    print("\nwrote FINAL_RUNS/P2/trade.json")


if __name__ == "__main__":
    main()
