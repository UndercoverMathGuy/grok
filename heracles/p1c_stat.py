"""P1C / step 3: solve the STATIONARITY SYSTEM directly, to census ALL stationary points
(minima, saddles, maxima), not just the ones an SLSQP descent happens to reach.

The system (all atoms assumed non-zero; see P1C.md for why that is the right normal form):

  (KL)  sum_l c_l prod_{j in T} v_lj = [T empty]                       2^n complex equations
  (i)   c_l * Lambda(v_l) = |c_l|^r / F,   F = sum_l |c_l|^r            N complex equations
  (ii)  Im[ alpha_lj / Lambda(v_l) ] = 0   for every atom l, slot j     n*N real equations

unknowns: phases phi (nN real), atoms c (2N real), dual coefficients lam (2^{n+1} real).
Equation count = unknown count, so this is a square system up to the n-dimensional slot-rotation
gauge; a least-squares root finder from random starts enumerates its solution branches.

WHY THIS IS THE WHOLE OF P1.  Condition (i) alone gives, with sigma = r/(1-r),

    F = ( sum_l |Lambda(v_l)|^{-sigma} )^{1-r} ,

so P1 at stationary points is EXACTLY  sum_l |Lambda(v_l)|^{-sigma} >= n+1.  And if all
|Lambda(v_l)| are equal then all |c_l| are equal, and then
F = N^{1-r} (N m)^r >= (n+1)^{1-r} because sum_l c_l = 1 forces N m >= 1 and Theorem A
forces N >= n+1.  So "equal moduli at every stationary point" => P1.

Outputs: every distinct solution branch found, with F/(n+1)^{1-r}, the modulus spread,
and sum_l |Lambda(v_l)|^{-sigma}.
"""
import argparse, json, os
import numpy as np
from scipy.optimize import least_squares
from p1c_core import Xmat, cost

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1C")


def residual(x, N, n, r):
    nn = N * n
    phi = x[:nn].reshape(N, n)
    c = x[nn:nn + N] + 1j * x[nn + N:nn + 2 * N]
    lam = x[nn + 2 * N:nn + 2 * N + (1 << n)] + 1j * x[nn + 2 * N + (1 << n):]
    X = Xmat(phi)
    out = []
    # (KL)
    b = np.zeros(1 << n, dtype=complex); b[0] = 1.0
    z = X.T @ c - b
    out += [z.real, z.imag]
    # (i)
    F = float(np.sum(np.abs(c) ** r))
    Lv = X @ lam
    w = c * Lv - (np.abs(c) ** r) / F
    out += [w.real, w.imag]
    # (ii)
    for j in range(n):
        lam_a = lam.copy()
        for m in range(1 << n):
            if (m >> j) & 1:
                lam_a[m] = 0.0
        alpha = X @ lam_a
        out.append((alpha / Lv).imag)
    return np.concatenate(out)


def solve_from(seed, N, n, r, minmod):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, (N, n))
    c = rng.normal(size=N) + 1j * rng.normal(size=N)
    c = c / np.sum(c)                      # start feasible in the C_empty = 1 sense
    lam = np.zeros(1 << n, dtype=complex); lam[0] = 1.0
    lam[1:] = 0.1 * (rng.normal(size=(1 << n) - 1) + 1j * rng.normal(size=(1 << n) - 1))
    x0 = np.concatenate([phi.ravel(), c.real, c.imag, lam.real, lam.imag])
    try:
        res = least_squares(residual, x0, args=(N, n, r), xtol=1e-15, ftol=1e-15,
                            gtol=1e-15, max_nfev=4000)
    except Exception:
        return None
    x = res.x
    nn = N * n
    phi = x[:nn].reshape(N, n)
    c = x[nn:nn + N] + 1j * x[nn + N:nn + 2 * N]
    lam = x[nn + 2 * N:nn + 2 * N + (1 << n)] + 1j * x[nn + 2 * N + (1 << n):]
    rn = float(np.abs(residual(x, N, n, r)).max())
    if rn > 1e-9:
        return None
    mods = np.abs(c)
    if mods.min() < minmod:
        return None
    Lv = Xmat(phi) @ lam
    sigma = r / (1 - r)
    return dict(F=cost(c, r), mods=np.sort(mods)[::-1].tolist(),
                spread=float(mods.max() / mods.min()),
                absL=np.sort(np.abs(Lv)).tolist(),
                sumL=float(np.sum(np.abs(Lv) ** (-sigma))),
                lam_empty=complex(lam[0]).real, lam_empty_im=complex(lam[0]).imag,
                resid=rn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:2:3", "2:2:4", "2:2:5", "2:2:6",
                                                   "2:2.5:4", "3:2:4", "3:2:5", "3:2:6"])
    ap.add_argument("--starts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--minmod", type=float, default=1e-3)
    ap.add_argument("--tag", default="stat")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    allc = []
    for cell in a.cells:
        n, q, N = cell.split(":")
        n, N, q = int(n), int(N), float(q)
        nu = n + 1
        r = q / nu
        target = nu ** (1 - r)
        sols = []
        for s in range(a.starts):
            out = solve_from(a.seed + 7919 * s, N, n, r, a.minmod)
            if out is not None:
                sols.append(out)
        # dedupe by (F, sorted moduli)
        seen, uniq = set(), []
        for o in sols:
            key = (round(o["F"], 8), tuple(round(m, 6) for m in o["mods"]))
            if key not in seen:
                seen.add(key)
                uniq.append(o)
        uniq.sort(key=lambda o: o["F"])
        rec = dict(n=n, q=q, N=N, r=r, target=target, nfound=len(sols),
                   nuniq=len(uniq), branches=uniq[:25])
        allc.append(rec)
        print(f"n={n} q={q} N={N} r={r:.4f} target={target:.9f}: "
              f"{len(sols)} converged, {len(uniq)} distinct full-support branches", flush=True)
        for o in uniq[:8]:
            print(f"   F={o['F']:.9f}  ratio={o['F']/target:.9f}  spread={o['spread']:.6f}  "
                  f"sum|L|^-sig={o['sumL']:.6f}  lam0={o['lam_empty']:.6f}"
                  f"{o['lam_empty_im']:+.1e}i  mods={['%.5f' % m for m in o['mods']]}",
                  flush=True)
        if uniq:
            worst = min(uniq, key=lambda o: o["F"])
            print(f"   >>> lowest branch ratio = {worst['F']/target:.12f}", flush=True)
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(allc, f, indent=1)


if __name__ == "__main__":
    main()
