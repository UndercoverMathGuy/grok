"""P2B step 2: measure the two constants P2B's proof of QPL needs, at n = 2.

Gauge-fixed domain: x = (e^{i a1}, e^{i a2}, 1), y = (e^{i b1}, e^{i b2}, 1); the four angles
run over [0, 2pi)^4.  Rotating either slot's phases changes none of the quantities below.

Quantities (all trigonometric polynomials in the four angles, except Delta which is a log
of one):

  S_j = sum_l v_lj ,  Q_j = sum_l v_lj^2 ,  Y = sum_l conj(x_l) y_l ,  Z = sum_l conj(x_l y_l)
  V_j^2 = 27 - 6|S_j|^2 - 3|Q_j|^2 + 2 Re(conj(S_j)^2 Q_j)                     [exact]
  Delta = -(1/2) sum_j log(V_j^2/27) >= D := (1/2) sum_j (1 - V_j^2/27)        [D polynomial]
  ||m||^2 = (|S_1|^2 + |S_2|^2)/9 + |conj(S_1 S_2) - Z|^2/36
  ||G||  <= gB := min( 3 , 1 + |S_1bar S_2 - Y|/6 + max(|S_1|,|S_2|)/3 )       [Hermitian,
           unit diagonal: ||G||_2 <= ||G||_inf; and ||G||_2 <= trace = 3]
  eps    = |L_0| / sqrt(|L_1|^2+|L_2|^2+|L_3|^2), L = the multi-affine polynomial vanishing
           at the three points (cofactors).  Every atom vector on these phases has eta >= eps.

Measured:
  C1(eta0) := sup { [3 eta ||m|| - D] / eta^2 : eps(v) <= eta <= eta0 }
  CG(eta0) := sup { [3 eta ||m|| + 3 Cdel gB eta^2 - D]/eta^2 : eps(v) <= eta <= eta0 }
              = the FULL QPL constant C, with Cdel = (-log(1-delta0)-delta0)/delta0^2,
              delta0 = sqrt3 eta0.
Predictions: C1(eta0) -> 9/8 = c_2 as eta0 -> 0 (the sharp tangent constant of Prop P2-4),
and the argmax sits at the (n+1)-block.
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize

OUT = "FINAL_RUNS/P2B"
SQ3 = np.sqrt(3.0)


def quant(ang):
    a1, a2, b1, b2 = ang
    x = np.array([np.exp(1j * a1), np.exp(1j * a2), 1.0 + 0j])
    y = np.array([np.exp(1j * b1), np.exp(1j * b2), 1.0 + 0j])
    S1, S2 = x.sum(), y.sum()
    Q1, Q2 = np.sum(x * x), np.sum(y * y)
    Y = np.sum(np.conj(x) * y)
    Z = np.sum(np.conj(x * y))
    V2a = 27 - 6 * abs(S1) ** 2 - 3 * abs(Q1) ** 2 + 2 * np.real(np.conj(S1) ** 2 * Q1)
    V2b = 27 - 6 * abs(S2) ** 2 - 3 * abs(Q2) ** 2 + 2 * np.real(np.conj(S2) ** 2 * Q2)
    V2a, V2b = max(V2a, 0.0), max(V2b, 0.0)
    D = 0.5 * ((1 - V2a / 27.0) + (1 - V2b / 27.0))
    Delta = np.inf if (V2a <= 0 or V2b <= 0) else -0.5 * (np.log(V2a / 27) + np.log(V2b / 27))
    m2 = (abs(S1) ** 2 + abs(S2) ** 2) / 9.0 + abs(np.conj(S1 * S2) - Z) ** 2 / 36.0
    gB = min(3.0, 1.0 + abs(np.conj(S1) * S2 - Y) / 6.0 + max(abs(S1), abs(S2)) / 3.0)
    t = x * y
    L3 = -((x[0] - 1) * (y[1] - 1) - (x[1] - 1) * (y[0] - 1))
    L2 = (x[0] - 1) * (t[1] - 1) - (x[1] - 1) * (t[0] - 1)
    L1 = -((y[0] - 1) * (t[1] - 1) - (y[1] - 1) * (t[0] - 1))
    L0 = (y[0] * t[1] - t[0] * y[1]) - (x[0] * t[1] - t[0] * x[1]) + (x[0] * y[1] - y[0] * x[1])
    tail = np.sqrt(abs(L1) ** 2 + abs(L2) ** 2 + abs(L3) ** 2)
    eps = abs(L0) / tail if tail > 1e-13 else 0.0
    return np.sqrt(m2), D, Delta, eps, gB


def inner(mm, d, e, eta0, K):
    """max over eta in [max(e,tiny), eta0] of (3 eta mm - d)/eta^2 * ... i.e. of
    (3 mm/eta - d/eta^2); returns -inf if e > eta0.   K is unused (kept for symmetry)."""
    if e > eta0:
        return -np.inf
    lo = max(e, 1e-9)
    star = 2.0 * d / (3.0 * mm) if mm > 1e-14 else np.inf
    cands = [lo, eta0] + ([star] if lo <= star <= eta0 else [])
    return max(3 * mm / t - d / t ** 2 for t in cands)


def obj(ang, eta0, cdel, withG):
    mm, d, Delta, e, gB = quant(ang)
    v = inner(mm, d, e, eta0, None)
    if v == -np.inf:
        return 1e6 + 1e3 * (e - eta0)
    if withG:
        v += 3.0 * cdel * gB
    return -v


def scan(eta0, cdel, withG, ngrid, npolish, seed):
    rng = np.random.default_rng(seed)
    best, arg = -np.inf, None
    # coarse grid
    g = np.linspace(0, 2 * np.pi, ngrid, endpoint=False)
    A1, A2, B1, B2 = np.meshgrid(g, g, g, g, indexing="ij")
    pts = np.stack([A1.ravel(), A2.ravel(), B1.ravel(), B2.ravel()], axis=1)
    vals = np.array([-obj(p, eta0, cdel, withG) for p in pts])
    order = np.argsort(-vals)[:npolish]
    starts = [pts[i] for i in order]
    for s1 in (1, 2):
        for s2 in (1, 2):
            base = np.array([2 * np.pi * s1 / 3, 2 * np.pi * (3 - s1) / 3,
                             2 * np.pi * s2 / 3, 2 * np.pi * (3 - s2) / 3])
            for sc in (1e-4, 1e-3, 1e-2, 0.05, 0.2):
                for _ in range(6):
                    starts.append(base + sc * rng.normal(size=4))
    for a0 in starts:
        r = minimize(obj, a0, args=(eta0, cdel, withG), method="Nelder-Mead",
                     options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 3000})
        if r.fun < 1e5 and -r.fun > best:
            best, arg = -r.fun, r.x.copy()
    return best, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngrid", type=int, default=14)
    ap.add_argument("--npolish", type=int, default=250)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}

    print("== landmarks (gauge x_3 = y_3 = 1) ==", flush=True)
    land = {}
    for name, ang in [
        ("block", [2 * np.pi / 3, 4 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3]),
        ("block'", [4 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3, 2 * np.pi / 3]),
        ("reflection", [2 * np.pi / 3, 4 * np.pi / 3, 4 * np.pi / 3, 2 * np.pi / 3]),
        ("reflection'", [4 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3]),
    ]:
        mm, d, Delta, e, gB = quant(np.array(ang))
        land[name] = dict(m=mm, D=d, Delta=Delta, eps=e, gB=gB)
        print(f"   {name:12s} ||m||={mm:.9f} D={d:.3e} Delta={Delta:.3e} eps={e:.9f} "
              f"gB={gB:.6f}", flush=True)
    res["landmarks"] = land

    print("\n== the common-phase diagonal y_l = x_l (eps = 0), symmetric slice ==", flush=True)
    rows = []
    for th in np.linspace(1e-4, np.pi - 1e-4, 6001):
        mm, d, Delta, e, gB = quant(np.array([th, -th, th, -th]))
        if d > 1e-14 and np.isfinite(Delta) and Delta > 1e-14:
            rows.append((th, mm ** 2 / Delta, mm ** 2 / d, mm, Delta, d, e, gB))
    kD = int(np.argmax([r[1] for r in rows]))
    kd = int(np.argmax([r[2] for r in rows]))
    print(f"   max ||m||^2/Delta = {rows[kD][1]:.9f} at th={rows[kD][0]:.5f} "
          f"(limit 1/2 = {rows[0][1]:.9f})", flush=True)
    print(f"   max ||m||^2/D     = {rows[kd][2]:.9f} at th={rows[kd][0]:.5f} "
          f"(limit 1/2 = {rows[0][2]:.9f})", flush=True)
    res["diagonal"] = {"max_m2_over_Delta": rows[kD][1], "at_theta_Delta": rows[kD][0],
                       "max_m2_over_D": rows[kd][2], "at_theta_D": rows[kd][0],
                       "limit_Delta": rows[0][1], "limit_D": rows[0][2]}

    print("\n== C1(eta0) = sup [3 eta ||m|| - D]/eta^2   over eps(v) <= eta <= eta0 ==",
          flush=True)
    res["C1"] = {}
    for eta0 in (0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4):
        v, arg = scan(eta0, 0.0, False, a.ngrid, a.npolish, seed=int(1000 * eta0))
        mm, d, Delta, e, gB = quant(arg)
        res["C1"][f"{eta0:.3f}"] = {"C1": v, "arg": list(map(float, arg)),
                                    "m": mm, "D": d, "eps": e, "gB": gB}
        print(f"   eta0={eta0:.3f}: C1 = {v:.6f}   (argmax ||m||={mm:.5f} D={d:.3e} "
              f"eps={e:.5f} gB={gB:.4f})", flush=True)

    print("\n== C(eta0) = C1-style sup including the second-order remainder 3 Cdel ||G|| ==",
          flush=True)
    res["C"] = {}
    for eta0 in (0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3):
        d0 = SQ3 * eta0
        cdel = (-np.log(1 - d0) - d0) / d0 ** 2
        v, arg = scan(eta0, cdel, True, a.ngrid, a.npolish, seed=77 + int(1000 * eta0))
        mm, d, Delta, e, gB = quant(arg)
        thr = 3 * 0.2440169 / v
        res["C"][f"{eta0:.3f}"] = {"C": v, "cdel": cdel, "arg": list(map(float, arg)),
                                   "m": mm, "D": d, "eps": e, "gB": gB,
                                   "chord_threshold": thr}
        print(f"   eta0={eta0:.3f}: Cdel={cdel:.5f}  C = {v:.6f}   gB@argmax={gB:.4f}  "
              f"chord threshold 3*rho_need/C = {thr:.5f} "
              f"({'OK <= eta0' if thr <= eta0 else 'exceeds eta0'})", flush=True)

    json.dump(res, open(f"{OUT}/scan.json", "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/scan.json", flush=True)


if __name__ == "__main__":
    main()
