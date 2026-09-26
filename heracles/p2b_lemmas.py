"""P2B step 3: the two finite lemmas the analytic reduction needs, at n = 2.

The reduction (P2B.md SS4) bounds the kill-point moment norm by

    ||m||^2 = (|S_1|^2+|S_2|^2)/9 + |W|^2/36 ,   W = conj(S_1 S_2) - Z ,
    |W| <= |S_1||S_2| + |Z| ,

so everything rests on two statements.

 (PS)  PER-SLOT.  For three unit-modulus numbers z_1,z_2,z_3 with S = sum z_l, Q = sum z_l^2,
       V^2 = 27 - 6|S|^2 - 3|Q|^2 + 2Re(conj(S)^2 Q) and Delta_1 = -(1/2)log(V^2/27):

            8|S|^2 + |Q|^2  <=  36 Delta_1 .

       Equality holds to second order at the cube-roots-of-unity configuration, which is
       exactly the (n+1)-block's slot.  This is the statement that makes the final constant
       9/8 = c_2 SHARP.  Two real dimensions after fixing the rotation.

 (P3)  CROSS-SLOT.  Three points (x_l, y_l) on T^2 lie on the graph of a unique
       circle-preserving Moebius map y = e^{i th}(x - a)/(1 - conj(a) x); the smallest junk
       energy the phases admit is eps = |a|/sqrt(2 + |a|^2).  Claim:

            |Z| <= (|Q_1| + |Q_2|)/2 + E ,     Z = sum_l conj(x_l y_l) ,   E = O(|a|^2) .

       At a = 0 (the common-phase family y_l = e^{i th} x_l) the three quantities are equal,
       and the first variation in a vanishes, so E is SECOND order in a -- hence O(eps^2),
       which is invisible at the order that decides the QPL constant.

This script measures both, reports the worst ratios found, and checks the Moebius/eps
dictionary (a from the cofactor null vector; eps = |a|/sqrt(2+|a|^2)).
"""
import argparse, json, os
import numpy as np
from scipy.optimize import minimize

OUT = "FINAL_RUNS/P2B"


# ------------------------------------------------------------------ per-slot (PS)
def slot(ang):
    z = np.array([np.exp(1j * ang[0]), np.exp(1j * ang[1]), 1.0 + 0j])
    S, Q = z.sum(), np.sum(z * z)
    V2 = 27 - 6 * abs(S) ** 2 - 3 * abs(Q) ** 2 + 2 * np.real(np.conj(S) ** 2 * Q)
    V2 = max(V2, 0.0)
    d1 = np.inf if V2 <= 0 else -0.5 * np.log(V2 / 27.0)
    return abs(S), abs(Q), d1, V2


def ps_ratio(ang):
    s, q, d1, _ = slot(ang)
    if not np.isfinite(d1):
        return 0.0
    if d1 <= 1e-15:
        return 0.0
    return (8 * s * s + q * q) / (36 * d1)


def ps_scan(ngrid=900):
    g = np.linspace(0, 2 * np.pi, ngrid, endpoint=False)
    best, arg = 0.0, None
    for a1 in g:
        for a2 in g:
            r = ps_ratio((a1, a2))
            if r > best:
                best, arg = r, (a1, a2)
    return best, arg


def ps_polish(arg, nrest=400, seed=3):
    rng = np.random.default_rng(seed)
    best, ba = 0.0, None
    starts = [np.array(arg)] + [rng.uniform(0, 2 * np.pi, 2) for _ in range(nrest)]
    base = np.array([2 * np.pi / 3, 4 * np.pi / 3])
    for sc in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.4):
        for _ in range(30):
            starts.append(base + sc * rng.normal(size=2))
            starts.append(np.array([4 * np.pi / 3, 2 * np.pi / 3]) + sc * rng.normal(size=2))
    for a0 in starts:
        r = minimize(lambda a: -ps_ratio(a), a0, method="Nelder-Mead",
                     options={"xatol": 1e-13, "fatol": 1e-15, "maxiter": 4000})
        if -r.fun > best:
            best, ba = -r.fun, r.x.copy()
    return best, ba


# ------------------------------------------------------------------ cross-slot (P3)
def cfg(ang):
    x = np.array([np.exp(1j * ang[0]), np.exp(1j * ang[1]), 1.0 + 0j])
    y = np.array([np.exp(1j * ang[2]), np.exp(1j * ang[3]), 1.0 + 0j])
    return x, y


def cross(ang):
    x, y = cfg(ang)
    Q1, Q2 = np.sum(x * x), np.sum(y * y)
    Z = np.sum(np.conj(x * y))
    t = x * y
    L3 = -((x[0] - 1) * (y[1] - 1) - (x[1] - 1) * (y[0] - 1))
    L2 = (x[0] - 1) * (t[1] - 1) - (x[1] - 1) * (t[0] - 1)
    L1 = -((y[0] - 1) * (t[1] - 1) - (y[1] - 1) * (t[0] - 1))
    L0 = (y[0] * t[1] - t[0] * y[1]) - (x[0] * t[1] - t[0] * x[1]) + (x[0] * y[1] - y[0] * x[1])
    tail = np.sqrt(abs(L1) ** 2 + abs(L2) ** 2 + abs(L3) ** 2)
    eps = abs(L0) / tail if tail > 1e-13 else 0.0
    aa = abs(L3) / abs(L2) if abs(L2) > 1e-13 else np.inf     # |alpha| = |L_3/L_2|
    F = abs(Z) - 0.5 * (abs(Q1) + abs(Q2))
    return F, eps, aa, abs(Z), abs(Q1), abs(Q2)


def moebius_check(trials, rng):
    """Build the configuration FROM (x, theta, alpha) and check eps = |a|/sqrt(2+|a|^2)
    and |alpha| = |L_3/L_2|."""
    w1 = w2 = 0.0
    for _ in range(trials):
        th = rng.uniform(0, 2 * np.pi)
        a = (rng.normal() + 1j * rng.normal()) * rng.uniform(0, 0.4)
        xa = rng.uniform(0, 2 * np.pi, 3)
        x = np.exp(1j * xa)
        y = np.exp(1j * th) * (x - a) / (1 - np.conj(a) * x)
        # re-gauge to x_3 = y_3 = 1
        xg, yg = x / x[2], y / y[2]
        ang = np.array([np.angle(xg[0]), np.angle(xg[1]), np.angle(yg[0]), np.angle(yg[1])])
        F, eps, aa, *_ = cross(ang)
        w1 = max(w1, abs(eps - abs(a) / np.sqrt(2 + abs(a) ** 2)))
        w2 = max(w2, abs(aa - abs(a)))
    return w1, w2


def p3_scan(rng, trials, amax):
    """sup of F/|a|^2 over random configurations with |a| <= amax."""
    best, rec = -np.inf, None
    for _ in range(trials):
        th = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, amax)
        a = r * np.exp(1j * rng.uniform(0, 2 * np.pi))
        x = np.exp(1j * rng.uniform(0, 2 * np.pi, 3))
        y = np.exp(1j * th) * (x - a) / (1 - np.conj(a) * x)
        Q1, Q2 = np.sum(x * x), np.sum(y * y)
        Z = np.sum(np.conj(x * y))
        F = abs(Z) - 0.5 * (abs(Q1) + abs(Q2))
        if r > 1e-9 and F / r ** 2 > best:
            best, rec = F / r ** 2, (float(r), float(F), float(abs(Z)))
    return best, rec


def p3_polish(amax, nstart, seed):
    """maximise F/|a|^2 by local search over (x angles, theta, alpha) with |a| <= amax."""
    rng = np.random.default_rng(seed)

    def f(p):
        xa, th, ar, ai = p[:3], p[3], p[4], p[5]
        r = np.hypot(ar, ai)
        if r > amax or r < 1e-7:
            return 1e6
        a = ar + 1j * ai
        x = np.exp(1j * xa)
        y = np.exp(1j * th) * (x - a) / (1 - np.conj(a) * x)
        Q1, Q2 = np.sum(x * x), np.sum(y * y)
        Z = np.sum(np.conj(x * y))
        return -(abs(Z) - 0.5 * (abs(Q1) + abs(Q2))) / r ** 2

    best, ba = -np.inf, None
    for _ in range(nstart):
        p0 = np.concatenate([rng.uniform(0, 2 * np.pi, 4),
                             rng.uniform(-amax / 2, amax / 2, 2)])
        r = minimize(f, p0, method="Nelder-Mead",
                     options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 4000})
        if r.fun < 1e5 and -r.fun > best:
            best, ba = -r.fun, r.x.copy()
    return best, ba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngrid", type=int, default=700)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    res = {}

    print("== (PS)  8|S|^2 + |Q|^2 <= 36 Delta_1  for three points on the circle ==", flush=True)
    b, arg = ps_scan(a.ngrid)
    print(f"   grid ({a.ngrid}^2): max ratio = {b:.9f} at {np.array(arg)}", flush=True)
    b2, arg2 = ps_polish(arg)
    print(f"   polished        : max ratio = {b2:.12f} at {arg2}", flush=True)
    s, q, d1, V2 = slot(arg2) if arg2 is not None else (0, 0, 0, 0)
    print(f"   at the argmax: |S|={s:.6f} |Q|={q:.6f} Delta_1={d1:.3e} V^2={V2:.6f}", flush=True)
    res["PS"] = {"grid_max": b, "polished_max": b2, "argmax": list(map(float, arg2)),
                 "S": s, "Q": q, "Delta1": d1}
    print(f"   VERDICT: (PS) is {'CONSISTENT (ratio <= 1)' if b2 <= 1 + 1e-9 else 'REFUTED'}",
          flush=True)

    print("\n== Moebius dictionary: eps = |a|/sqrt(2+|a|^2), |a| = |L_3/L_2| ==", flush=True)
    w1, w2 = moebius_check(4000, rng)
    res["moebius"] = {"eps_err": w1, "alpha_err": w2}
    print(f"   max |eps - |a|/sqrt(2+|a|^2)| = {w1:.3e}", flush=True)
    print(f"   max ||L_3/L_2| - |a||         = {w2:.3e}", flush=True)

    print("\n== (P3)  F := |Z| - (|Q_1|+|Q_2|)/2  <=  E ; is F/|a|^2 bounded? ==", flush=True)
    res["P3"] = {}
    for amax in (0.05, 0.1, 0.2, 0.3, 0.5):
        v1, rec = p3_scan(rng, 120000, amax)
        v2, pa = p3_polish(amax, 260, seed=int(1000 * amax))
        res["P3"][f"{amax:.2f}"] = {"random_sup": v1, "polished_sup": max(v1, v2)}
        print(f"   |a| <= {amax:.2f}:  sup F/|a|^2  random = {v1:.6f}   polished = {v2:.6f}",
              flush=True)

    json.dump(res, open(f"{OUT}/lemmas.json", "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/lemmas.json", flush=True)


if __name__ == "__main__":
    main()
