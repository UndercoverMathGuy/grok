"""hr5: independent branch-and-bound for the (7,2) two-feature 'simple bound' of R14, written from the math only.
Bound (R12 + R13):  ratio <= [K1 A1 + K2 A2 - (7/2) min(mu) max(A1 x1, A2 x2)] / [Gamma_s (sum_j (F_j A_j)^{1/s})^s],
F_j = max(1 - c1(s) x_j, 3^{1-s} max(1, x_j)), A1 + A2 = 1, Gamma_s = (7/2) 3^{-s}.
Box bounds by monotonicity (not interval arithmetic): numerator max at A1 hi/A2 hi, x lo; denominator min using
F at s_hi, min over x-interval (V-shape, vertex x=1), power mean at s_lo, Gamma at s_hi.  Float with 1e-9 safety.
Also: pointwise sup with per-point optimal mu (LP) by multistart.
Usage: python hr5_72.py s_lo s_hi TARGET nsteps XMAX"""
import sys
import numpy as np
from scipy.optimize import linprog, minimize

s_lo, s_hi, TARGET, NST, XMAX = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
kap = np.array([[1 - np.cos(2 * np.pi * j * d / 7) for d in (1, 2, 3)] for j in (1, 2)])   # kap[j-1, d-1]
SAFE = 1e-9


def Fval(x, s):
    c1 = 1 - 3 ** (1 - s)
    return max(1 - c1 * x, 3 ** (1 - s) * max(1.0, x))


def Fmin_interval(xa, xb, s):
    if xb == np.inf:
        return 3 ** (1 - s) * xa          # x >= xa >= XMAX >= 1
    xs = min(max(1.0, xa), xb)
    return Fval(xs, s)


def mu_lp(A1, A2, M):
    # minimise 2 sum_d mu_d (kap1 A1 + kap2 A2) - 3.5 t M  s.t. t <= mu_d, 2 sum mu = 1
    c = np.concatenate([2 * (kap[0] * A1 + kap[1] * A2), [-3.5 * M]])
    A_ub = np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]], float)
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(3), A_eq=np.array([[2, 2, 2, 0.]]), b_eq=[1], bounds=[(0, None)] * 4, method='highs')
    mu = np.maximum(r.x[:3], 0); return mu / (2 * mu.sum())


def box_bound(mu, t0, t1, xa, xb, ya, yb, s0, s1):
    K1 = 2 * mu @ kap[0]; K2 = 2 * mu @ kap[1]; mm = mu.min()
    A1hi, A1lo = 1 - t0, 1 - t1; A2hi, A2lo = t1, t0
    M1lo = 0.0 if xb == np.inf else A1lo * xa
    M2lo = 0.0 if yb == np.inf else A2lo * ya
    num = K1 * A1hi + K2 * A2hi - 3.5 * mm * max(M1lo, M2lo)
    num = num * (1 + SAFE) + SAFE
    if num <= 0:
        return num
    y1 = Fmin_interval(xa, xb, s1) * A1lo; y2 = Fmin_interval(ya, yb, s1) * A2lo
    pm = (y1 ** (1 / s0) + y2 ** (1 / s0)) ** s0
    den = 3.5 * 3 ** (-s1) * pm * (1 - SAFE)
    if den <= 0:
        return np.inf
    return num / den


def run():
    ss = np.linspace(s_lo, s_hi, NST + 1)
    nbox = 0; worst = 0; fails = 0
    for k in range(NST):
        s0, s1 = ss[k], ss[k + 1]
        stack = [(0.0, 1.0, xa, xb, ya, yb, 0) for (xa, xb) in [(0, XMAX), (XMAX, np.inf)] for (ya, yb) in [(0, XMAX), (XMAX, np.inf)]]
        while stack:
            t0, t1, xa, xb, ya, yb, dep = stack.pop()
            xc = 0.5 * (xa + xb) if xb < np.inf else 10.0; yc = 0.5 * (ya + yb) if yb < np.inf else 10.0
            tc = 0.5 * (t0 + t1)
            Mc = max(0 if xb == np.inf else (1 - tc) * xc, 0 if yb == np.inf else tc * yc)
            mu = mu_lp(1 - tc, tc, Mc)
            r = box_bound(mu, t0, t1, xa, xb, ya, yb, s0, s1)
            nbox += 1
            if r <= TARGET:
                worst = max(worst, r); continue
            if dep > 45:
                fails += 1
                if fails < 10:
                    print("FAIL", s0, s1, t0, t1, xa, xb, ya, yb, r)
                continue
            w = [t1 - t0, (xb - xa) / 1.75 if xb < np.inf else 0, (yb - ya) / 1.75 if yb < np.inf else 0]
            i = int(np.argmax(w))
            if i == 0:
                m = 0.5 * (t0 + t1); stack += [(t0, m, xa, xb, ya, yb, dep + 1), (m, t1, xa, xb, ya, yb, dep + 1)]
            elif i == 1:
                m = 0.5 * (xa + xb); stack += [(t0, t1, xa, m, ya, yb, dep + 1), (t0, t1, m, xb, ya, yb, dep + 1)]
            else:
                m = 0.5 * (ya + yb); stack += [(t0, t1, xa, xb, ya, m, dep + 1), (t0, t1, xa, xb, m, yb, dep + 1)]
    print(f"[hr5 B&B] s in [{s_lo},{s_hi}] ({NST} s-cells) XMAX={XMAX} TARGET={TARGET}: boxes={nbox} fails={fails} worst={worst:.5f}")


def pointwise_sup(s, starts=300, seed=0):
    rng = np.random.default_rng(seed)

    def val(z):
        tau = min(max(z[0], 1e-9), 1 - 1e-9); x1 = abs(z[1]); x2 = abs(z[2])
        A1, A2 = 1 - tau, tau; M = max(A1 * x1, A2 * x2)
        mu = mu_lp(A1, A2, M)
        num = 2 * mu @ (kap[0] * A1 + kap[1] * A2) - 3.5 * mu.min() * M
        den = 3.5 * 3 ** (-s) * ((Fval(x1, s) * A1) ** (1 / s) + (Fval(x2, s) * A2) ** (1 / s)) ** s
        return num / den
    best = -1; arg = None
    for k in range(starts):
        z0 = np.array([rng.random(), rng.uniform(0, 2.5), rng.uniform(0, 2.5)])
        r = minimize(lambda z: -val(z), z0, method='Nelder-Mead', options=dict(xatol=1e-9, fatol=1e-12, maxiter=3000))
        if -r.fun > best:
            best = -r.fun; arg = r.x
    return best, arg


if __name__ == '__main__':
    run()
    for s in (s_lo, s_hi):
        b, a = pointwise_sup(s, starts=150)
        print(f"[hr5 pointwise] s={s}: sup over (tau,x1,x2) of optimal-mu bound = {b:.5f} at tau={a[0]:.4f} x1={abs(a[1]):.4f} x2={abs(a[2]):.4f}")
