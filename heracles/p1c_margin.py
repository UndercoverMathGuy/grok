"""P1C / step 4c: the hybrid bound, fast and with the SAFETY MARGIN reported.

Same three proved ingredients as p1c_hybrid2.py; this version precomputes the vertex
basis inverses once (the polyhedron's normal directions do not depend on (R, s), only its
right-hand side does), so the whole (R, s) box can be swept on a fine grid, and it reports

  * R1 : the largest R for which the Key-Lemma-with-a-tail bound ALONE already suffices,
         i.e. A_G(R) + R^r >= (n+1)^{1-r}                        [analytic, one variable]
  * R2 : the smallest R for which the crude bound alone suffices,
         (n+1) s^r + R s^{r-1} >= (n+1)^{1-r} for every s > 0    [analytic, closed form]
  * m(R) = min over s of Val(R, s) / (n+1)^{1-r} on the grid, and its minimum over the
    interval [R1, R2] where the ladder is actually needed -- the SAFETY MARGIN of the
    certificate.  A margin comfortably above 1 is what makes the grid sweep meaningful.
"""
import argparse, json, os
from itertools import combinations
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1C")

S_D = {1: 1.0, 2: 2.0 / np.sqrt(3.0), 3: 9.0 / 7.0,
       4: 1.380900987, 5: 1.452990673, 6: 1.509590371}
KAPPA = {d: (1.0 - 2.0 ** (-d)) / S_D[d] for d in S_D}


def D_h(h, n):
    if h == 0:
        return 1.0
    best = 0.0

    def rec(rem, parts, prod):
        nonlocal best
        if len(parts) == h:
            if rem == 0:
                best = max(best, prod)
            return
        for d in range(1, rem - (h - len(parts) - 1) + 1):
            rec(rem - d, parts + [d], prod * KAPPA[d])
    rec(n, [], 1.0)
    return best


def A_G(R, n, r):
    return (n + 1) ** (1 - r) * np.maximum(1.0 - 2 ** n * np.asarray(R, dtype=float), 0.0) ** r


def R2(n, r):
    K = (r * (n + 1) / (1.0 - r)) ** (1.0 - r) / r
    return ((n + 1) ** (1 - r) / K) ** (1.0 / r)


def R1(n, r, ngrid=2000001):
    """largest R with A_G(R) + R^r >= (n+1)^{1-r}; 0 if none beyond 0."""
    t = (n + 1) ** (1 - r)
    Rs = np.linspace(0.0, 2.0 ** (-n), ngrid)
    ok = (A_G(Rs, n, r) + Rs ** r) >= t - 1e-15
    # the set is an interval [0, R1] near 0
    bad = np.argmax(~ok)
    if ok.all():
        return float(Rs[-1])
    return float(Rs[max(bad - 1, 0)])


class Ladder:
    """Precomputed vertex bases for  min sum_{l=1}^{n} x_l^r  over
       x_1 >= ... >= x_n >= s,  sum_{h < l <= n} x_l >= D_h(n) - R - s  (h = 0..n-1),
       plus the pure feasibility test  s + R >= D_n(n)."""

    def __init__(self, n, r):
        self.n, self.r = n, r
        m = n
        rows, kindA, kindB = [], [], []
        # ladder rows h = 0..n-1 : coefficient vector, rhs = D_h - R - s
        for h in range(0, n):
            a = np.zeros(m); a[h:] = 1.0
            rows.append(a); kindA.append(D_h(h, n)); kindB.append(-1.0)   # rhs = A - R - s
        for i in range(m - 1):
            a = np.zeros(m); a[i] = 1.0; a[i + 1] = -1.0
            rows.append(a); kindA.append(0.0); kindB.append(0.0)          # rhs = 0
        a = np.zeros(m); a[m - 1] = 1.0
        rows.append(a); kindA.append(0.0); kindB.append(1.0)              # rhs = s
        self.A = np.array(rows)
        self.cA = np.array(kindA)          # constant part of rhs
        self.cB = np.array(kindB)          # coefficient: -1 -> (-R - s); +1 -> (+s); 0 -> 0
        self.Dn = D_h(n, n)
        self.bases = []
        for idx in combinations(range(len(self.A)), m):
            M = self.A[list(idx)]
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            self.bases.append((list(idx), np.linalg.inv(M)))

    def rhs(self, R, s):
        b = self.cA.copy()
        b = np.where(self.cB < 0, self.cA - R - s, b)
        b = np.where(self.cB > 0, s, b)
        b = np.where(self.cB == 0, self.cA, b)
        return b

    def value(self, R, s):
        if s + R < self.Dn - 1e-12:
            return np.inf                   # infeasible: no configuration with these (R, s)
        b = self.rhs(R, s)
        best = np.inf
        for idx, Minv in self.bases:
            x = Minv @ b[idx]
            if np.any(x < -1e-12):
                continue
            if np.any(self.A @ x < b - 1e-9):
                continue
            best = min(best, float(np.sum(np.maximum(x, 0.0) ** self.r)))
        if not np.isfinite(best):
            return np.inf
        return best + s ** self.r


def profile(n, q, nR=1201, nS=1201):
    nu = n + 1
    r = q / nu
    t = nu ** (1 - r)
    lad = Ladder(n, r)
    Rmax, smax = R2(n, r), 1.0 / nu
    Rs = np.linspace(0.0, Rmax, nR)
    ss = np.linspace(0.0, smax, nS)
    mR = np.empty(nR)
    argS = np.empty(nR)
    for i, R in enumerate(Rs):
        best, bs = np.inf, np.nan
        ag = A_G(R, n, r)
        for s in ss:
            L = lad.value(R, s)
            if not np.isfinite(L):
                continue
            if s <= 0.0:
                if R > 0.0:
                    continue          # R > 0 forces a tail atom of modulus <= s, so s > 0
                tail = 0.0
            else:
                tail = max(R ** r, R * s ** (r - 1.0))
            v = max(ag, L) + tail
            if v < best:
                best, bs = v, s
        mR[i], argS[i] = best / t, bs
    return dict(n=n, q=q, r=r, target=t, Rs=Rs, mR=mR, argS=argS,
                R1=R1(n, r), R2=float(Rmax), smax=float(smax))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:2", "2:2.25", "2:2.5", "2:2.75",
                                                   "3:2", "3:2.5", "3:3", "3:3.5",
                                                   "4:2", "4:3", "5:2"])
    ap.add_argument("--nR", type=int, default=801)
    ap.add_argument("--nS", type=int, default=801)
    ap.add_argument("--tag", default="margin")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cell in a.cells:
        n, q = cell.split(":")
        n, q = int(n), float(q)
        P = profile(n, q, a.nR, a.nS)
        Rs, mR = P["Rs"], P["mR"]
        k = int(np.argmin(mR))
        mid = (Rs >= P["R1"]) & (Rs <= P["R2"])
        kmid = int(np.argmin(np.where(mid, mR, np.inf)))
        rec = dict(n=n, q=q, r=P["r"], target=P["target"], R1=P["R1"], R2=P["R2"],
                   min_ratio=float(mR.min()), argmin_R=float(Rs[k]),
                   min_ratio_mid=float(mR[kmid]), argmin_R_mid=float(Rs[kmid]),
                   argmin_s_mid=float(P["argS"][kmid]),
                   profile=[[float(x), float(y)] for x, y in zip(Rs[::max(1, len(Rs)//60)],
                                                                 mR[::max(1, len(mR)//60)])])
        rows.append(rec)
        print(f"n={n} nu={n+1} q={q} r={P['r']:.5f}: HYBRID min ratio={mR.min():.9f} "
              f"at R={Rs[k]:.5f};  over R>0: {mR[kmid]:.6f} at R={Rs[kmid]:.5f}, "
              f"s={P['argS'][kmid]:.5f};  R1={P['R1']:.5f} R2={P['R2']:.5f}", flush=True)
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
