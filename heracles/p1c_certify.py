"""P1C / step 4d: a CELL-WISE RIGOROUS certificate for the hybrid bound (Theorem H).

p1c_margin.py evaluates Val(R, s) at grid POINTS, which certifies nothing between them.
This script instead computes, for every grid CELL [R_lo, R_hi] x [s_lo, s_hi], a guaranteed
LOWER bound on Val over the whole cell, using only monotonicity in each variable:

  * A_G(R) = (n+1)^{1-r} max(1 - 2^n R, 0)^r is DECREASING in R      -> evaluate at R_hi
  * R^r is INCREASING in R                                           -> evaluate at R_lo
  * R s^{r-1} is increasing in R and decreasing in s (r < 1)         -> evaluate at (R_lo, s_hi)
  * Lad(R, s) = min over vertex bases of sum_l x_l(R,s)^r + s^r, and every vertex
    x(R,s) = M^{-1} b(R,s) is AFFINE in (R, s) because the polyhedron's normals are fixed
    and only its right-hand side moves.  So each coordinate x_l = p + qR + ws is minimised
    over the cell at an explicit corner, and t -> t^r is increasing:
        Lad >= min over bases of [ sum_l max(x_l^min, 0)^r ] + s_lo^r .
    A basis is skipped only when it is provably infeasible on the WHOLE cell (some
    constraint's affine slack has a negative maximum over the cell), so the minimum is
    taken over a superset of the feasible vertices and the bound stays valid.

Outside the swept box the bound is analytic (see P1C.md Sec 3.2):
  R <= R1  : A_G(R) + R^r >= (n+1)^{1-r}   (f(rho) is unimodal with f(0) = 0)
  R >= R2  : (n+1) s^r + R s^{r-1} >= K_n(r) R^r >= (n+1)^{1-r}
  s >= 1/(n+1) : F >= (n+1) s^r > (n+1)^{1-r}

So a reported cell minimum >= (n+1)^{1-r} over [R1, R2] x [0, 1/(n+1)] closes (P1) for that
cell of (n, q), up to float64 rounding.
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
    return (n + 1) ** (1 - r) * max(1.0 - 2 ** n * R, 0.0) ** r


def K_n(r, n):
    return (r * (n + 1) / (1.0 - r)) ** (1.0 - r) / r


def R2(n, r):
    return ((n + 1) ** (1 - r) / K_n(r, n)) ** (1.0 / r)


def R1(n, r):
    """largest R with A_G(R) + R^r >= (n+1)^{1-r}, by bisection on the unimodal f."""
    t = (n + 1) ** (1 - r)
    f = lambda R: A_G(R, n, r) + R ** r - t
    lo, hi = 0.0, 2.0 ** (-n)
    if f(hi) >= 0:
        return hi
    # f > 0 just right of 0; bisect for the first sign change
    a = 1e-12
    while f(a) < 0 and a < hi:
        a *= 2
    b = hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(m) >= 0:
            a = m
        else:
            b = m
    return a


class Bases:
    """Affine vertex parametrisation of the ladder polyhedron in x_1..x_n.

    Rows (all of the form  A_k . x  >=  b_k(R, s)  with b affine):
      h = 0..n-1 : a = (0..0,1..1) starting at h,   b = D_h(n) - R - s
      order      : x_i - x_{i+1} >= 0,               b = 0
      last       : x_n >= s,                         b = s
    Feasibility side condition (no x in it): s + R >= D_n(n).
    """

    def __init__(self, n, r):
        self.n, self.r, m = n, r, n
        rows, b0, bR, bs = [], [], [], []
        for h in range(0, n):
            a = np.zeros(m); a[h:] = 1.0
            rows.append(a); b0.append(D_h(h, n)); bR.append(-1.0); bs.append(-1.0)
        for i in range(m - 1):
            a = np.zeros(m); a[i] = 1.0; a[i + 1] = -1.0
            rows.append(a); b0.append(0.0); bR.append(0.0); bs.append(0.0)
        a = np.zeros(m); a[m - 1] = 1.0
        rows.append(a); b0.append(0.0); bR.append(0.0); bs.append(1.0)
        self.A = np.array(rows)
        self.b0 = np.array(b0); self.bR = np.array(bR); self.bs = np.array(bs)
        self.Dn = D_h(n, n)
        self.items = []
        for idx in combinations(range(len(self.A)), m):
            M = self.A[list(idx)]
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            Mi = np.linalg.inv(M)
            i = list(idx)
            self.items.append((i, Mi @ self.b0[i], Mi @ self.bR[i], Mi @ self.bs[i]))
        # vectorised form: P, Q, W are (B, n); CP, CQ, CW are (B, K) constraint-slack coeffs
        self.P = np.array([it[1] for it in self.items])
        self.Q = np.array([it[2] for it in self.items])
        self.W = np.array([it[3] for it in self.items])
        self.CP = self.P @ self.A.T - self.b0[None, :]
        self.CQ = self.Q @ self.A.T - self.bR[None, :]
        self.CW = self.W @ self.A.T - self.bs[None, :]

    def lower_bound(self, Rlo, Rhi, slo, shi):
        """Guaranteed lower bound on Lad(R, s) over the cell (vectorised over bases)."""
        if shi + Rhi < self.Dn - 1e-15:
            return np.inf                     # no configuration exists in this cell
        Rmin = np.where(self.Q > 0, Rlo, Rhi); smin = np.where(self.W > 0, slo, shi)
        Rmax = np.where(self.Q > 0, Rhi, Rlo); smax = np.where(self.W > 0, shi, slo)
        Xmin = self.P + self.Q * Rmin + self.W * smin
        Xmax = self.P + self.Q * Rmax + self.W * smax
        ok = ~np.any(Xmax < -1e-12, axis=1)
        hi = (self.CP + np.where(self.CQ > 0, self.CQ * Rhi, self.CQ * Rlo)
              + np.where(self.CW > 0, self.CW * shi, self.CW * slo))
        ok &= ~np.any(hi < -1e-9, axis=1)
        if not np.any(ok):
            return np.inf
        vals = np.sum(np.maximum(Xmin[ok], 0.0) ** self.r, axis=1)
        return float(vals.min()) + slo ** self.r

    def lower_bound_slow(self, Rlo, Rhi, slo, shi):
        if shi + Rhi < self.Dn - 1e-15:
            return np.inf
        best = np.inf
        for idx, p, q, w in self.items:
            # x_l(R,s) = p + q R + w s : min over the cell at an explicit corner
            Rmin = np.where(q > 0, Rlo, Rhi)
            smin = np.where(w > 0, slo, shi)
            xmin = p + q * Rmin + w * smin
            Rmax = np.where(q > 0, Rhi, Rlo)
            smax = np.where(w > 0, shi, slo)
            xmax = p + q * Rmax + w * smax
            if np.any(xmax < -1e-12):
                continue                       # x_l < 0 on the whole cell: not a vertex here
            # constraint slacks A x - b, affine in (R, s); skip only if provably infeasible
            infeas = False
            for k in range(len(self.A)):
                cp = float(self.A[k] @ p) - self.b0[k]
                cq = float(self.A[k] @ q) - self.bR[k]
                cw = float(self.A[k] @ w) - self.bs[k]
                hi = cp + (cq * Rhi if cq > 0 else cq * Rlo) + (cw * shi if cw > 0 else cw * slo)
                if hi < -1e-9:
                    infeas = True
                    break
            if infeas:
                continue
            best = min(best, float(np.sum(np.maximum(xmin, 0.0) ** self.r)))
        if not np.isfinite(best):
            return np.inf
        return best + slo ** self.r


def certify(n, q, nR=400, nS=400):
    nu = n + 1
    r = q / nu
    t = nu ** (1 - r)
    B = Bases(n, r)
    r1, r2 = R1(n, r), R2(n, r)
    smax = 1.0 / nu
    Rg = np.linspace(r1, r2, nR + 1)
    sg = np.linspace(0.0, smax, nS + 1)
    worst, arg = np.inf, None
    for i in range(nR):
        Rlo, Rhi = Rg[i], Rg[i + 1]
        ag = A_G(Rhi, n, r)                      # A_G is decreasing in R
        tail_flat = Rlo ** r
        for j in range(nS):
            slo, shi = sg[j], sg[j + 1]
            lad = B.lower_bound(Rlo, Rhi, slo, shi)
            if not np.isfinite(lad):
                continue
            tail = max(tail_flat, Rlo * shi ** (r - 1.0)) if shi > 0 else tail_flat
            v = max(ag, lad) + tail
            if v < worst:
                worst, arg = v, (float(Rlo), float(slo))
    return dict(n=n, q=q, r=r, target=t, R1=r1, R2=r2, smax=smax, nR=nR, nS=nS,
                cert_min=float(worst), cert_ratio=float(worst / t), argcell=arg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:2", "2:2.5", "3:2", "3:3", "4:2", "5:2"])
    ap.add_argument("--nR", type=int, default=400)
    ap.add_argument("--nS", type=int, default=400)
    ap.add_argument("--tag", default="certify")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cell in a.cells:
        n, q = cell.split(":")
        n, q = int(n), float(q)
        rec = certify(n, q, a.nR, a.nS)
        rows.append(rec)
        verdict = "CLOSES" if rec["cert_ratio"] >= 1.0 else "does NOT close"
        print(f"n={n} nu={n+1} q={q} r={rec['r']:.5f}: target={rec['target']:.9f}  "
              f"cell-wise certified min over [R1,R2]x[0,1/nu] = {rec['cert_min']:.9f} "
              f"(ratio {rec['cert_ratio']:.6f})  -> {verdict};  "
              f"R1={rec['R1']:.6f} R2={rec['R2']:.6f}  grid {a.nR}x{a.nS}", flush=True)
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
