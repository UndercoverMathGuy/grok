"""P1B: the `r -> 1` window, by reduction to ALIGNED phases plus a dual (test-polynomial) bound.

THE REDUCTION.  Put S := sum_l |c_l| >= |A| = 1.  Subadditivity gives cost >= S^r, so P1 is
immediate whenever S >= (n+1)^{(1-r)/r}.  As r -> 1 that threshold tends to 1, so the ONLY
configurations that survive near r = 1 are those with S close to 1 -- i.e. with the c_l almost
aligned in phase.  At S = 1 exactly, the triangle inequality is an equality, every c_l has the
same phase, and after a global rotation x_l = c_l >= 0 with sum_l x_l = 1:

    the rule z is then a genuine PROBABILITY measure on T^n whose every non-constant
    multi-affine moment vanishes -- a probability cubature rule.

THE DUAL BOUND (Proposition A1).  For a probability cubature rule, apply the exact pairing
sum_l x_l Lambda(v_l) = lambda_empty to the kernel-based test polynomial

    Lambda(z) = (1 + t) - t * prod_{j in J} (1 + z_j conj(v_{0 j})),     |J| = d,  t > 0,

which has lambda_empty = 1.  Taking real parts, with  m_d := min over T^d of Re prod_j (1 + w_j),

    1 = sum_l x_l Re Lambda(v_l) <= x_0 [1 - t(2^d - 1)] + (1 - x_0)[1 + t(1 - m_d)] ,

whence, for every atom and every d <= n,

    max_l x_l  <=  A_d := (1 - m_d) / (2^d - m_d) .

Since r - 1 < 0, a bound x_l <= kappa for all l gives sum_l x_l^r >= (sum_l x_l) kappa^{r-1}, so
max_l x_l <= 1/(n+1) would give P1 for EVERY r at once in the aligned case.  A_1 = 1/2 and A_2 = 1/3 are exactly 1/(n+1) -- so the aligned case of P1 is PROVED
at n = 1 and n = 2, for every r in (0,1].  A_3 = 3/10 > 1/4, so n >= 3 needs a better certificate;
block Bopt searches for one over all normalised multi-affine Lambda.

Blocks:
  M   the constants m_d and A_d (diagonal formula + global check)
  W   the true maximum weight of a probability cubature rule (LP over weights, search over nodes)
  C   the true minimum of sum_l x_l^r over probability cubature rules
  B   Proposition A1 checked against sampled probability rules
  O   the best possible test polynomial: inf over normalised multi-affine Lambda of
      (b - 1)/(b - a), b = max_{T^n} Re Lambda, a = Re Lambda(v_0)
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import linprog, minimize

import p1b_core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1B")


# ---------------------------------------------------------------- m_d and A_d

def ReK(a):
    """Re prod_j (1 + w_j) with w_j = exp(2 i a_j):  2^d prod_j cos(a_j) * cos(sum_j a_j)."""
    a = np.asarray(a)
    return (2.0 ** a.size) * np.prod(np.cos(a)) * np.cos(np.sum(a))


def m_d(d, starts=400, seed=0):
    rng = np.random.default_rng(seed)
    best = np.inf
    for i in range(starts):
        a0 = rng.uniform(-np.pi / 2, np.pi / 2, d) if i else np.full(d, np.pi / 4)
        res = minimize(ReK, a0, method="Nelder-Mead",
                       options=dict(maxiter=8000, xatol=1e-13, fatol=1e-15))
        best = min(best, float(res.fun))
    # diagonal closed form for comparison
    f = lambda a: (2.0 ** d) * np.cos(a) ** d * np.cos(d * a)
    diag = min(float(minimize(f, np.array([x]), method="Nelder-Mead",
                              options=dict(xatol=1e-14, fatol=1e-16)).fun)
               for x in np.linspace(0.05, np.pi / 2 - 0.05, 40))
    return best, diag


# ---------------------------------------------------------------- probability cubature rules

def moment_rows(v):
    """Real linear system for the vanishing non-constant multi-affine moments, plus sum x = 1."""
    M = C.moment_matrix(v)              # (2^n, N)
    rows = [np.ones(M.shape[1])]
    rhs = [1.0]
    for T in range(1, M.shape[0]):
        rows.append(M[T].real); rhs.append(0.0)
        rows.append(M[T].imag); rhs.append(0.0)
    return np.array(rows), np.array(rhs)


def max_weight(v):
    """max x_0 over x >= 0 with sum x = 1 and all non-constant moments zero (an LP)."""
    Aeq, beq = moment_rows(v)
    c = np.zeros(v.shape[0]); c[0] = -1.0
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * v.shape[0], method="highs")
    if not res.success:
        return None, None
    return -res.fun, res.x


def min_cost_rule(v, r):
    """min sum x^r over the same polytope: concave, so a vertex; SLSQP from many starts is used
    together with the LP vertices as starts."""
    Aeq, beq = moment_rows(v)
    N = v.shape[0]
    best, bx = np.inf, None
    rng = np.random.default_rng(0)
    starts = []
    for k in range(N):
        c = np.zeros(N); c[k] = -1.0
        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * N, method="highs")
        if res.success:
            starts.append(res.x)
    if not starts:
        return None, None
    for x0 in starts + [rng.dirichlet(np.ones(N)) for _ in range(12)]:
        res = minimize(lambda x: float(np.sum(np.maximum(x, 0) ** r)), x0, method="SLSQP",
                       bounds=[(0, None)] * N,
                       constraints=[dict(type="eq", fun=lambda x: Aeq @ x - beq)],
                       options=dict(maxiter=400, ftol=1e-14))
        if res.success and np.max(np.abs(Aeq @ res.x - beq)) < 1e-9 and np.all(res.x > -1e-9):
            if res.fun < best:
                best, bx = float(res.fun), res.x
    return best, bx


def search_rules(n, N, r, trials=400, seed=0, mode="weight"):
    """Search over node sets for the extreme probability cubature rule."""
    rng = np.random.default_rng(seed)
    best = -np.inf if mode == "weight" else np.inf
    bestv = None
    def obj(theta):
        v = np.exp(1j * theta.reshape(N, n))
        if mode == "weight":
            w, _ = max_weight(v)
            return -w if w is not None else 1e3
        val, _ = min_cost_rule(v, r)
        return val if val is not None else 1e3
    for i in range(trials):
        if i == 0:
            vb = np.vstack([C.block(n), np.exp(1j * rng.uniform(0, 2 * np.pi, (N - n - 1, n)))])
            th0 = np.angle(vb).ravel()
        else:
            th0 = rng.uniform(0, 2 * np.pi, N * n)
        res = minimize(obj, th0, method="Nelder-Mead",
                       options=dict(maxiter=1500, maxfev=1500, xatol=1e-9, fatol=1e-12))
        val = -res.fun if mode == "weight" else res.fun
        if (mode == "weight" and val > best) or (mode != "weight" and val < best):
            best, bestv = val, res.x.copy()
    return best, bestv


def check_propA1(n, N, trials=600, seed=0):
    """Proposition A1 on sampled probability cubature rules."""
    rng = np.random.default_rng(seed)
    mds = {d: m_d(d, starts=60, seed=1)[0] for d in range(1, n + 1)}
    Ad = min((1 - mds[d]) / (2 ** d - mds[d]) for d in range(1, n + 1))
    worst = -np.inf
    used = 0
    for _ in range(trials):
        v = np.vstack([C.block(n) * np.exp(1j * rng.normal(0, rng.choice([0.0, 0.05, 0.3, 1.5]),
                                                           (n + 1, n))),
                       np.exp(1j * rng.uniform(0, 2 * np.pi, (N - n - 1, n)))])
        w, x = max_weight(v)
        if w is None:
            continue
        used += 1
        worst = max(worst, w)
    return dict(n=n, N=N, used=used, A_d=float(Ad), one_over_np1=1.0 / (n + 1),
                max_weight_seen=float(worst), A1_holds=bool(worst <= Ad + 1e-9))


# ---------------------------------------------------------------- best test polynomial

def best_certificate(n, starts=200, seed=0, grid=48):
    """inf over normalised multi-affine Lambda of (b-1)/(b-a), b = max Re Lambda on T^n,
    a = Re Lambda at v_0 = (1,..,1).  This is the best bound on max_l x_l that ANY single
    test polynomial can give for a probability cubature rule."""
    rng = np.random.default_rng(seed)
    pts = np.stack(np.meshgrid(*[np.linspace(0, 2 * np.pi, grid, endpoint=False)] * n,
                               indexing="ij"), axis=-1).reshape(-1, n)
    Z = np.exp(1j * pts)
    masks = []
    for T in range(1 << n):
        m = np.ones(Z.shape[0], dtype=complex)
        for j in range(n):
            if T >> j & 1:
                m = m * Z[:, j]
        masks.append(m)
    Mgrid = np.stack(masks, axis=1)          # (npts, 2^n)
    ones = np.ones(1 << n, dtype=complex)    # value at v_0 = (1,...,1)

    def f(p):
        lam = np.concatenate([[1.0 + 0j], p[:(1 << n) - 1] + 1j * p[(1 << n) - 1:]])
        vals = (Mgrid @ lam).real
        b = float(vals.max())
        a = float((ones @ lam).real)
        if b - a < 1e-12:
            return 1e3
        return (b - 1.0) / (b - a)

    best = np.inf
    for i in range(starts):
        p0 = rng.normal(0, 1.0, 2 * ((1 << n) - 1))
        res = minimize(f, p0, method="Nelder-Mead",
                       options=dict(maxiter=20000, maxfev=20000, xatol=1e-11, fatol=1e-13))
        best = min(best, float(res.fun))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="MBA")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--tag", default="align")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}

    if "M" in args.blocks:
        rows = []
        for d in range(1, 7):
            g, dg = m_d(d, starts=200)
            rows.append(dict(d=d, m_d=g, m_d_diagonal=dg, A_d=(1 - g) / (2 ** d - g),
                             one_over_dplus1=1.0 / (d + 1)))
            print("M  d=%d  m_d = %.9f (diag %.9f)  A_d = %.9f   1/(d+1) = %.9f  %s"
                  % (d, g, dg, rows[-1]["A_d"], rows[-1]["one_over_dplus1"],
                     "SHARP" if abs(rows[-1]["A_d"] - 1.0 / (d + 1)) < 1e-6 else ""))
        out["M"] = rows

    if "B" in args.blocks:
        out["B"] = [check_propA1(n, 2 ** (n + 1), trials=args.trials) for n in (2, 3)]
        for row in out["B"]:
            print("B  n=%d  A_d = %.9f  1/(n+1) = %.6f  max weight seen %.9f  Prop A1 %s"
                  % (row["n"], row["A_d"], row["one_over_np1"], row["max_weight_seen"],
                     "holds" if row["A1_holds"] else "VIOLATED"))

    if "W" in args.blocks:
        out["W"] = []
        for n in (2, 3):
            w, _ = search_rules(n, 2 ** (n + 1), 0.5, trials=max(args.trials // 10, 20), mode="weight")
            out["W"].append(dict(n=n, max_weight=float(w), one_over_np1=1.0 / (n + 1)))
            print("W  n=%d  max weight over probability cubature rules = %.9f   (1/(n+1) = %.6f)"
                  % (n, w, 1.0 / (n + 1)))

    if "A" in args.blocks:
        out["A"] = check_A2(trials=max(args.trials * 40, 4000))
        print("A  Prop A2 (n=2): %d configs, %d violations, min slack %.9f (delta=%.4g, max|c|=%.6f, N=%d)"
              % (out["A"]["configs"], out["A"]["violations"], out["A"]["min_slack"],
                 out["A"]["worst_at"]["delta"], out["A"]["worst_at"]["max_c"], out["A"]["worst_at"]["N"]))
        print("   torus constants: min Re K = %.9f (-1/2), max|Im K| = %.9f (3sqrt3/2 = %.9f)"
              % (out["A"]["torus_constants"]["min_ReK"], out["A"]["torus_constants"]["max_abs_ImK"],
                 out["A"]["torus_constants"]["three_sqrt3_over_2"]))

    if "O" in args.blocks:
        out["O"] = []
        for n in (1, 2, 3):
            b = best_certificate(n, starts=60 if n < 3 else 120, grid=40 if n < 3 else 24)
            out["O"].append(dict(n=n, best_single_Lambda_bound=b, one_over_np1=1.0 / (n + 1)))
            print("O  n=%d  best single-Lambda bound on max weight = %.9f  (1/(n+1) = %.6f)"
                  % (n, b, 1.0 / (n + 1)))

    with open(os.path.join(OUT, args.tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- Proposition A2 (n = 2)

def A2_bound(delta):
    """max_l |c_l| <= 1/3 + [8.5 d + (3 sqrt3/2) sqrt(2(1+d)d)] / 4.5,  d = sum_l|c_l| - |A|."""
    return 1.0 / 3.0 + (8.5 * delta + (3 * np.sqrt(3) / 2) * np.sqrt(2 * (1 + delta) * delta)) / 4.5


def check_A2(trials=25000, seed=3):
    """Proposition A2 on (KL) systems at n = 2, N = 4..7, and the two torus constants it uses."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi / 2, np.pi / 2, (400000, 2))
    K = (1 + np.exp(2j * a[:, 0])) * (1 + np.exp(2j * a[:, 1]))
    consts = dict(max_abs_ImK=float(np.abs(K.imag).max()), three_sqrt3_over_2=float(3 * np.sqrt(3) / 2),
                  min_ReK=float(K.real.min()))
    worst, worst_at, cnt, viol = np.inf, None, 0, 0
    for N in (4, 5, 6, 7):
        for _ in range(trials):
            s = rng.choice([0.01, 0.05, 0.2, 0.6, 2.0])
            v = np.vstack([C.block(2) * np.exp(1j * rng.normal(0, s, (3, 2))),
                           np.exp(1j * rng.uniform(0, 2 * np.pi, (N - 3, 2)))])
            M = C.moment_matrix(v); rhs = np.zeros(4, dtype=complex); rhs[0] = 1.0
            if N == 4:
                try:
                    z = np.linalg.solve(M, rhs)
                except np.linalg.LinAlgError:
                    continue
            else:
                z, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                if np.linalg.norm(M @ z - rhs) > 1e-9:
                    continue
                Kn = C.nullspace(v)
                if Kn.shape[1] > 0:
                    t = rng.normal(size=Kn.shape[1]) + 1j * rng.normal(size=Kn.shape[1])
                    z = z + rng.choice([1e-4, 1e-3, 1e-2, 0.1, 0.5]) * Kn @ t / max(1.0, np.linalg.norm(t))
            if not np.all(np.isfinite(z)):
                continue
            x = np.abs(z); S = float(x.sum())
            if S > 50:
                continue
            cnt += 1
            sl = A2_bound(S - 1.0) - float(x.max())
            if sl < 0:
                viol += 1
            if sl < worst:
                worst, worst_at = sl, (S - 1.0, float(x.max()), N)
    return dict(configs=cnt, violations=viol, min_slack=worst,
                worst_at=dict(delta=worst_at[0], max_c=worst_at[1], N=worst_at[2]), torus_constants=consts)
