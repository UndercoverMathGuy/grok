"""p3b_routes.py -- the two OTHER routes to P3 that the dual-measure barrier leaves open, measured.

ROUTE A (blocks A1-A3): symmetrisation by the TASK's own translation group.
  The diagonal translation  T_tau:  a_j -> a_j + tau (all j),  c -> c + n tau  is a genuine symmetry
  of the full-table task.  On a network it acts by e_ij(.) -> e_ij(. - tau), w_i(.) -> w_i(. - n tau).
  Lemma A (checked here to fp64):  margin, P_q and EVERY atom c_ik are invariant, while
        J_{k,T}  ->  omega^{2 k tau |T|} J_{k,T},     omega = exp(2 pi i / p),
  so the whole same-|k| junk is rotated by a character and the clean part is untouched.
  Consequence: the weighted direct sum  eta = (+)_tau s_tau . T_tau theta  has
        G_k(eta) = (sum_tau g_tau) G_k(theta),   g_tau := s_tau^nu,
        J_{k,T}(eta) = ghat(-2k|T|) J_{k,T}(theta),
        P_q(eta) = (sum_tau g_tau^r) P_q(theta),   r = q/nu,
        margin(eta) >= (sum_tau g_tau) margin(theta)   (min of a sum >= sum of mins).
  Normalising sum_tau g_tau = 1, the whole construction costs the factor  C(g) := sum_tau g_tau^r
  and buys (H) as soon as ghat vanishes on  U := { +-2k|T| mod p : k in H, 1 <= |T| <= n }.
  A2 computes  C_min := min { sum_tau g_tau^r : g >= 0, sum g = 1, ghat|_U = 0 }  and compares it
  with the 1 that (T) would need -- this is the inherited "symmetrisation is circular" statement in
  its exact quantitative form.

ROUTE B (block B1): the SUCCESSOR route -- per-frequency torus minima at a common row.
  Agent 2's Theorem B prices junk EXACTLY at one frequency: sum_l |c_l|^r >= (n+1)^{1-r} R^r with
  R = min over the torus T^n of Re h.  Extending it to K frequencies needs
        min over rows a of sum_k w_kD Re h^(k)(u^(k)(a))   ~=   sum_k w_kD R_k ,
  i.e. one row must nearly realise EVERY frequency's own torus minimum at once.  The gap between
  the two sides (always >= 0) is the "simultaneity defect"; B1 measures it.
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gopt_full as GF  # noqa: E402
import p3b_max as PM  # noqa: E402


# ============================================================================= ROUTE A
def translate(E, W, p, n, tau):
    """T_tau: e_ij(.) -> e_ij(. - tau)  (all j),  w_i(.) -> w_i(. - n tau)."""
    return np.roll(E, tau, axis=2), np.roll(W, (n * tau) % p, axis=0)


def route_A1(p, n, q, m, rng, trials=6):
    """Lemma A to fp64 on random networks."""
    idx, y = GF.rows_and_wrong(p, n)
    om = np.exp(2j * np.pi / p)
    worst = dict(dmargin=0.0, dPq=0.0, dG=0.0, djunk=0.0)
    for _ in range(trials):
        E = rng.normal(size=(n, m, p)); W = rng.normal(size=(p, m))
        mg0, _ = PM.margin_of(E, W, idx, y, n)
        P0 = (((E ** 2).sum((0, 2)) + (W ** 2).sum(0)) ** (q / 2)).sum()
        _, _, G0, J0 = PM.atoms(E, W, p, n)
        for tau in range(1, p):
            E2, W2 = translate(E, W, p, n, tau)
            mg1, _ = PM.margin_of(E2, W2, idx, y, n)
            P1 = (((E2 ** 2).sum((0, 2)) + (W2 ** 2).sum(0)) ** (q / 2)).sum()
            _, _, G1, J1 = PM.atoms(E2, W2, p, n)
            worst["dmargin"] = max(worst["dmargin"], abs(mg1 - mg0) / max(abs(mg0), 1e-300))
            worst["dPq"] = max(worst["dPq"], abs(P1 - P0) / P0)
            worst["dG"] = max(worst["dG"], np.abs(G1 - G0).max() / np.abs(G0).max())
            ks = np.arange(1, (p - 1) // 2 + 1)
            for T, j0 in J0.items():
                pred = om ** (2 * ks * tau * len(T)) * j0
                worst["djunk"] = max(worst["djunk"],
                                     float(np.abs(J1[T] - pred).max() / max(np.abs(j0).max(), 1e-300)))
    return worst


def U_set(p, n):
    """U = { +- 2 k |T| mod p : k = 1..(p-1)/2, |T| = 1..n } -- the characters the junk is rotated by."""
    U = set()
    for k in range(1, (p - 1) // 2 + 1):
        for t in range(1, n + 1):
            U.add((2 * k * t) % p); U.add((-2 * k * t) % p)
    U.discard(0)
    return sorted(U)


def route_A2(p, n, q, starts=200, seed=0):
    """C_min = min { sum_tau g^r : g >= 0, sum g = 1, ghat(u) = 0 for u in U }."""
    nu, r = n + 1, q / (n + 1)
    U = U_set(p, n)
    rng = np.random.default_rng(seed)
    # real linear constraints: Re ghat(u) = Im ghat(u) = 0 for u in U (only u and -u paired once)
    rows = []
    seen = set()
    for u in U:
        if u in seen:
            continue
        seen.add(u); seen.add((-u) % p)
        t = np.arange(p)
        rows.append(np.cos(2 * np.pi * u * t / p))
        rows.append(np.sin(2 * np.pi * u * t / p))
    A = np.array(rows) if rows else np.zeros((0, p))
    cons = [{"type": "eq", "fun": lambda g: A @ g, "jac": lambda g: A},
            {"type": "eq", "fun": lambda g: np.array([g.sum() - 1.0]),
             "jac": lambda g: np.ones((1, p))}]
    best = np.inf
    for s in range(starts):
        g0 = rng.dirichlet(np.full(p, rng.uniform(0.2, 3.0)))
        res = minimize(lambda g: (np.maximum(g, 0) ** r).sum(),
                       g0, jac=lambda g: r * np.maximum(g, 1e-14) ** (r - 1),
                       method="SLSQP", bounds=[(0, 1)] * p, constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-14})
        g = np.maximum(res.x, 0)
        if np.abs(A @ g).max() < 1e-9 and abs(g.sum() - 1) < 1e-9:
            best = min(best, float((g ** r).sum()))
    # the uniform solution is always feasible
    unif = float(p * (1.0 / p) ** r)
    return dict(p=p, n=n, q=q, r=r, nU=len(U), n_free_fourier=p - len(U),
                C_min=best, C_uniform=unif, needed_for_T=1.0,
                block_factor=(n + 1) ** (1 - r))


# ============================================================================= ROUTE B
def h_of(c, v, u):
    """h(u) = sum_l c_l prod_j (1 + u_j v_lj);  c (N,), v (N,n), u (n,) complex."""
    return np.sum(c * np.prod(1.0 + u[None, :] * v, axis=1))


def torus_min_Re(c, v, n, ngrid=180, polish=True):
    """R = min over the n-torus of Re h."""
    ax = np.linspace(0, 2 * np.pi, ngrid, endpoint=False)
    grids = np.meshgrid(*[ax] * n, indexing="ij")
    U = [np.exp(1j * g) for g in grids]
    H = np.zeros(grids[0].shape, complex)
    for l in range(len(c)):
        term = np.full(grids[0].shape, c[l], dtype=complex)
        for j in range(n):
            term = term * (1.0 + U[j] * v[l, j])
        H = H + term
    R = float(np.real(H).min())
    if polish:
        i = np.unravel_index(np.argmin(np.real(H)), H.shape)
        x0 = np.array([ax[i[j]] for j in range(n)])
        res = minimize(lambda x: float(np.real(h_of(c, v, np.exp(1j * x)))), x0,
                       method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 4000})
        R = min(R, float(res.fun))
    return R


def route_B1(p, n, K, N, rng, trials, junk_scales):
    """the simultaneity defect:  min_a sum_k w_kD Re h^(k)(u^(k)(a))   vs   sum_k w_kD R_k."""
    nf = (p - 1) // 2
    S = rng.choice(np.arange(1, nf + 1), size=K, replace=False) if K < nf else np.arange(1, nf + 1)[:K]
    rows = np.stack(np.meshgrid(*[np.arange(p)] * n, indexing="ij"), -1).reshape(-1, n)
    out = []
    for js in junk_scales:
        worst = -np.inf; recs = []
        for _ in range(trials):
            cs, vs, Rs = [], [], []
            for k in S:
                # a block of N atoms + a random junk-creating perturbation of size js
                ph = np.exp(-2j * np.pi * np.arange(N) / (n + 1))
                v = np.tile(ph[:, None], (1, n)) * np.exp(1j * js * rng.normal(size=(N, n)))
                c = (np.ones(N) / N) * np.exp(1j * js * rng.normal(size=N))
                cs.append(c); vs.append(v); Rs.append(torus_min_Re(c, v, n))
            # evaluate every frequency at every row
            Hre = np.zeros((len(rows), K))
            for a_i, k in enumerate(S):
                u = np.exp(-4j * np.pi * k * rows / p)          # (R, n)
                tot = np.zeros(len(rows), complex)
                for l in range(N):
                    term = np.full(len(rows), cs[a_i][l], dtype=complex)
                    for j in range(n):
                        term = term * (1.0 + u[:, j] * vs[a_i][l, j])
                    tot = tot + term
                Hre[:, a_i] = np.real(tot)
            Gk = np.array([abs(cs[i].sum()) for i in range(K)])
            L, Rr, sc = np.inf, np.inf, np.inf
            for D in range(1, p):
                w = 1 - np.cos(2 * np.pi * S * D / p)
                L = min(L, float((Hre * w[None, :]).sum(1).min()))
                Rr = min(Rr, float((np.array(Rs) * w).sum()))
                sc = min(sc, float((Gk * w).sum()))
            # defect = [min_D LHS] - [min_D RHS], in units of the junk-free clean value min_D sum_k w|G_k|
            best_ratio = (L - Rr) / max(sc, 1e-30)
            recs.append(dict(lhs_over_rhs_minD=best_ratio))
            worst = max(worst, best_ratio)
        out.append(dict(junk_scale=js, K=K, N=N, support=S.tolist(),
                        worst_lhs_over_rhs=float(worst),
                        median_lhs_over_rhs=float(np.median([r["lhs_over_rhs_minD"] for r in recs]))))
    return out


# ============================================================================= driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="A1A2B1")
    ap.add_argument("--ps", type=int, nargs="+", default=[5, 7, 11, 13])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    n, nu = a.n, a.n + 1
    rng = np.random.default_rng(a.seed)
    out = {}
    if "A1" in a.block:
        print("[A1] Lemma A (diagonal translation) on random networks, worst relative deviations:")
        A1 = []
        for p in a.ps[:2]:
            w = route_A1(p, n, a.q, nu + 2, rng, trials=a.trials)
            A1.append(dict(p=p, **w))
            print(f"  p={p}: d(margin)={w['dmargin']:.2e}  d(P_q)={w['dPq']:.2e}  "
                  f"d(G_k)={w['dG']:.2e}  junk-rotation-law residual={w['djunk']:.2e}")
        out["A1"] = A1
    if "A2" in a.block:
        print("\n[A2] the cost of killing the junk by translation averaging "
              "(C(g) = sum_tau g^r; (T) needs C <= 1, and C >= 1 always with equality only at a point mass):")
        A2 = []
        for p in a.ps:
            d = route_A2(p, n, a.q)
            A2.append(d)
            print(f"  p={p:3d}: |U|={d['nU']:3d} of {p-1} non-zero characters; free Fourier slots "
                  f"{d['n_free_fourier']};  C_min={d['C_min']:.6f}  (uniform g: {d['C_uniform']:.6f}; "
                  f"block factor (n+1)^(1-r)={d['block_factor']:.6f})")
        out["A2"] = A2
    if "B1" in a.block:
        print("\n[B1] simultaneity defect:  min_a sum_k w_kD Re h^(k)  /  sum_k w_kD R_k   "
              "(=1 would make the K-frequency junk bound exact; >1 is the defect):")
        B1 = []
        for p in a.ps:
            if p < 11:
                continue
            for K in (1, 2, 3):
                if K > (p - 1) // 2:
                    continue
                res = route_B1(p, n, K, n + 1, rng, a.trials, [0.0, 0.1, 0.3, 0.6, 1.0])
                for rr in res:
                    B1.append(dict(p=p, **rr))
                    print(f"  p={p:3d} K={K} junk={rr['junk_scale']:.2f}: "
                          f"median {rr['median_lhs_over_rhs']:.4f}   worst {rr['worst_lhs_over_rhs']:.4f}")
        out["B1"] = B1
    d = ROOT / "FINAL_RUNS" / "P3B"; d.mkdir(parents=True, exist_ok=True)
    (d / f"routes{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote", d / f"routes{a.tag}.json")


if __name__ == "__main__":
    main()
