"""p3b_dual.py -- P3 by the DUAL MEASURE (min-max / KKT) route, and the barrier that kills it.

Setting (gated chain, full n-operand modular-addition table mod p):

    z(a_1..a_n, c) = sum_i w_i(c) prod_j e_ij(a_j),      nu = n+1,
    Phi(a, c) := z(a, s(a)) - z(a, c)   for c != s(a) = sum_j a_j mod p,
    margin(theta) = min over (a, c) of Phi(a, c),
    P_q(theta) = sum_i ||theta_i||^q,   gamma_q = margin / P_q^{nu/q}.

The max-margin problem is a min-max:  gamma_q^* = max_{P_q <= 1} min_{mu in Delta} E_mu[Phi],
where mu is a probability measure on the (row, wrong class) pairs.  WEAK DUALITY gives, for EVERY mu,

    gamma_q^*  <=  V_q(mu) := max { E_mu[Phi(theta)] : P_q(theta) <= 1 }.

This script establishes, numerically, the two facts proved in P3B.md:

  (B1)  V_q(mu) = nu^{-nu/2} * sigma(T_mu)   for every q in [2, nu]  -- INDEPENDENT OF q.
        sigma = injective (spectral) norm of the (n+1)-way tensor
            T_mu[a_1..a_n, c] = m(a) [c = s(a)] - mu(a, c),    m(a) = sum_c mu(a,c),
        and the maximiser is a SINGLE neuron.
  (B2)  inf_mu V_q(mu) >= gamma_nu^*  (the q = nu / Morwani max-margin value) >= B Gamma_1(p)
        >= B Gamma_{nu/q}(p) = (n+1)^{(nu-q)/q} * target.
        So the dual-measure route can never prove (T) for q < nu.

sigma(T_mu) is a max of linear functionals of mu, hence CONVEX in mu, so inf_mu V(mu) is a convex
programme and projected subgradient descent finds its global value.  fp64, local CPU.
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gopt_full import gamma_star, rows_and_wrong, build_block  # noqa: E402
from kstar import ladder  # noqa: E402


# ----------------------------------------------------------------------------- tensor spectral norm
def sigma(T, n, rng, restarts=40, iters=400, tol=1e-14):
    """injective norm max{ T[x_1,..,x_n,y] : ||x_j|| = ||y|| = 1 } by alternating maximisation."""
    p = T.shape[0]
    best, bestvecs = -np.inf, None
    for r in range(restarts):
        V = [rng.normal(size=p) for _ in range(n + 1)]
        V = [v / np.linalg.norm(v) for v in V]
        val = -np.inf
        for _ in range(iters):
            prev = val
            for slot in range(n + 1):
                # contract every axis but `slot`
                C = T
                axes = [ax for ax in range(n + 1) if ax != slot]
                for ax in sorted(axes, reverse=True):
                    C = np.tensordot(C, V[ax], axes=([ax], [0]))
                nv = np.linalg.norm(C)
                if nv < 1e-300:
                    break
                V[slot] = C / nv
                val = nv
            if val - prev < tol * max(1.0, abs(val)):
                break
        if val > best:
            best, bestvecs = val, [v.copy() for v in V]
    return best, bestvecs


def tensor_of(mu, p, n, idx, y):
    """T[a_1..a_n, c] = m(a) [c = s(a)] - mu(a, c).   mu given as (R, p) with mu[:, s(a)] = 0."""
    R = idx.shape[0]
    T = -mu.copy()
    T[np.arange(R), y] += mu.sum(1)
    return T.reshape((p,) * n + (p,))


# ----------------------------------------------------------------------------- primal check of (B1)
def primal_V(mu, p, n, q, m, idx, y, rng, restarts=8, seed=None):
    """max { E_mu[Phi(theta)] : P_q(theta) <= 1 } over raw (E, W) of width m.

    Both E_mu[Phi] (degree nu) and P_q^{nu/q} (degree nu) are homogeneous, so the constrained max
    equals the unconstrained max of the SCALE-INVARIANT ratio; that is what is optimised (L-BFGS-B,
    analytic gradient), which needs no constraint handling and cannot run away.
    """
    nu = n + 1
    R = idx.shape[0]
    ar = np.arange(R)

    def unpack(x):
        nE = n * m * p
        return x[:nE].reshape(n, m, p), x[nE:].reshape(p, m)

    def negratio(x):
        E, W = unpack(x)
        H = np.ones((R, m))
        for j in range(n):
            H = H * E[j][:, idx[:, j]].T
        z = H @ W.T
        num = float((mu * (z[ar, y][:, None] - z)).sum())
        A = -mu.copy(); A[ar, y] += mu.sum(1)             # d num / d z[a,c]
        gnW = H.T @ A                                     # (m, p)
        gnE = np.zeros((n, m, p))
        AW = A @ W                                        # (R, m)
        for j in range(n):
            Q = np.ones((R, m))
            for j2 in range(n):
                if j2 != j:
                    Q = Q * E[j2][:, idx[:, j2]].T
            coef = Q * AW
            for b in range(p):
                gnE[j, :, b] = coef[idx[:, j] == b].sum(0)
        nrm2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(0)
        P = float((nrm2 ** (q / 2)).sum())
        if P < 1e-300:
            return 0.0, np.zeros_like(x)
        d = (q / 2) * nrm2 ** (q / 2 - 1)
        gpE = 2 * E * d[None, :, None]; gpW = 2 * W * d[None, :]
        Pn = P ** (nu / q)
        val = num / Pn
        gE = gnE / Pn - val * (nu / q) * gpE / P
        gW = gnW.T / Pn - val * (nu / q) * gpW / P
        return -val, -np.concatenate([gE.reshape(-1), gW.reshape(-1)])

    starts = []
    if seed is not None:                                   # the single neuron built from sigma's argmax
        E0 = np.zeros((n, m, p)); W0 = np.zeros((p, m))
        for j in range(n):
            E0[j, 0] = seed[j]
        W0[:, 0] = seed[n]
        starts.append((E0, W0))
        for nz in (0.02, 0.2):
            starts.append((E0 + nz * rng.normal(size=E0.shape), W0 + nz * rng.normal(size=W0.shape)))
    for _ in range(restarts):
        starts.append((rng.normal(size=(n, m, p)), rng.normal(size=(p, m))))
    best = -np.inf
    for E0, W0 in starts:
        x0 = np.concatenate([E0.reshape(-1), W0.reshape(-1)])
        res = minimize(negratio, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 4000, "ftol": 1e-16, "gtol": 1e-14})
        best = max(best, -res.fun)
    return best


# ----------------------------------------------------------------------------- min over mu (convex)
def min_over_mu(p, n, idx, y, rng, steps=4000, restarts=25, seed_uniform=True):
    """projected subgradient descent on the simplex for the convex function mu -> sigma(T_mu)."""
    R, nu = idx.shape[0], n + 1
    mask = np.ones((R, p)); mask[np.arange(R), y] = 0.0
    if seed_uniform:
        mu = mask / mask.sum()
    else:
        mu = rng.random((R, p)) * mask; mu /= mu.sum()
    best = np.inf; bestmu = mu.copy()
    for t in range(steps):
        T = tensor_of(mu, p, n, idx, y)
        s, V = sigma(T, n, rng, restarts=restarts if t % 50 == 0 else 4, iters=200)
        if s < best:
            best, bestmu = s, mu.copy()
        # subgradient: d sigma / d mu(a,c) = prod_j x_j(a_j) * ( y_vec(s(a)) - y_vec(c) )
        Hx = np.ones(R)
        for j in range(n):
            Hx = Hx * V[j][idx[:, j]]
        g = Hx[:, None] * (V[n][y][:, None] - V[n][None, :])
        g = g * mask
        step = 0.5 / np.sqrt(t + 1) * (abs(s) / max(np.linalg.norm(g), 1e-12))
        mu = mu - step * g
        # project onto the simplex restricted to the mask
        mu = project_simplex(mu, mask)
    return best, bestmu


def project_simplex(v, mask):
    x = np.where(mask > 0, v, -np.inf).reshape(-1)
    fin = np.isfinite(x)
    u = np.sort(x[fin])[::-1]
    css = np.cumsum(u) - 1
    ind = np.arange(1, len(u) + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    th = css[cond][-1] / rho
    out = np.zeros_like(x); out[fin] = np.maximum(x[fin] - th, 0)
    return out.reshape(v.shape)


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0, 2.5, 3.0])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    p, n, nu = a.p, a.n, a.n + 1
    rng = np.random.default_rng(a.seed)
    idx, y = rows_and_wrong(p, n)
    R = idx.shape[0]
    mask = np.ones((R, p)); mask[np.arange(R), y] = 0.0
    Bc = 2 * (2 * p * nu) ** (-nu / 2)
    G1 = gamma_star(p, 1.0, a.kmax)                       # Gamma_1 = the q = nu (flat) programme
    out = dict(p=p, n=n, B=Bc, Gamma_1=G1, cells=[])
    print(f"p={p} n={n} nu={nu}   B={Bc:.6e}   Gamma_1={G1:.6f}   B*Gamma_1={Bc*G1:.6e}")

    # --- (B1) q-independence of V(mu), on a few measures
    print("\n[B1] V_q(mu) = nu^{-nu/2} sigma(T_mu), independent of q:")
    mus = {"uniform": mask / mask.sum()}
    r1 = rng.random((R, p)) * mask; mus["random"] = r1 / r1.sum()
    r2 = (rng.random((R, p)) ** 4) * mask; mus["spiky"] = r2 / r2.sum()
    b1 = []
    for name, mu in mus.items():
        T = tensor_of(mu, p, n, idx, y)
        s, vecs = sigma(T, n, rng, restarts=80)
        V = nu ** (-nu / 2) * s
        row = dict(mu=name, sigma=s, V_tensor=V, primal={})
        for q in a.qs:
            if q > nu:
                continue
            vq = primal_V(mu, p, n, q, nu + 2, idx, y, rng, restarts=6, seed=vecs)
            row["primal"][f"q={q:g}"] = vq
            print(f"  mu={name:8s} q={q:g}: primal max E_mu[Phi] = {vq:.10e}   "
                  f"tensor bound = {V:.10e}   ratio = {vq/V:.8f}")
        b1.append(row)
    out["B1"] = b1

    # --- (B2) the barrier: inf_mu V(mu) vs B*Gamma_1 and vs the targets
    print("\n[B2] the barrier  inf_mu V(mu) >= gamma_nu^* = B Gamma_1:")
    best, bestmu = min_over_mu(p, n, idx, y, rng, steps=a.steps, restarts=25)
    Vmin = nu ** (-nu / 2) * best
    print(f"  min over mu of V(mu) = {Vmin:.10e}    B*Gamma_1 = {Bc*G1:.10e}"
          f"    ratio = {Vmin/(Bc*G1):.8f}")
    out["Vmin"] = Vmin
    for q in a.qs:
        if q > nu:
            continue
        Gs = gamma_star(p, nu / q, a.kmax)
        tgt = nu ** (-(nu - q) / q) * Bc * Gs
        print(f"  q={q:g}: Gamma_{{nu/q}}={Gs:.6f}  unrestricted bound B*Gamma_s={Bc*Gs:.6e}  "
              f"target (T)={tgt:.6e}   ->  Vmin/target = {Vmin/tgt:.6f}   "
              f"(needed for (T): <= 1;  circularity factor (n+1)^{{(nu-q)/q}}="
              f"{nu**((nu-q)/q):.4f})")
        out["cells"].append(dict(q=q, Gamma_s=Gs, target=tgt, Vmin_over_target=Vmin / tgt,
                                 unrestricted=Bc * Gs, Vmin_over_unrestricted=Vmin / (Bc * Gs)))

    # --- (B2) sanity: the block's own value certifies the barrier for every mu
    print("\n[B2-check] for 200 random mu:  V(mu) / (B*Gamma_1)  (must be >= 1)")
    worst = np.inf
    for _ in range(200):
        z = rng.random((R, p)) ** rng.uniform(0.3, 6.0) * mask
        mu = z / z.sum()
        s, _ = sigma(tensor_of(mu, p, n, idx, y), n, rng, restarts=6, iters=150)
        worst = min(worst, nu ** (-nu / 2) * s / (Bc * G1))
    print(f"  min over the 200 random mu of V(mu)/(B*Gamma_1) = {worst:.8f}")
    out["random_mu_min_ratio"] = worst

    d = ROOT / "FINAL_RUNS" / "P3B"; d.mkdir(parents=True, exist_ok=True)
    (d / f"dual_p{p}n{n}{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote", d / f"dual_p{p}n{n}{a.tag}.json")


if __name__ == "__main__":
    main()
