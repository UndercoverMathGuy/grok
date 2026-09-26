"""gopt_verify.py -- fp64 verification of every step of the new proof (GLOBAL_OPT.md Sec. 3).

The new argument bounds  R = min over the torus of Re h(u),  h(u) = sum_l c_l prod_j (1 + u_j v_lj),
with NO junk-free assumption and NO constraint on c, v at all.  Steps checked, on random configurations:

  (A)  kill identity: setting u_j = -conj(v_lj) makes atom l's product vanish exactly.
  (B)  N <= n  =>  R <= 0            (inject atoms into slots and kill them all)
  (C)  bijection bound, N = n+1:  R <= |c_l0| * prod_j |v_{sigma(j) j} - v_{l0 j}|  for every l0 and
       every bijection sigma from slots onto the other n atoms.
  (D)  geometric mean over sigma, product over l0, Hadamard:
           R^{n+1} <= prod_l |c_l| * (prod_j V_j^2)^{1/n},   V_j = prod_{l<m} |v_lj - v_mj| <= (n+1)^{(n+1)/2}
       hence prod_l |c_l| >= (R/(n+1))^{n+1}   whenever R > 0.
  (E)  AM-GM:  sum_l |c_l|^r >= (n+1)^{1-r} R^r  for r in (0,1].
  (F)  the margin identity used to pass from the network to h:
           margin = 2^{-n} min_{u in (mu_p)^n, D != 0} Re[(1 - e^{2 pi i k D / p}) h(u)]
           = 2^{-n} min_u [ M_1 Re h(u) - sin(2 pi / p) |Im h(u)| ],
       checked against the brute-force margin of the actual p^n x p logit table of the network.
"""
import argparse, itertools, json, math
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(ROOT))
from gopt_atom import R_torus, h_on_points, block  # noqa: E402


def rand_cfg(N, n, g):
    c = torch.complex(torch.randn(N, generator=g), torch.randn(N, generator=g))
    v = torch.polar(torch.ones(N, n), torch.rand(N, n, generator=g) * 2 * math.pi)
    return c, v


def net_from_atoms(c, v, p, k=1):
    """Build the actual gated-chain network (balanced factors) whose atom data is (c, v).

    v_lj = e^{-2 i alpha_lj}  ->  alpha_lj = -arg(v_lj)/2 ;  psi_l = arg c_l = sum_j alpha_lj - beta_l.
    Amplitudes: |c_l| = (prod_j r_lj) rho_l ; balanced choice r_lj = rho_l = |c_l|^{1/(n+1)}.
    """
    n = v.shape[1]
    N = c.shape[0]
    a = torch.arange(p) * (2 * math.pi * k / p)
    alpha = -torch.angle(v) / 2                                   # (N, n)
    beta = alpha.sum(1) - torch.angle(c)                          # (N,)
    amp = c.abs().pow(1.0 / (n + 1))
    E = torch.stack([amp[:, None] * torch.cos(a[None, :] + alpha[:, j][:, None]) for j in range(n)])
    W = (amp[:, None] * torch.cos(a[None, :] + beta[:, None])).T  # (p, N)
    return E, W


def brute_margin(E, W, p, n):
    idx = torch.cartesian_prod(*[torch.arange(p)] * n) if n > 1 else torch.arange(p)[:, None]
    h = torch.ones(idx.shape[0], E.shape[1])
    for j in range(n):
        h = h * E[j][:, idx[:, j]].T
    z = h @ W.T                                                   # (p^n, p)
    y = idx.sum(1) % p
    zy = z.gather(1, y[:, None])
    z2 = z.scatter(1, y[:, None], -float("inf"))
    return float((zy.squeeze(1) - z2.max(1).values).min())


def main(a):
    g = torch.Generator().manual_seed(a.seed)
    out = {"A_kill": 0.0, "B_Nlen": [], "C_bij": [], "D_had": [], "E_key": [], "F_margin": []}
    for n in range(1, a.nmax + 1):
        # (A) kill identity
        for _ in range(20):
            c, v = rand_cfg(n + 2, n, g)
            l = int(torch.randint(n + 2, (1,), generator=g))
            j = int(torch.randint(n, (1,), generator=g))
            u = torch.polar(torch.ones(1, n), torch.rand(1, n, generator=g) * 2 * math.pi)
            u[0, j] = -v[l, j].conj()
            val = float(((1 + u[0] * v[l]).prod()).abs())
            out["A_kill"] = max(out["A_kill"], val)
        # (B) N <= n  => R <= 0
        for N in range(1, n + 1):
            worst = -1e18
            for _ in range(200):
                c, v = rand_cfg(N, n, g)
                worst = max(worst, float(R_torus(c, v)) / float(c.abs().sum()))
            out["B_Nlen"].append(dict(n=n, N=N, max_R_over_l1=worst))
        # (C)-(E) N = n+1, random (junk present)
        N = n + 1
        worstC, worstD, worstE, npos = -1e18, -1e18, -1e18, 0
        for tr in range(a.trials):
            # half fully random, half block + noise (random configs with N = n+1 rarely have R > 0)
            if tr % 2 == 0:
                c, v = rand_cfg(N, n, g)
            else:
                c0, v0 = block(n)
                nz = 10.0 ** float(torch.empty(1).uniform_(-3, -0.3, generator=g))
                c = c0 + nz * torch.complex(torch.randn(N, generator=g), torch.randn(N, generator=g))
                v = torch.polar(torch.ones(N, n), torch.angle(v0) + nz * torch.randn(N, n, generator=g))
            R = float(R_torus(c, v))
            if R <= 0:
                continue
            npos += 1
            # (C) all bijections
            for l0 in range(N):
                others = [l for l in range(N) if l != l0]
                for perm in itertools.permutations(others):
                    u = torch.stack([-v[perm[j], j].conj() for j in range(n)])
                    hv = float((c * (1 + u[None, :] * v).prod(1)).sum().real)
                    rhs = float(c[l0].abs() * torch.stack(
                        [(v[perm[j], j] - v[l0, j]).abs() for j in range(n)]).prod())
                    worstC = max(worstC, R / rhs)
                    assert abs(hv - float((c[l0] * (1 + u * v[l0]).prod()).real)) < 1e-9 * max(1, abs(hv))
            # (D) Hadamard + product
            prodc = float(c.abs().log().sum().exp())
            worstD = max(worstD, (R / (n + 1)) ** (n + 1) / prodc)
            Vmax = 0.0
            for j in range(n):
                V = 1.0
                for l in range(N):
                    for m in range(l + 1, N):
                        V *= float((v[l, j] - v[m, j]).abs())
                Vmax = max(Vmax, V / (n + 1) ** ((n + 1) / 2))
            # (E) key inequality for several r
            for r in (0.25, 0.4, 0.5, 2 / 3, 0.8, 1.0):
                lhs = float((c.abs() ** r).sum())
                worstE = max(worstE, (n + 1) ** (1 - r) * R ** r / lhs)
            out["D_had"].append(dict(n=n, max_V_over_hadamard=Vmax))
        Vh = max([d["max_V_over_hadamard"] for d in out["D_had"]], default=float("nan"))
        out["C_bij"].append(dict(n=n, npos=npos, max_R_over_bijection_bound=worstC))
        out["E_key"].append(dict(n=n, max_target_over_cost=worstE, max_prodratio=worstD,
                                 max_V_over_hadamard=Vh))
        print(f"n={n}: (A) kill residual <= {out['A_kill']:.2e} | "
              f"(B) max R/||c||_1 over N<=n = {max(d['max_R_over_l1'] for d in out['B_Nlen'] if d['n']==n):.3e} | "
              f"[{npos} configs with R>0] (C) max R/bijection-bound = {worstC:.6f} | "
              f"(D) max (R/(n+1))^(n+1)/prod|c| = {worstD:.6f}, max V/Hadamard = {Vh:.6f} | "
              f"(E) max (n+1)^(1-r)R^r / sum|c|^r = {worstE:.6f}", flush=True)
        out["D_had"] = []
    # (F) margin identity against the real table
    for (p, n) in [(7, 2), (11, 2), (13, 2), (7, 3), (11, 3), (5, 4)]:
        err = 0.0
        for _ in range(6):
            N = n + 2
            c, v = rand_cfg(N, n, g)
            E, W = net_from_atoms(c, v, p)
            m_true = brute_margin(E, W, p, n)
            M1 = 1 - math.cos(2 * math.pi / p); s = math.sin(2 * math.pi / p)
            th = torch.arange(p) * (2 * math.pi / p)
            u1 = torch.polar(torch.ones(p), th)
            idx = torch.cartesian_prod(*[torch.arange(p)] * n)
            H = h_on_points(c, v, u1[idx.reshape(-1, n)])
            thD = torch.arange(1, p) * (2 * math.pi / p)
            # exact:  Re[(1 - e^{i th_D}) H] = (1 - cos th_D) Re H + sin(th_D) Im H
            m_id = float(((1 - thD.cos())[:, None] * H.real[None, :]
                          + thD.sin()[:, None] * H.imag[None, :]).min()) * 2.0 ** (-n)
            # the upper bound actually used in the proof (tight iff |arg H| <= pi/p, i.e. margin >= 0)
            m_bd = float((M1 * H.real - s * H.imag.abs()).min()) * 2.0 ** (-n)
            err = max(err, abs(m_true - m_id) / max(1e-12, abs(m_true)))
            assert m_id <= m_bd + 1e-12 * max(1.0, abs(m_bd))
        out["F_margin"].append(dict(p=p, n=n, max_rel_err=err))
        print(f"(F) p={p} n={n}: margin identity rel err {err:.3e}", flush=True)
    d = ROOT / "FINAL_RUNS" / "GOPT"; d.mkdir(parents=True, exist_ok=True)
    (d / "verify.json").write_text(json.dumps(out, indent=1))
    print("wrote", d / "verify.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
