"""gopt_atom.py -- the ATOM PROGRAMME with junk allowed (new reduction, GLOBAL_OPT.md Sec. 2-3).

Setting.  Gated chain, one frequency k, neurons that are pure single-frequency cosines.  Writing
    c_l  = (prod_j r_lj) rho_l e^{i psi_l}   (neuron l's signed clean amplitude, psi_l = sum_j alpha_lj - beta_l)
    v_lj = e^{-2 i alpha_lj}                 (unit modulus, records the phase of neuron l's j-th embedding)
the WHOLE logit table of the population is carried by the single multi-affine polynomial

    h(u) = sum_l c_l prod_{j=1}^n (1 + u_j v_lj),      u_j on the unit circle,

because  z(a, s(a) - D) = 2^{-n} Re[ e^{i theta_D} h(u(a)) ]  with u_j(a) = e^{-2 i theta_{a_j}},
theta_x = 2 pi k x / p.  Hence, exactly,

    Phi_D(a) := z(a, s(a)) - z(a, s(a) - D) = 2^{-n} Re[ (1 - e^{i theta_D}) h(u(a)) ],
    margin   = 2^{-n} min over a in Z_p^n, D != 0  of  Re[ (1 - e^{i theta_D}) h(u(a)) ].

Plain reading: h is a "generating function" for the 2^n Fourier sign-patterns of the population at this
frequency; its CONSTANT term is the clean amplitude, and h being non-constant is exactly junk.
The (n+1)-neuron common-phase block is the configuration h == const.

Two quantities are tested here:
    R_torus = min over the full torus u in T^n of Re h(u)            (continuum idealisation)
    R_grid  = min over u in (p-th roots of unity)^n of Re h(u)       (what the task actually sees)
and the two candidate inequalities

    (K-torus)  sum_l |c_l|^r >= (n+1)^{1-r} max(R_torus, 0)^r        r = q/nu in (0,1]
    (K-grid)   sum_l |c_l|^r >= (n+1)^{1-r} max(R_grid , 0)^r

(K-grid) is EXACTLY the single-frequency case of the target inequality (T); (K-torus) implies it only
in the limit p -> infinity, since R_grid >= R_torus.

Subcommands:
  block    -- build the (n+1)-block, check h == const and that (T) is attained with ratio 1
  torus    -- maximise R_torus / ((n+1)^{-(1-r)/r} ||c||_r) over c, v for N = n .. n+5  (tests G1 + junk)
  grid     -- maximise the EXACT single-frequency margin ratio on Z_p (tests (T) itself at finite p)
All float64.  Local CPU.
"""
import argparse, itertools, json, math
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- core evaluation
def h_on_points(c, v, U):
    """h(u) at a batch of points.  c (N,) complex, v (N,n) complex, U (M,n) complex -> (M,) complex."""
    # prod_j (1 + U[m,j] v[l,j])
    P = (1.0 + U[:, None, :] * v[None, :, :]).prod(-1)     # (M, N)
    return P @ c


def grid_points(n, m):
    """m equally spaced points on the circle in each of n slots -> (m^n, n) complex."""
    th = torch.arange(m) * (2 * math.pi / m)
    u = torch.polar(torch.ones(m), th)
    mesh = torch.cartesian_prod(*[torch.arange(m)] * n) if n > 1 else torch.arange(m)[:, None]
    return u[mesh.reshape(-1, n)]


def R_torus(c, v, m=None):
    """min over the torus of Re h, computed EXACTLY in slot 1 (h is affine in u_1) and on a grid in the rest.

    For fixed u_2..u_n:  min over u_1 of Re(A + u_1 B) = Re A - |B|,  A = h with u_1 = 0, B = dh/du_1.
    """
    n = v.shape[1]
    if m is None:
        m = 4096 if n == 2 else (256 if n == 3 else 48)
    if n == 1:
        A = c.sum(); B = (c * v[:, 0]).sum()
        return (A.real - B.abs())
    U = grid_points(n - 1, m)                                  # (m^(n-1), n-1)
    P = (1.0 + U[:, None, :] * v[None, :, 1:]).prod(-1)        # (M, N)
    A = P @ c
    B = P @ (c * v[:, 0])
    return (A.real - B.abs()).min()


def R_grid(c, v, p):
    """min over u in (mu_p)^n of Re h(u)  (exact, all p^n points)."""
    n = v.shape[1]
    th = torch.arange(p) * (2 * math.pi / p)
    u = torch.polar(torch.ones(p), th)
    idx = torch.cartesian_prod(*[torch.arange(p)] * n) if n > 1 else torch.arange(p)[:, None]
    U = u[idx.reshape(-1, n)]
    return h_on_points(c, v, U).real.min()


def margin_grid(c, v, p, k=1):
    """EXACT margin of the pure single-frequency population, as a multiple of 2^{-n}.

    margin = 2^{-n} min_{a, D != 0} Re[(1 - e^{i theta_D}) h(u(a))];  u(a) ranges over all of (mu_p)^n
    because 2k is invertible mod p, and D ranges over the p-1 non-zero residues independently.
    """
    n = v.shape[1]
    th = torch.arange(p) * (2 * math.pi / p)
    u = torch.polar(torch.ones(p), th)
    idx = torch.cartesian_prod(*[torch.arange(p)] * n) if n > 1 else torch.arange(p)[:, None]
    H = h_on_points(c, v, u[idx.reshape(-1, n)])                       # (p^n,)
    D = torch.arange(1, p)
    zeta = 1.0 - torch.polar(torch.ones(p - 1), D * (2 * math.pi * k / p))   # (p-1,)
    return (zeta[:, None] * H[None, :]).real.min()


def cost_r(c, r):
    return (c.abs() ** r).sum()


# ---------------------------------------------------------------- the (n+1)-block
def block(n, A=1.0):
    """c_l = A/(n+1),  v_lj = e^{-2 pi i l/(n+1)}  (all slots the same rotation) -> h == A."""
    N = n + 1
    c = torch.full((N,), A / N, dtype=torch.cdouble)
    w = torch.polar(torch.ones(N), -2 * math.pi * torch.arange(N) / N)
    v = w[:, None].repeat(1, n)
    return c, v


# ---------------------------------------------------------------- optimisation
def ascent(n, N, r, objective, steps=4000, restarts=24, seed=0, beta0=20.0, beta1=4000.0, verbose=False):
    """Maximise objective(c, v) / ||c||_r by Adam on (Re c, Im c, phases).  objective is a SOFT min."""
    g = torch.Generator().manual_seed(seed)
    best = (-1e18, None, None)
    for t in range(restarts):
        x = torch.randn(2 * N, generator=g) / N
        ph = torch.rand(N, n, generator=g) * 2 * math.pi
        x.requires_grad_(True); ph.requires_grad_(True)
        opt = torch.optim.Adam([x, ph], lr=0.03)
        for s in range(steps):
            beta = beta0 * (beta1 / beta0) ** (s / max(steps - 1, 1))
            c = torch.complex(x[:N], x[N:])
            v = torch.polar(torch.ones(N, n), ph)
            val = objective(c, v, beta)
            cst = (c.abs() ** 2 + 1e-24).pow(r / 2).sum().pow(1.0 / r)
            loss = -val / cst
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            c = torch.complex(x[:N], x[N:]); v = torch.polar(torch.ones(N, n), ph)
            yield c.detach().clone(), v.detach().clone()


def softmin(t, beta):
    return -torch.logsumexp(-beta * t, dim=0) / beta


# ---------------------------------------------------------------- subcommands
def cmd_block(a):
    out = []
    for n in (1, 2, 3, 4):
        c, v = block(n)
        rt = float(R_torus(c, v))
        dev = float((h_on_points(c, v, grid_points(n, 8 if n <= 3 else 4)) - 1.0).abs().max())
        row = dict(n=n, R_torus=rt, max_dev_h_from_const=dev)
        for p in (7, 11, 23, 41):
            if p ** n > 3_000_000:
                continue
            rg = float(R_grid(c, v, p))
            mg = float(margin_grid(c, v, p))
            M1 = 1 - math.cos(2 * math.pi / p)
            row[f"p{p}"] = dict(R_grid=rg, margin_over_M1=mg / M1)
        out.append(row)
        print(f"n={n}: h const to {dev:.2e}, R_torus={rt:.12f}, "
              + "  ".join(f"p{p}: m/M1={row[f'p{p}']['margin_over_M1']:.12f}" for p in (7, 11, 23, 41)
                          if f"p{p}" in row), flush=True)
    return out


def cmd_torus(a):
    """max R_torus / ((n+1)^{-(1-r)/r} ||c||_r) over c, v.  > 1 would break (K-torus)."""
    rows = []
    for n in a.ns:
        for q in a.qs:
            nu = n + 1
            if q > nu:
                continue
            r = q / nu
            tgt_const = (n + 1) ** (-(1 - r) / r)
            for N in range(max(1, n - 1), n + 1 + a.extra + 1):
                mgrid = 512 if n == 2 else (64 if n == 3 else 24)

                def obj(c, v, beta, mgrid=mgrid, n=n):
                    if n == 1:
                        A = c.sum(); B = (c * v[:, 0]).sum()
                        return A.real - B.abs()
                    U = grid_points(n - 1, mgrid)
                    P = (1.0 + U[:, None, :] * v[None, :, 1:]).prod(-1)
                    t = (P @ c).real - (P @ (c * v[:, 0])).abs()
                    return softmin(t, beta)

                best = (-1e18, None, None)
                for c, v in ascent(n, N, r, obj, steps=a.steps, restarts=a.restarts, seed=a.seed):
                    R = float(R_torus(c, v))          # exact-in-slot-1, fine grid elsewhere
                    val = R / float(cost_r(c, r)) ** (1 / r)
                    if val > best[0]:
                        best = (val, c, v)
                ratio = best[0] / tgt_const
                amps = sorted(best[1].abs().tolist(), reverse=True)
                rows.append(dict(n=n, q=q, r=r, N=N, value=best[0], target=tgt_const, ratio=ratio,
                                 amps=[round(u, 6) for u in amps]))
                print(f"n={n} q={q:g} r={r:.4f} N={N:2d}  max R/||c||_r = {best[0]:.8f}   "
                      f"target (n+1)^(-(1-r)/r) = {tgt_const:.8f}   ratio {ratio:.6f}   "
                      f"amps {[f'{u:.4f}' for u in amps[:7]]}", flush=True)
    return rows


def cmd_grid(a):
    """max (exact single-frequency margin) / (target) on Z_p.  > 1 breaks (T) restricted to one frequency."""
    rows = []
    for p in a.ps:
        M1 = 1 - math.cos(2 * math.pi / p)
        for n in a.ns:
            if p ** n > 2_000_000:
                continue
            for q in a.qs:
                nu = n + 1
                if q > nu:
                    continue
                r = q / nu
                tgt_const = (n + 1) ** (-(1 - r) / r)
                th = torch.arange(p) * (2 * math.pi / p)
                uu = torch.polar(torch.ones(p), th)
                idx = torch.cartesian_prod(*[torch.arange(p)] * n) if n > 1 else torch.arange(p)[:, None]
                U = uu[idx.reshape(-1, n)]
                D = torch.arange(1, p)
                zeta = 1.0 - torch.polar(torch.ones(p - 1), D * (2 * math.pi / p))

                def obj(c, v, beta, U=U, zeta=zeta):
                    H = h_on_points(c, v, U)
                    t = (zeta[:, None] * H[None, :]).real.reshape(-1)
                    return softmin(t, beta) / M1

                for N in range(max(1, n), n + 1 + a.extra + 1):
                    best = (-1e18, None, None)
                    for c, v in ascent(n, N, r, obj, steps=a.steps, restarts=a.restarts, seed=a.seed):
                        val = float(margin_grid(c, v, p)) / M1 / float(cost_r(c, r)) ** (1 / r)
                        if val > best[0]:
                            best = (val, c, v)
                    # also seed with the block
                    cb, vb = block(n)
                    vb0 = float(margin_grid(cb, vb, p)) / M1 / float(cost_r(cb, r)) ** (1 / r)
                    ratio = best[0] / tgt_const
                    rows.append(dict(p=p, n=n, q=q, r=r, N=N, value=best[0], block=vb0,
                                     target=tgt_const, ratio=ratio,
                                     amps=[round(u, 6) for u in sorted(best[1].abs().tolist(), reverse=True)]))
                    print(f"p={p} n={n} q={q:g} N={N:2d}  best {best[0]:.8f}  block {vb0:.8f}  "
                          f"target {tgt_const:.8f}  ratio {ratio:.6f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["block", "torus", "grid"])
    ap.add_argument("--ns", type=int, nargs="+", default=[2])
    ap.add_argument("--ps", type=int, nargs="+", default=[7, 11, 23])
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0])
    ap.add_argument("--extra", type=int, default=3, help="max N = n+1+extra")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    res = {"block": cmd_block, "torus": cmd_torus, "grid": cmd_grid}[a.cmd](a)
    d = ROOT / "FINAL_RUNS" / "GOPT"; d.mkdir(parents=True, exist_ok=True)
    (d / f"atom_{a.cmd}{a.tag}.json").write_text(json.dumps(res, indent=1))
    print("wrote", d / f"atom_{a.cmd}{a.tag}.json")
