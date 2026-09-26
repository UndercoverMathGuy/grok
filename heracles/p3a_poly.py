"""p3a_poly.py -- the POLYDISC RADIUS of the junk-mode moment body: the constant that
decides P3 through the Riesz-product route.

Set-up.  For a set of distinct non-zero modes  m_1..m_M  in Z_p^n  write, for a
probability density w on Z_p^n,   mu(m) := sum_a w(a) omega^{-m.a}   (omega = e^{2 pi i/p}).
Define the POLYDISC RADIUS

    A(m_1..m_M ; p, n) := min over target phases phi_1..phi_M of
                          max over probability densities w of
                          min_i  Re[ e^{-i phi_i} mu(m_i) ] .

Plain reading: "how large a moment can I put on EVERY one of these M modes at once,
with whatever phase I like, using a single non-negative test density".

Two facts.
 * M = 1 gives A = cos(pi/p) (a point mass, aimed to within pi/p of the target phase).
 * If the mode set is DISSOCIATE (the 3^M sums sum_i eps_i m_i, eps in {0,+-1}, are all
   distinct), the Riesz product  w(a) = p^{-n} prod_i (1 + 2 Re(lambda_i omega^{m_i.a}))
   is non-negative, has total mass 1 and has mu(m_i) = conj(lambda_i) EXACTLY, so
   A >= 1/2 -- independently of M.  This is the only mechanism known that does not
   dilute by the number of frequencies.

Why P3 needs it (P3A.md sec 2-3).  Picking, for each frequency k, the sign pattern
T_k that carries its largest junk coefficient, a density with |mu(m_{k,T_k})| >= A and
free phases gives the margin penalty

    Pen(D) >= A * sum_k |1 - e^{i theta_kD}| * max_T |C_T^{(k)}|
            >= A * sum_k sqrt(2 w_kD) * eta_k x_k / sqrt(2^n - 1),

and (T) then follows for every junk level and every K provided

    A / sqrt(2^n - 1)  >=  rho_need(n,q) = max_eta [1 - phi(eta)^{-1}]/eta ,

whose value at the (exactly known) endpoint eta = sqrt(2^n-1) is
(1 - (n+1)^{-(nu-q)/q})/sqrt(2^n-1).  At A = 1/2 the sqrt(2^n-1) cancels and the
condition is exactly  (n+1)^{(nu-q)/q} <= 2.

This script computes A exactly (an LP in w for fixed phases, minimised over the phases)
for the junk-mode sets that actually occur.

Usage
    uv run python p3a_poly.py --p 11 13 17 19 23 --n 2
    uv run python p3a_poly.py --p 11 13 --n 2 --allT

All float64, local CPU, no pods.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3a_prof import OUT, gamma_s, subsets  # noqa: E402
from p3a_lemma import dissociate  # noqa: E402


def char_matrix(p, n, mlist):
    """X[i, a] = omega^{-m_i . a}."""
    r = np.arange(p ** n)
    ops = np.stack([(r // p ** j) % p for j in range(n)], 1)
    M = np.array(mlist, dtype=np.int64)
    ph = (ops @ M.T).T % p                       # (M, p^n)
    return np.exp(-2j * np.pi * ph / p)


def A_given_phase(X, phi):
    """max over prob densities w of min_i Re[e^{-i phi_i} mu(m_i)]  (an LP)."""
    M, P = X.shape
    R = (np.exp(-1j * phi)[:, None] * X).real           # (M, P)
    # variables z = (w_1..w_P, t); maximise t
    c = np.zeros(P + 1)
    c[-1] = -1.0
    A_ub = np.zeros((M, P + 1))
    A_ub[:, :P] = -R
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(M)
    A_eq = np.zeros((1, P + 1))
    A_eq[0, :P] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * P + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method='highs')
    if not res.success:
        return -np.inf
    return float(res.x[-1])


def polydisc_radius(p, n, mlist, grid=8, polish=True, seed=0):
    """A = min over phases of the LP value.  Exhaustive coarse grid + Nelder-Mead polish."""
    X = char_matrix(p, n, mlist)
    M = X.shape[0]
    best = np.inf
    bphi = None
    if M <= 4:
        cand = itertools.product(np.linspace(0, 2 * np.pi, grid, endpoint=False),
                                 repeat=M - 1)
        starts = [np.concatenate([[0.0], np.array(c)]) for c in cand]
    else:
        rng = np.random.default_rng(seed)
        starts = [np.concatenate([[0.0], rng.uniform(0, 2 * np.pi, M - 1)])
                  for _ in range(200)]
    for phi in starts:
        v = A_given_phase(X, phi)
        if v < best:
            best, bphi = v, phi.copy()
    if polish and bphi is not None:
        res = minimize(lambda ph: A_given_phase(X, np.concatenate([[0.0], ph])),
                       bphi[1:], method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12})
        if res.fun < best:
            best, bphi = float(res.fun), np.concatenate([[0.0], res.x])
    return best, bphi


def astar(p, mlist, n=1, starts=300, seed=0):
    """A*(modes) = min over z of  max_a Re sum_i z_i omega^{-m_i.a} / sum_i |z_i| .

    This is the EXACT inradius of the moment body for the polydisc: A* is the largest
    rho such that EVERY phase vector is realisable with all moments of modulus exactly
    rho (equivalently 1/Sidon-constant of the mode set).  A Riesz product certifies
    A* >= 1/2 on a dissociate set; the Dirichlet kernel shows A* = 1/(p-1) when the
    modes are all of Z_p.
    """
    X = char_matrix(p, n, mlist)                    # (M, p^n)
    M = X.shape[0]
    Xr, Xi = X.real, X.imag
    rng = np.random.default_rng(seed)

    def val(w):
        a, b = w[:M], w[M:]
        f = a @ Xr - b @ Xi
        d = np.sqrt(a ** 2 + b ** 2).sum()
        return 1e9 if d <= 1e-300 else float(f.max() / d)

    def soft(w, tau):
        a, b = w[:M], w[M:]
        f = a @ Xr - b @ Xi
        mx = f.max()
        e = np.exp((f - mx) / tau)
        Z = e.sum()
        fs = mx + tau * np.log(Z)
        Om = e / Z
        nk = np.sqrt(a ** 2 + b ** 2)
        d = nk.sum()
        if d <= 1e-12:
            return 1e9, np.zeros_like(w)
        g_num = np.concatenate([Xr @ Om, -(Xi @ Om)])
        sc = np.where(nk > 1e-12, 1.0 / np.maximum(nk, 1e-12), 0.0)
        g_den = np.concatenate([sc * a, sc * b])
        v = fs / d
        return v, (g_num - v * g_den) / d

    best, bw = np.inf, None
    for s in range(starts):
        if s == 0:
            w = np.concatenate([-np.ones(M), np.zeros(M)])
        elif s == 1:
            w = np.concatenate([np.ones(M), np.zeros(M)])
        else:
            w = rng.standard_normal(2 * M)
        for tau in (0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4):
            res = minimize(soft, w, args=(tau,), jac=True, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13})
            w = res.x / max(1e-300, np.linalg.norm(res.x))
        res = minimize(val, w, method='Nelder-Mead',
                       options={'maxiter': 30000, 'xatol': 1e-12, 'fatol': 1e-14})
        v = val(res.x)
        if v < best:
            best, bw = v, res.x.copy()
    return best, bw


def block_astar(args):
    n, q = args.n, args.q
    r = q / (n + 1)
    phimax = (n + 1) ** ((1 - r) / r)
    needA = 1.0 - phimax ** (-1.0 / n)
    print(f"n={n} q={q}  phi_max={phimax:.9f}   one-atom-per-frequency theorem needs "
          f"A* >= 1 - phi_max^(-1/n) = {needA:.9f}  (on a Gamma-optimal support)\n")
    rows = []
    for p in args.p:
        G, xopt = gamma_s(p, r, starts=args.gstarts)
        nf = (p - 1) // 2
        allk = list(range(1, nf + 1))
        S = [allk[i] for i in np.nonzero(xopt > 1e-7 * xopt.max())[0]]
        for tag, ks in (('Gamma-optimal', S), ('all frequencies', allk)):
            ml = [[(2 * k) % p] for k in ks]
            A, _ = astar(p, ml, n=1, starts=args.astarts)
            dis = dissociate([np.array(m) for m in ml], p, 1)
            print(f"  p={p:3d} {tag:<16s} S={str(ks):<26s} K={len(ks):2d}  "
                  f"dissociate={str(dis):<5s}  A* = {A:.9f}  "
                  f"(1-A*)^n = {(1-A)**n:.6f}   {'OK' if A >= needA else 'FAIL'}")
            rows.append(dict(p=p, tag=tag, S=ks, K=len(ks), dissociate=dis, A=A,
                             needA=needA))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'astar_n{n}_q{q}.json').write_text(json.dumps(rows, indent=1))
    print('\nwrote', OUT / f'astar_n{n}_q{q}.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int, nargs='+', default=[11, 13, 17, 19, 23])
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--q', type=float, default=2.0)
    ap.add_argument('--grid', type=int, default=8)
    ap.add_argument('--gstarts', type=int, default=200)
    ap.add_argument('--allT', action='store_true',
                    help='sweep every choice of one sign pattern per frequency')
    ap.add_argument('--block', default='poly', choices=['poly', 'astar'])
    ap.add_argument('--astarts', type=int, default=120)
    args = ap.parse_args()
    if args.block == 'astar':
        block_astar(args)
        return
    n, q = args.n, args.q
    r = q / (n + 1)
    nu = n + 1
    phimax = nu ** ((1 - r) / r)
    need_end = (1 - 1 / phimax) / np.sqrt(2 ** n - 1)
    print(f"n={n} nu={nu} q={q} r={r:.6f}   phi_max = (n+1)^((nu-q)/q) = {phimax:.9f}")
    print(f"rho_need at the endpoint = (1 - 1/phi_max)/sqrt(2^n-1) = {need_end:.9f}")
    print(f"the Riesz constant A = 1/2 would give A/sqrt(2^n-1) = "
          f"{0.5/np.sqrt(2**n-1):.9f}   -> {'SUFFICES' if 0.5/np.sqrt(2**n-1) >= need_end else 'NOT ENOUGH'}"
          f"   (equivalent to phi_max <= 2)\n")
    Ts = subsets(n, nonempty=True)
    rows = []
    for p in args.p:
        G, xopt = gamma_s(p, r, starts=args.gstarts)
        nf = (p - 1) // 2
        allk = list(range(1, nf + 1))
        S = [allk[i] for i in np.nonzero(xopt > 1e-7 * xopt.max())[0]]
        print(f"p={p}  Gamma={G:.9f}  Gamma-optimal support S={S}")
        choices = ([tuple([t] * len(S)) for t in range(len(Ts))] if not args.allT
                   else list(itertools.product(range(len(Ts)), repeat=len(S))))
        worst, wsel = np.inf, None
        for sel in choices:
            mlist = []
            for i, k in enumerate(S):
                m = [0] * n
                for j in Ts[sel[i]]:
                    m[j] = (2 * k) % p
                mlist.append(m)
            if len({tuple(m) for m in mlist}) < len(mlist):
                continue                       # repeated mode: not a valid mode set
            dis = dissociate([np.array(m) for m in mlist], p, n)
            A, phi = polydisc_radius(p, n, mlist, grid=args.grid)
            tag = "".join("".join(str(j + 1) for j in Ts[s]) + "," for s in sel)[:-1]
            if not args.allT:
                print(f"   T_k = {tag:<20s} dissociate={str(dis):<5s}  A = {A:.9f}  "
                      f"A/sqrt(2^n-1) = {A/np.sqrt(2**n-1):.9f}  "
                      f"{'OK' if A/np.sqrt(2**n-1) >= need_end else 'FAIL'}")
            if A < worst:
                worst, wsel = A, (tag, dis)
            rows.append(dict(p=p, S=S, T=tag, dissociate=dis, A=A))
        print(f"   ---> worst over the {len(choices)} pattern choices: A = {worst:.9f}"
              f"   (at {wsel[0]}, dissociate={wsel[1]})   A/sqrt(2^n-1) = "
              f"{worst/np.sqrt(2**n-1):.9f}  "
              f"{'OK' if worst/np.sqrt(2**n-1) >= need_end else 'FAIL'}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'polydisc_n{n}_q{q}.json').write_text(json.dumps(
        dict(n=n, q=q, phimax=phimax, need_end=need_end, rows=rows), indent=1))
    print('wrote', OUT / f'polydisc_n{n}_q{q}.json')


if __name__ == '__main__':
    main()
