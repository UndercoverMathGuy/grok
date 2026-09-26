"""p3a_truth.py -- P3: how much margin does junk on K >= 2 frequencies REALLY cost?

The question (problem P3 of GOPT_STATE.md).  Everything about the junk trade is
currently phrased with two per-frequency numbers:

    x_k    = |C_empty^{(k)}|                         the CLEAN amplitude of frequency k
    eta_k  = ||junk of k||_2 / x_k                   its RELATIVE junk, in l_2

and two ingredients:

    (cost)    sum_{k,l} |c_lk|^r  >=  sum_k Cl2_r(eta_k) x_k^r         [Theorem J's curve]
    (margin)  Lemma Q: 2^n margin <= min_D [ sum_k x_k w_kD
                                             - (2^n-1)^{-1/2} max_k eta_k x_k sqrt(w_kD/2) ]

with  w_kD = 1 - cos(2 pi k D / p).  Agent 4 showed the pair closes (T) only for
eta below an explicit threshold (0.67 at K=2 down to 0.19 at K=10, nu=3 q=2), and
that at the top of the junk range the pair leaves a residual factor V/target = 1.30.

BEFORE inventing a stronger margin lemma one has to know which inequality is even
TRUE.  This script computes, for pure networks, the best possible bound that the
(x, eta) bookkeeping can ever give, by replacing Lemma Q with the EXACT margin of
the profile:

    Vtilde(p) := max over profiles (x_k >= 0, C_T^{(k)} in C) of
                 Lambda(profile) / ( sum_k Cl2_r(eta_k) x_k^r )^{1/r},

    Lambda = min over a in Z_p^n, D != 0 of sum_k Re[(1 - e^{2 pi i kD/p}) h_k(a)],
    h_k(a) = sum_T C_T^{(k)} omega^{-2 k sigma_T(a)},   sigma_T(a) = sum_{j in T} a_j.

Reading of the result:
  * Vtilde <= target  =>  the (x, eta) framework CAN close P3; what is missing is
    only a provable margin lemma, and Vtilde says how strong it has to be.
  * Vtilde >  target  =>  the framework is DEAD however strong the margin lemma is,
    because the cost side (Cl2 = the cheapest shape at junk level eta) and the margin
    side (the actual shape, chosen by the adversary) are evaluated at different shapes.

The script also reports the EXACT ratio at the arg-max, i.e. with the cost of the
profile's own shape,  Cprof_r(|C_T|),  which is what a real network pays.  That is a
genuine test of (T) itself on pure networks, in profile coordinates.

Usage
    uv run python p3a_truth.py --mode check
    uv run python p3a_truth.py --mode vtilde --p 11 13 17 19 23 --starts 60
    uv run python p3a_truth.py --mode curve --p 13 --starts 40

All float64, local CPU, no pods.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3a_prof import (OUT, Profile, cl2_grid, cprof, gamma_s, subsets)  # noqa: E402


# ------------------------------------------------------------------ verification
def check_identity(p, n, ks, N, seed=0):
    """Brute-force test that 2^n * margin of a pure network = Lambda of its profile."""
    rng = np.random.default_rng(seed)
    K = len(ks)
    c = rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
    alpha = rng.uniform(0, 2 * np.pi, (K, N, n))           # embedding phases
    beta = rng.uniform(0, 2 * np.pi, (K, N))               # readout phase
    rad = np.abs(c)
    psi = np.angle(c)
    # a pure neuron: e_lj(a) = rad_lj cos(2 pi k a/p + alpha_lj), w_l(c) = rho cos(... + beta)
    # choose rad_lj = 1 for j < n, rad_ln = rad_l, rho_l = 1, and set
    # psi_l = sum_j alpha_lj - beta_l by solving for beta.
    beta = alpha.sum(axis=2) - psi
    a_grid = np.arange(p) * 2 * np.pi / p
    m = K * N
    E = np.zeros((n, m, p))
    W = np.zeros((p, m))
    i = 0
    for kk, k in enumerate(ks):
        for l in range(N):
            for j in range(n):
                amp = rad[kk, l] if j == 0 else 1.0
                E[j, i] = amp * np.cos(k * a_grid + alpha[kk, l, j])
            W[:, i] = np.cos(k * a_grid + beta[kk, l])
            i += 1
    # exact logit table
    rows = np.arange(p ** n)
    ops = np.stack([(rows // p ** j) % p for j in range(n)], 1)
    h = np.ones((p ** n, m))
    for j in range(n):
        h *= E[j][:, ops[:, j]].T
    z = h @ W.T                                            # (p^n, p)
    y = ops.sum(1) % p
    d = z[np.arange(p ** n), y][:, None] - z                # z_correct - z_c
    d[np.arange(p ** n), y] = np.inf
    margin = float(d.min())
    # profile
    pr = Profile(p, n, ks)
    Ts = pr.Ts
    C = np.zeros((K, len(Ts)), dtype=complex)
    v = np.exp(-2j * alpha)                                 # (K, N, n)
    cc = rad * np.exp(1j * psi)
    for t, T in enumerate(Ts):
        w = np.ones((K, N), dtype=complex)
        for j in T:
            w = w * v[:, :, j]
        C[:, t] = (cc * w).sum(axis=1)
    lam = pr.Lam(C)
    return margin, lam / 2 ** n, abs(margin - lam / 2 ** n) / max(1e-30, abs(margin))


# ------------------------------------------------------------------ the maximisation
class Obj:
    """ratio = Lambda / cost^{1/r} / target, over profiles, cost from the Cl2 curve."""

    def __init__(self, p, n, ks, r, fgrid, target):
        self.pr = Profile(p, n, ks)
        self.n, self.r, self.target = n, r, target
        self.K = self.pr.K
        self.nTj = 2 ** n - 1
        self.eta_grid, self.C_grid = fgrid
        self.emax = float(self.eta_grid[-1])
        # derivative of the Cl2 curve (monotone decreasing, piecewise linear)
        self.dC_grid = np.gradient(self.C_grid, self.eta_grid)

    def f(self, eta):
        return np.interp(np.clip(eta, 0.0, self.emax), self.eta_grid, self.C_grid)

    def df(self, eta):
        out = np.interp(np.clip(eta, 0.0, self.emax), self.eta_grid, self.dC_grid)
        return np.where(eta > self.emax, 0.0, out)

    def unpack(self, z):
        K, nTj = self.K, self.nTj
        u = z[:K]
        A = z[K:K + K * nTj].reshape(K, nTj)
        B = z[K + K * nTj:].reshape(K, nTj)
        return u, A, B

    def profile(self, z):
        u, A, B = self.unpack(z)
        x = u ** 2
        C = np.zeros((self.K, 1 + self.nTj), dtype=complex)
        C[:, 0] = x
        C[:, 1:] = A + 1j * B
        return x, C

    def pieces(self, z):
        u, A, B = self.unpack(z)
        x = u ** 2 + 1e-300
        s = np.sqrt((A ** 2 + B ** 2).sum(1))               # ||junk_k||_2
        eta = s / x
        cost = float(np.sum(self.f(eta) * x ** self.r))
        return x, s, eta, cost

    def ratio_exactmin(self, z):
        x, s, eta, cost = self.pieces(z)
        if cost <= 1e-14:
            return -1e9
        _, C = self.profile(z)
        return (self.pr.Lam(C) / cost ** (1.0 / self.r)) / self.target

    def neg_soft(self, z, tau):
        """-(softmin Lambda)/cost^{1/r}/target, plus gradient."""
        x, s, eta, cost = self.pieces(z)
        if cost <= 1e-14:
            return 1e9, np.zeros_like(z)
        _, C = self.profile(z)
        M = self.pr.M(C)                                     # (p^n, p-1)
        mn = M.min()
        e = np.exp(-(M - mn) / tau)
        Z = e.sum()
        lam = mn - tau * math.log(Z)
        Om = e / Z                                           # softmin weights
        cinv = cost ** (-1.0 / self.r)
        val = lam * cinv / self.target

        # d lam / d profile
        alpha = np.einsum('ad,kd->ka', Om, self.pr.W.real)   # (K, p^n)
        betaI = np.einsum('ad,kd->ka', Om, self.pr.W.imag)
        Pr, Pi = self.pr.PH.real, self.pr.PH.imag            # (K, nT, p^n)
        dl_dx = alpha.sum(1)                                 # T = empty is real x_k
        dl_dA = np.einsum('ka,kta->kt', alpha, Pr[:, 1:]) - np.einsum('ka,kta->kt', betaI, Pi[:, 1:])
        dl_dB = -np.einsum('ka,kta->kt', alpha, Pi[:, 1:]) - np.einsum('ka,kta->kt', betaI, Pr[:, 1:])

        # d cost / d profile
        fe, dfe = self.f(eta), self.df(eta)
        xr1 = x ** (self.r - 1.0)
        dc_dx = xr1 * (self.r * fe - dfe * eta)
        u, A, B = self.unpack(z)
        sc = np.where(s > 1e-300, dfe * xr1 / np.maximum(s, 1e-300), 0.0)
        dc_dA = sc[:, None] * A
        dc_dB = sc[:, None] * B

        pref = cinv / self.target
        g_x = pref * dl_dx - (val / (self.r * cost)) * dc_dx
        g_A = pref * dl_dA - (val / (self.r * cost)) * dc_dA
        g_B = pref * dl_dB - (val / (self.r * cost)) * dc_dB
        g_u = g_x * 2.0 * u
        return -val, -np.concatenate([g_u, g_A.ravel(), g_B.ravel()])


def _renorm(ob, z):
    """rescale the (scale-invariant) profile so that max_k x_k = 1."""
    u, A, B = ob.unpack(z)
    mx = float((u ** 2).max())
    if not np.isfinite(mx) or mx <= 0:
        return z
    s = 1.0 / mx
    return np.concatenate([u * math.sqrt(s), (A * s).ravel(), (B * s).ravel()])


def maximise(ob, starts, seed=0, taus=(0.5, 0.2, 0.08, 0.03, 0.01, 0.003, 0.001),
             x0hint=None):
    """Maximise the ratio.  Starts: junk-free block, block + noise, MAXIMUM junk
    (one atom per frequency: |C_T| = x_k for every T, rank-one phases), random."""
    rng = np.random.default_rng(seed)
    K, nTj = ob.K, ob.nTj
    n = ob.n
    Ts = subsets(n, nonempty=True)
    hint = np.maximum(x0hint, 0.0) if x0hint is not None else rng.uniform(0, 1, K)

    def rank1(x, ph):
        """single-atom junk profile: C_T = x_k prod_{j in T} e^{i ph_kj}."""
        C = np.zeros((K, nTj), dtype=complex)
        for t, T in enumerate(Ts):
            w = np.ones(K, dtype=complex)
            for j in T:
                w = w * np.exp(1j * ph[:, j])
            C[:, t] = x * w
        return C.real, C.imag

    best, bz = -np.inf, None
    for s in range(starts):
        if s == 0:                                   # the junk-free block
            u = np.sqrt(hint)
            A = np.zeros((K, nTj)); B = np.zeros((K, nTj))
        elif s < 5:                                  # block + junk noise
            u = np.sqrt(hint)
            sc = 10.0 ** (-s) * max(hint.max(), 1e-12)
            A = rng.standard_normal((K, nTj)) * sc
            B = rng.standard_normal((K, nTj)) * sc
        elif s < 13:                                 # MAXIMUM junk, programme shares
            u = np.sqrt(hint)
            A, B = rank1(hint, rng.uniform(0, 2 * np.pi, (K, n)))
        elif s < 21:                                 # MAXIMUM junk, flat shares
            x = rng.uniform(0.3, 1.0, K)
            u = np.sqrt(x)
            A, B = rank1(x, rng.uniform(0, 2 * np.pi, (K, n)))
        elif s < 29:                                 # MAXIMUM junk on a random sub-support
            x = np.zeros(K)
            idx = rng.choice(K, rng.integers(1, K + 1), replace=False)
            x[idx] = rng.uniform(0.3, 1.0, len(idx))
            u = np.sqrt(x)
            A, B = rank1(x, rng.uniform(0, 2 * np.pi, (K, n)))
        else:
            u = rng.uniform(0.1, 1.2, K)
            sc = rng.uniform(0.0, 1.2) * (u ** 2).max()
            A = rng.standard_normal((K, nTj)) * sc
            B = rng.standard_normal((K, nTj)) * sc
        z = np.concatenate([u, A.ravel(), B.ravel()])
        for tau in taus:
            res = minimize(ob.neg_soft, z, args=(tau,), jac=True, method='L-BFGS-B',
                           options={'maxiter': 4000, 'ftol': 1e-16, 'gtol': 1e-12})
            z = _renorm(ob, res.x)
        res = minimize(lambda w: -ob.ratio_exactmin(w), z, method='Nelder-Mead',
                       options={'maxiter': 6000, 'maxfev': 9000,
                                'xatol': 1e-11, 'fatol': 1e-13})
        z = _renorm(ob, res.x)
        v = ob.ratio_exactmin(z)
        if v > best:
            best, bz = v, z.copy()
    return best, bz


# ------------------------------------------------------------------ drivers
def run_vtilde(args):
    n, q = args.n, args.q
    r = q / (n + 1)
    eta_grid, C_grid = cl2_grid(n, q, npts=args.npts, starts=args.cl2starts)
    print(f"Cl2_r grid ({len(eta_grid)} pts, n={n} q={q} r={r:.6f}):")
    for e, c in zip(eta_grid[::max(1, len(eta_grid) // 10)],
                    C_grid[::max(1, len(C_grid) // 10)]):
        print(f"    eta={e:7.4f}  Cl2={c:.10f}  (rel {c / C_grid[0]:.8f})")
    rows = []
    for p in args.p:
        t0 = time.time()
        G, xopt = gamma_s(p, r, starts=args.gstarts)
        target = (n + 1) ** (-(1 - r) / r) * G
        nf = (p - 1) // 2
        ks = list(range(1, nf + 1))
        ob = Obj(p, n, ks, r, (eta_grid, C_grid), target)
        V, z = maximise(ob, args.starts, seed=args.seed, x0hint=xopt)
        x, C = ob.profile(z)
        _, s, eta, cost = ob.pieces(z)
        sel = x > 1e-7 * x.max()
        # exact cost of the winning profile's own shape
        Ts = subsets(n, nonempty=True)
        exact_cost = 0.0
        ok = True
        for kk in np.nonzero(sel)[0]:
            m = np.abs(C[kk, 1:]) / x[kk]
            cp = cprof(n, r, m, starts=args.cpstarts)
            if not np.isfinite(cp):
                ok = False
                break
            exact_cost += cp * x[kk] ** r
        exact_ratio = (ob.pr.Lam(C) / exact_cost ** (1.0 / r) / target) if ok else float('nan')
        print(f"\np={p}  nf={nf}  Gamma={G:.9f}  target={target:.9f}   ({time.time()-t0:.0f}s)")
        print(f"   Vtilde/target          = {V:.9f}")
        print(f"   exact ratio at arg-max = {exact_ratio:.9f}   (cost of the profile's own shape)")
        print(f"   support k = {[ks[i] for i in np.nonzero(sel)[0]]}")
        print(f"   x   = {np.array2string(x[sel], precision=6)}")
        print(f"   eta = {np.array2string(eta[sel], precision=6)}")
        rows.append(dict(p=p, n=n, q=q, r=r, Gamma=G, target=target, Vtilde=V,
                         exact_ratio=float(exact_ratio),
                         ks=[int(ks[i]) for i in np.nonzero(sel)[0]],
                         x=x[sel].tolist(), eta=eta[sel].tolist(),
                         C=[[float(np.abs(C[i, t])) for t in range(1, C.shape[1])]
                            for i in np.nonzero(sel)[0]]))
    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print('\nwrote', args.out)


def run_curve(args):
    """R(x, eta): the exact best margin at FIXED clean amplitudes and junk level."""
    n, q = args.n, args.q
    r = q / (n + 1)
    for p in args.p:
        G, xopt = gamma_s(p, r, starts=args.gstarts)
        target = (n + 1) ** (-(1 - r) / r) * G
        nf = (p - 1) // 2
        ks = list(range(1, nf + 1))
        sel = np.nonzero(xopt > 1e-7 * xopt.max())[0]
        S = [ks[i] for i in sel]
        x = xopt[sel]
        pr = Profile(p, n, S)
        K = len(S)
        nTj = 2 ** n - 1
        clean = pr.Lam(np.concatenate([x[:, None], np.zeros((K, nTj))], 1).astype(complex))
        print(f"\np={p}  support {S}  K={K}  x={np.array2string(x, precision=5)}")
        print(f"  clean Lambda = {clean:.9f}   (target-normalised {clean/(target*np.sum(x**r)**(1/r)/ (n+1)**(-(1-r)/r) * (n+1)**(-(1-r)/r)):.6f})")
        emax = math.sqrt(2 ** n - 1)
        print(f"  {'eta':>7} {'R(x,eta)':>13} {'R/clean':>10} {'needed':>10}   "
              f"(needed = Cl2(0)/Cl2(eta))^(1/r) inverse, i.e. 1/phi)")
        eta_grid, C_grid = cl2_grid(n, q, npts=args.npts, starts=args.cl2starts)
        rng = np.random.default_rng(args.seed)
        rows = []
        for eta in np.linspace(0.0, emax, args.levels):
            best = -np.inf
            for s in range(args.starts):
                A = rng.standard_normal((K, nTj))
                B = rng.standard_normal((K, nTj))
                nrm = np.sqrt((A ** 2 + B ** 2).sum(1))[:, None]
                A, B = A / nrm * eta * x[:, None], B / nrm * eta * x[:, None]
                z = np.concatenate([A.ravel(), B.ravel()])

                def negf(w):
                    Am = w[:K * nTj].reshape(K, nTj)
                    Bm = w[K * nTj:].reshape(K, nTj)
                    nn = np.sqrt((Am ** 2 + Bm ** 2).sum(1))[:, None]
                    nn = np.maximum(nn, 1e-300)
                    Am = Am / nn * eta * x[:, None]
                    Bm = Bm / nn * eta * x[:, None]
                    C = np.concatenate([x[:, None].astype(complex),
                                        Am + 1j * Bm], 1)
                    return -pr.Lam(C)

                if eta == 0.0:
                    best = clean
                    break
                res = minimize(negf, z, method='Nelder-Mead',
                               options={'maxiter': 60000, 'xatol': 1e-11, 'fatol': 1e-13})
                res = minimize(negf, res.x, method='Powell',
                               options={'maxiter': 60000, 'xtol': 1e-12, 'ftol': 1e-14})
                best = max(best, -res.fun)
            phi = np.interp(eta, eta_grid, C_grid)
            need = (phi / C_grid[0]) ** (1.0 / r)     # = 1/phi(eta): allowed margin ratio
            print(f"  {eta:7.4f} {best:13.9f} {best/clean:10.6f} {need:10.6f}"
                  f"   {'OK' if best/clean <= need + 1e-9 else 'VIOLATES'}")
            rows.append(dict(eta=float(eta), R=float(best), rel=float(best / clean),
                             allowed=float(need)))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"curve_p{p}_n{n}_q{q}.json").write_text(
            json.dumps(dict(p=p, n=n, q=q, S=S, x=x.tolist(), clean=clean, rows=rows), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='vtilde', choices=['check', 'vtilde', 'curve'])
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--q', type=float, default=2.0)
    ap.add_argument('--p', type=int, nargs='+', default=[11, 13, 17, 19, 23])
    ap.add_argument('--starts', type=int, default=40)
    ap.add_argument('--levels', type=int, default=9)
    ap.add_argument('--npts', type=int, default=25)
    ap.add_argument('--cl2starts', type=int, default=20)
    ap.add_argument('--cpstarts', type=int, default=24)
    ap.add_argument('--gstarts', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(OUT / 'vtilde.json'))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'check':
        print("brute-force logit table  vs  profile identity   (pure networks)")
        worst = 0.0
        for p in (7, 11, 13):
            for n in (2, 3):
                if p ** n > 3000:
                    continue
                for ks in ([1], [1, 3], [2, 3, 5]):
                    if max(ks) > (p - 1) // 2:
                        continue
                    for N in (1, 2, 3):
                        m, lam, rel = check_identity(p, n, ks, N)
                        worst = max(worst, rel)
                        print(f"  p={p} n={n} ks={ks} N={N}: margin={m:.12f} "
                              f"Lambda/2^n={lam:.12f}  rel.err {rel:.2e}")
        print(f"\nworst relative error {worst:.3e}")
    elif args.mode == 'vtilde':
        run_vtilde(args)
    else:
        run_curve(args)


if __name__ == '__main__':
    main()
