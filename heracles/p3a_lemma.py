"""p3a_lemma.py -- the junk lemma P3 needs, reduced to ONE constant, and that constant measured.

THE REDUCTION (derived in P3A.md sec 2; everything here is exact bookkeeping).

For a pure gated-chain network with profile (x_k, C_T^{(k)}) write, for a wrong-class
offset D != 0,

    Lambda_D(a) = sum_k Re[ (1 - e^{2 pi i k D/p}) h_k(a) ],   h_k(a) = sum_T C_T^{(k)} omega^{-2 k sigma_T(a)},
    mean_a Lambda_D = sum_k w_kD x_k ,       w_kD = 1 - cos(2 pi k D/p)   (clean amplitudes taken real >= 0),
    Lambda = min over a and D of Lambda_D(a)      ( = 2^n * margin ).

Because a test density is just a convex combination of rows, the BEST bound any test
density can give at offset D is  mean_a Lambda_D - (mean_a Lambda_D - min_a Lambda_D),
i.e. the margin penalty that any junk lemma can hope to extract is exactly

    Pen(D) = mean_a Lambda_D - min_a Lambda_D  =  max_a [ - psi_D(a) ],
    psi_D(a) := sum_k sum_{T != empty} Re[ (1 - e^{i theta_kD}) C_T^{(k)} omega^{-2 k sigma_T(a)} ].

Feeding the cost relief  sum_{k,l}|c_lk|^r >= sum_k Cl2_r(eta_k) x_k^r  through the
programme Gamma (see P3A.md sec 2), the target (T) FOLLOWS as soon as, for every D,

    Pen(D)  >=  sum_k w_kD x_k ( 1 - phi(eta_k)^{-1} ),     phi(eta) := (Cl2_r(0)/Cl2_r(eta))^{1/r}.

So a junk lemma of the shape  Pen(D) >= rho * sum_k w_kD * eta_k x_k  closes P3 for EVERY
junk level and every K as soon as

    rho  >=  rho_need(n,q) := max over 0 < eta <= sqrt(2^n - 1) of [ 1 - phi(eta)^{-1} ] / eta .

Block 'need' below computes rho_need from the measured Cl2_r curve.

Block 'rho' measures the largest rho for which such a lemma is TRUE.  Since
|1 - e^{i theta}| = sqrt(2 w) and w <= 2, it is enough (and is what we measure) to have

    (AC)   max_{a in Z_p^n} [ - Re sum_{k,T != empty} z_kT omega^{-2 k sigma_T(a)} ]
           >=  rho * sum_k ( sum_{T != empty} |z_kT|^2 )^{1/2}

for all complex coefficients z, i.e. the moment body
Omega = conv{ (omega^{-2 k sigma_T(a)})_{k,T} : a in Z_p^n }  contains the ball of radius
rho of the mixed norm "max over frequencies of the l_2 norm over sign patterns".
Plain reading: (AC) says a junk-carrying network always has SOME row on which the gap
dips below its level-set average by at least rho times the TOTAL (summed over
frequencies) junk amplitude -- no dilution by the number of frequencies K, which is
exactly what Lemma Q cannot give.

Block 'riesz' certifies (AC) constructively where it can: if the mode set is dissociate,
a Riesz product is a non-negative density with the prescribed moments of modulus 1/2.

Usage
    uv run python p3a_lemma.py --block need  --n 2 --q 2
    uv run python p3a_lemma.py --block rho   --p 11 13 17 19 23 --starts 400
    uv run python p3a_lemma.py --block riesz --p 11 13 17 19 23

All float64, local CPU, no pods.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3a_prof import OUT, cl2_grid, gamma_s, subsets  # noqa: E402


# ------------------------------------------------------------------ rho_need
def rho_need(n, q, npts=41, cl2starts=24, verbose=True):
    r = q / (n + 1)
    eta, C = cl2_grid(n, q, npts=npts, starts=cl2starts)
    phi = (C[0] / C) ** (1.0 / r)
    need = np.zeros_like(eta)
    need[1:] = (1.0 - 1.0 / phi[1:]) / eta[1:]
    i = int(np.argmax(need))
    if verbose:
        print(f"  n={n} q={q} r={r:.6f}  Cl2_r(0)={C[0]:.12f}  "
              f"eta_max={eta[-1]:.9f}")
        print(f"  {'eta':>9} {'Cl2':>15} {'phi':>12} {'(1-1/phi)/eta':>15}")
        for j in range(0, len(eta), max(1, len(eta) // 12)):
            print(f"  {eta[j]:9.5f} {C[j]:15.10f} {phi[j]:12.8f} {need[j]:15.9f}")
        print(f"  {eta[-1]:9.5f} {C[-1]:15.10f} {phi[-1]:12.8f} {need[-1]:15.9f}")
        print(f"  --> rho_need = {need[i]:.9f}   at eta = {eta[i]:.6f}"
              f"   (endpoint value {need[-1]:.9f})")
    return float(need[i]), float(eta[i]), eta, C, need


# ------------------------------------------------------------------ the mode set
def modes(p, n, ks):
    """m[k, t] in Z_p^n, t indexing the NON-EMPTY subsets T, m = 2k on T and 0 off T."""
    Ts = subsets(n, nonempty=True)
    M = np.zeros((len(ks), len(Ts), n), dtype=np.int64)
    for i, k in enumerate(ks):
        for t, T in enumerate(Ts):
            for j in T:
                M[i, t, j] = (2 * k) % p
    return M, Ts


def chi_table(p, n, ks):
    """X[k, t, a] = omega^{- m_{k,t} . a}, a running over Z_p^n."""
    M, Ts = modes(p, n, ks)
    r = np.arange(p ** n)
    ops = np.stack([(r // p ** j) % p for j in range(n)], 1)
    ph = np.einsum('ktj,aj->kta', M, ops) % p
    return np.exp(-2j * np.pi * ph / p), Ts


# ------------------------------------------------------------------ rho (AC)
def rho_of(p, n, ks, starts=300, seed=0, verbose=False):
    """min over z of  max_a[-Re<z, chi(a)>] / sum_k ||z_k||_2 ."""
    X, Ts = chi_table(p, n, ks)                       # (K, nT, p^n)
    K, nT, P = X.shape
    Xr, Xi = X.real, X.imag
    rng = np.random.default_rng(seed)

    def val(w):
        A = w[:K * nT].reshape(K, nT)
        B = w[K * nT:].reshape(K, nT)
        # -Re<z, chi> = -(A*Xr - B*Xi) summed
        f = -(np.einsum('kt,kta->a', A, Xr) - np.einsum('kt,kta->a', B, Xi))
        den = np.sqrt((A ** 2 + B ** 2).sum(1)).sum()
        if den <= 1e-300:
            return 1e9
        return float(f.max() / den)

    def soft(w, tau):
        A = w[:K * nT].reshape(K, nT)
        B = w[K * nT:].reshape(K, nT)
        f = -(np.einsum('kt,kta->a', A, Xr) - np.einsum('kt,kta->a', B, Xi))
        mx = f.max()
        e = np.exp((f - mx) / tau)
        Z = e.sum()
        fs = mx + tau * np.log(Z)
        Om = e / Z
        nk = np.sqrt((A ** 2 + B ** 2).sum(1))
        den = nk.sum()
        if den <= 1e-12:
            return 1e9, np.zeros_like(w)
        dA = -np.einsum('a,kta->kt', Om, Xr)
        dB = np.einsum('a,kta->kt', Om, Xi)
        g_num = np.concatenate([dA.ravel(), dB.ravel()])
        sc = np.where(nk > 1e-12, 1.0 / np.maximum(nk, 1e-12), 0.0)
        dnA = sc[:, None] * A
        dnB = sc[:, None] * B
        g_den = np.concatenate([dnA.ravel(), dnB.ravel()])
        v = fs / den
        return v, (g_num - v * g_den) / den

    best, bw = np.inf, None
    for s in range(starts):
        if s < nT:                                     # all mass on one sign pattern
            A = np.zeros((K, nT)); B = np.zeros((K, nT))
            A[:, s] = 1.0
            B[:, s] = rng.standard_normal(K) * 0.3
        elif s < nT + 6:
            A = np.zeros((K, nT)); B = np.zeros((K, nT))
            A[:, nT - 1] = rng.standard_normal(K)
            B[:, nT - 1] = rng.standard_normal(K)
        else:
            A = rng.standard_normal((K, nT))
            B = rng.standard_normal((K, nT))
        w = np.concatenate([A.ravel(), B.ravel()])
        for tau in (0.3, 0.1, 0.03, 0.01, 0.003, 1e-3, 3e-4):
            res = minimize(soft, w, args=(tau,), jac=True, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12})
            w = res.x
            nn = np.linalg.norm(w)
            if nn > 0:
                w = w / nn
        res = minimize(val, w, method='Nelder-Mead',
                       options={'maxiter': 60000, 'xatol': 1e-12, 'fatol': 1e-14})
        v = val(res.x)
        if v < best:
            best, bw = v, res.x.copy()
    A = bw[:K * nT].reshape(K, nT)
    B = bw[K * nT:].reshape(K, nT)
    Z = A + 1j * B
    nk = np.abs(Z).sum() and np.sqrt((A ** 2 + B ** 2).sum(1))
    if verbose:
        shape = np.abs(Z) / max(1e-300, np.abs(Z).max())
        print("    worst-case junk shape |z_kT| (rows = frequencies, cols = "
              + ",".join("".join(str(j + 1) for j in T) for T in Ts) + "):")
        for i in range(K):
            print("      k=%2d  " % ks[i] + " ".join(f"{v:.4f}" for v in shape[i]))
    return best, Z


# ------------------------------------------------------------------ dissociativity / Riesz
def dissociate(mode_list, p, n):
    """True if all 3^M sums sum_i eps_i m_i (eps in {0,+-1}) are DISTINCT mod p."""
    M = len(mode_list)
    if M > 12:
        return None
    seen = set()
    for eps in itertools.product((0, 1, -1), repeat=M):
        s = tuple(int(sum(e * m[j] for e, m in zip(eps, mode_list)) % p) for j in range(n))
        if s in seen:
            return False
        seen.add(s)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', default='need', choices=['need', 'rho', 'riesz'])
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--q', type=float, default=2.0)
    ap.add_argument('--p', type=int, nargs='+', default=[11, 13, 17, 19, 23])
    ap.add_argument('--starts', type=int, default=250)
    ap.add_argument('--npts', type=int, default=41)
    ap.add_argument('--cl2starts', type=int, default=24)
    ap.add_argument('--gstarts', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.block == 'need':
        res = {}
        for (n, q) in [(2, 2.0), (2, 2.5), (3, 2.0), (3, 3.0), (4, 2.0)]:
            if args.n and args.q and (n, q) not in [(args.n, args.q)] and args.p == [11, 13, 17, 19, 23]:
                pass
            print(f"\n== rho_need, n={n} q={q}")
            try:
                v, e, eta, C, need = rho_need(n, q, npts=args.npts,
                                              cl2starts=args.cl2starts)
            except Exception as ex:
                print("   failed:", ex)
                continue
            res[f"{n}:{q}"] = dict(rho_need=v, at_eta=e,
                                   eta=eta.tolist(), Cl2=C.tolist(),
                                   curve=need.tolist())
        (OUT / 'rho_need.json').write_text(json.dumps(res, indent=1))
        print('\nwrote', OUT / 'rho_need.json')

    elif args.block == 'rho':
        need, _, _, _, _ = rho_need(args.n, args.q, npts=args.npts,
                                    cl2starts=args.cl2starts, verbose=False)
        r = args.q / (args.n + 1)
        print(f"rho_need(n={args.n}, q={args.q}) = {need:.9f}\n")
        rows = []
        for p in args.p:
            G, xopt = gamma_s(p, r, starts=args.gstarts)
            nf = (p - 1) // 2
            allk = list(range(1, nf + 1))
            S = [allk[i] for i in np.nonzero(xopt > 1e-7 * xopt.max())[0]]
            for tag, ks in (('Gamma-optimal support', S),
                            ('first 2 freqs', allk[:2]),
                            ('all freqs', allk)):
                if len(ks) == 0:
                    continue
                if p ** args.n > 4000 and tag == 'all freqs':
                    continue
                v, Z = rho_of(p, args.n, ks, starts=args.starts, seed=args.seed,
                              verbose=(tag == 'Gamma-optimal support'))
                flag = 'OK  ' if v >= need else 'FAIL'
                print(f"  p={p:3d} K={len(ks):2d} {tag:<22s} rho = {v:.9f}  "
                      f"{flag} (need {need:.6f})")
                rows.append(dict(p=p, n=args.n, q=args.q, tag=tag, ks=ks,
                                 rho=v, need=need))
        (OUT / f'rho_n{args.n}_q{args.q}.json').write_text(json.dumps(rows, indent=1))
        print('\nwrote', OUT / f'rho_n{args.n}_q{args.q}.json')

    else:
        r = args.q / (args.n + 1)
        for p in args.p:
            G, xopt = gamma_s(p, r, starts=args.gstarts)
            nf = (p - 1) // 2
            allk = list(range(1, nf + 1))
            S = [allk[i] for i in np.nonzero(xopt > 1e-7 * xopt.max())[0]]
            M, Ts = modes(p, args.n, S)
            flat = [M[i, t] for i in range(len(S)) for t in range(len(Ts))]
            diag = [M[i, len(Ts) - 1] for i in range(len(S))]
            single = [M[i, 0] for i in range(len(S))]
            print(f"p={p:3d} support {S}: "
                  f"full junk-mode set dissociate = {dissociate(flat, p, args.n)}, "
                  f"diagonal modes {{2k(1..1)}} dissociate = {dissociate(diag, p, args.n)}, "
                  f"slot-1 modes dissociate = {dissociate(single, p, args.n)}")


if __name__ == '__main__':
    main()
