"""gopt4_dual.py -- Proposition D1, checked with convex solvers.

Proposition D1 (proved in GLOBAL_OPT4 Sec 3).  Fix N unit-modulus phase vectors
v_1..v_N in T^n and weights x_1..x_N > 0.  Write

    PRIM(x) := min { max_l |c_l| / x_l  :  sum_l c_l prod_{j in T} v_lj = [T empty] }
    DUAL(x) := min { sum_l x_l |Lambda(v_l)| : Lambda multi-affine, lambda_empty = 1 }

Then PRIM(x) * DUAL(x) = 1.  Consequently a modulus profile x is realisable by a
feasible atom vector (|c_l| <= x_l for all l) if and only if

    sum_l x_l |Lambda(v_l)|  >=  |A|   for every normalised multi-affine Lambda,

so statement S1 is EXACTLY a statement about the numbers |Lambda(v_l)|; the
phases of the c_l carry no extra information.  This is what identifies the one
lossy step in the peeling ladder of GLOBAL_OPT3 (which replaces |Lambda(v_l)| by
||Lambda||_inf).

Both minimisations are CONVEX, so they are solved here by smoothing
(|w| -> sqrt(|w|^2 + eta), max -> log-sum-exp) with eta driven to 0 and L-BFGS
restarts -- Nelder-Mead on the raw non-smooth objectives does NOT converge in
2(2^n - 1) dimensions and produced spurious values in an earlier pass.

Weak duality (PRIM * DUAL >= 1) is elementary and is also checked here as a
guard: any measured product below 1 would be a solver artefact.
"""
import json
import argparse
import itertools
import numpy as np
from scipy.optimize import minimize


def subsets(n):
    out = []
    for k in range(n + 1):
        for T in itertools.combinations(range(n), k):
            out.append(T)
    return out


def chi(v, Ts):
    N = v.shape[0]
    X = np.ones((N, len(Ts)), dtype=complex)
    for t, T in enumerate(Ts):
        for j in T:
            X[:, t] *= v[:, j]
    return X


def dual_min(x, X, restarts=6, seed=0):
    """min_{lambda_empty = 1} sum_l x_l |Lambda(v_l)|   (convex)."""
    nT = X.shape[1]
    m = nT - 1
    rng = np.random.default_rng(seed)
    Xr, Xi = X.real, X.imag

    def val(z, eta):
        lr = np.concatenate([[1.0], z[:m]])
        li = np.concatenate([[0.0], z[m:]])
        wr = Xr @ lr - Xi @ li
        wi = Xr @ li + Xi @ lr
        a = np.sqrt(wr ** 2 + wi ** 2 + eta)
        f = float(np.sum(x * a))
        d = x / a
        gr = (Xr * (d * wr)[:, None] + Xi * (d * wi)[:, None]).sum(axis=0)
        gi = (-Xi * (d * wr)[:, None] + Xr * (d * wi)[:, None]).sum(axis=0)
        return f, np.concatenate([gr[1:], gi[1:]])

    best = np.inf
    for s in range(restarts):
        z = np.zeros(2 * m) if s == 0 else 0.7 * rng.standard_normal(2 * m)
        for eta in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
            res = minimize(val, z, args=(eta,), jac=True, method='L-BFGS-B',
                           options={'maxiter': 4000, 'ftol': 1e-18,
                                    'gtol': 1e-14})
            z = res.x
        best = min(best, val(z, 0.0)[0])
    return best


def prim_min(x, X, restarts=6, seed=1):
    """min over feasible c of max_l |c_l| / x_l   (convex)."""
    N = X.shape[0]
    A = X.T
    b = np.zeros(A.shape[0], dtype=complex)
    b[0] = 1.0
    Ar = np.vstack([np.hstack([A.real, -A.imag]),
                    np.hstack([A.imag, A.real])])
    br = np.concatenate([b.real, b.imag])
    c0, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    if np.linalg.norm(Ar @ c0 - br) > 1e-9:
        return np.inf
    _, s, vt = np.linalg.svd(Ar)
    ns = vt[int(np.sum(s > 1e-10)):].T
    k = ns.shape[1]
    rng = np.random.default_rng(seed)

    def exact(t):
        z = c0 + ns @ t if k else c0
        return float(np.max(np.sqrt(z[:N] ** 2 + z[N:] ** 2) / x))

    if k == 0:
        return exact(np.zeros(0))

    def val(t, beta, eta):
        z = c0 + ns @ t
        u, w = z[:N], z[N:]
        a = np.sqrt(u ** 2 + w ** 2 + eta) / x
        mx = a.max()
        e = np.exp(beta * (a - mx))
        S = e.sum()
        f = mx + np.log(S) / beta
        da = e / S / x / np.sqrt(u ** 2 + w ** 2 + eta)
        gz = np.concatenate([da * u, da * w])
        return float(f), ns.T @ gz

    best = np.inf
    for s_ in range(restarts):
        t = np.zeros(k) if s_ == 0 else 0.5 * rng.standard_normal(k)
        for beta, eta in ((50., 1e-6), (200., 1e-8), (1e3, 1e-10),
                          (1e4, 1e-12), (1e5, 1e-14)):
            res = minimize(val, t, args=(beta, eta), jac=True,
                           method='L-BFGS-B',
                           options={'maxiter': 4000, 'ftol': 1e-18,
                                    'gtol': 1e-14})
            t = res.x
        best = min(best, exact(t))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=25)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', type=str,
                    default='FINAL_RUNS/GOPT4/dual.json')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    worst_hi, worst_lo = 0.0, np.inf
    rows = []
    for n in (1, 2, 3):
        Ts = subsets(n)
        for N in (n + 1, n + 2, n + 4, 2 ** n + 2):
            errs = []
            for _ in range(args.trials):
                v = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(N, n)))
                x = rng.uniform(0.3, 1.5, size=N)
                X = chi(v, Ts)
                p = prim_min(x, X)
                if not np.isfinite(p):
                    continue
                d = dual_min(x, X)
                prod = p * d
                errs.append(prod - 1.0)
                worst_hi = max(worst_hi, prod - 1.0)
                worst_lo = min(worst_lo, prod - 1.0)
            if errs:
                print(f"  n={n} N={N}: max(prod-1)={max(errs):.3e}  "
                      f"min(prod-1)={min(errs):.3e}  over {len(errs)} trials")
                rows.append({'n': n, 'N': N, 'max_err': max(errs),
                             'min_err': min(errs), 'trials': len(errs)})
    print(f"\n[D] Proposition D1: product PRIM*DUAL - 1 in "
          f"[{worst_lo:.3e}, {worst_hi:.3e}]  (weak duality forces >= 0)")
    with open(args.out, 'w') as fh:
        json.dump({'rows': rows, 'worst_hi': worst_hi, 'worst_lo': worst_lo},
                  fh, indent=1, default=float)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
