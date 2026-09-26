"""gopt4_s1.py -- statement S1: checks of the new dual reformulation, of Lemma M,
and a global search in the first genuinely open case (n = 2, N = 4).

Blocks
  D   Proposition D1 (exact dual reformulation).  For fixed unit phases v_l and
      weights x_l > 0,
          min { max_l |c_l|/x_l : (KL) with A = 1 }  *  min { sum_l x_l |Lambda(v_l)| :
                                                              Lambda multi-affine, lambda_empty = 1 }  =  1.
      Hence: a modulus profile x is achievable iff sum_l x_l |Lambda(v_l)| >= |A|
      for every normalised multi-affine Lambda.  (Proved in GLOBAL_OPT4 Sec 3;
      verified here.)
  M   Lemma M.  Mahler measure of a multi-affine Lambda is >= |lambda_empty|.
  O   The averaging obstruction: for any probability measure pi on normalised
      multi-affine polynomials, sup_v E_pi |Lambda(v)|^r >= 1.
  C   n = 2, N = 4: c is the unique solution of a 4x4 Cramer system, so
      S1 there is the closed statement  sum_l |delta_l|^r >= 3^{1-r} |Delta|^r
      for the four 3x3 minors of the 3x4 matrix (a; b; ab).  Global search.
  G   Same for n = 2, N = 5, 6 and n = 3, N = 5, 6 by direct SLSQP, as a check
      that nothing dips below (n+1)^{1-r}.

All float64, local CPU.
"""
import json
import argparse
import itertools
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------- helpers
def subsets(n):
    out = []
    for k in range(n + 1):
        for T in itertools.combinations(range(n), k):
            out.append(T)
    return out


def chi(v, Ts):
    """chi[l, t] = prod_{j in Ts[t]} v[l, j]."""
    N = v.shape[0]
    X = np.ones((N, len(Ts)), dtype=complex)
    for t, T in enumerate(Ts):
        for j in T:
            X[:, t] *= v[:, j]
    return X


# ---------------------------------------------------------------- block D
def dual_min(x, X):
    """min over multi-affine Lambda with lambda_empty = 1 of sum_l x_l |Lambda(v_l)|."""
    nT = X.shape[1]

    def f(z):
        lam = np.concatenate([[1.0 + 0j], z[:nT - 1] + 1j * z[nT - 1:]])
        return float(np.sum(x * np.abs(X @ lam)))

    best = np.inf
    rng = np.random.default_rng(0)
    for s in range(12):
        z0 = np.zeros(2 * (nT - 1)) if s == 0 else rng.standard_normal(2 * (nT - 1))
        res = minimize(f, z0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-13, 'fatol': 1e-15})
        res = minimize(f, res.x, method='Powell',
                       options={'maxiter': 20000, 'xtol': 1e-14, 'ftol': 1e-16})
        best = min(best, res.fun)
    return best


def primal_min(x, X):
    """min over feasible c (X^T c = e_empty) of max_l |c_l| / x_l."""
    N = X.shape[0]
    A = X.T                      # (nT, N)
    b = np.zeros(A.shape[0], dtype=complex)
    b[0] = 1.0
    # split real/imag
    Ar = np.vstack([np.hstack([A.real, -A.imag]),
                    np.hstack([A.imag, A.real])])
    br = np.concatenate([b.real, b.imag])
    # particular solution + null space
    c0, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    if np.linalg.norm(Ar @ c0 - br) > 1e-9:
        return np.inf
    u, s, vt = np.linalg.svd(Ar)
    ns = vt[np.sum(s > 1e-10):].T          # (2N, k)
    k = ns.shape[1]

    def f(t):
        z = c0 + ns @ t if k else c0
        m = np.sqrt(z[:N] ** 2 + z[N:] ** 2)
        return float(np.max(m / x))

    if k == 0:
        return f(np.zeros(0))
    best = np.inf
    rng = np.random.default_rng(1)
    for s_ in range(8):
        t0 = np.zeros(k) if s_ == 0 else rng.standard_normal(k)
        res = minimize(f, t0, method='Nelder-Mead',
                       options={'maxiter': 40000, 'xatol': 1e-13, 'fatol': 1e-15})
        res = minimize(f, res.x, method='Powell',
                       options={'maxiter': 40000, 'xtol': 1e-14, 'ftol': 1e-16})
        best = min(best, res.fun)
    return best


def check_D(rng, trials=12):
    worst = 0.0
    rows = []
    for n in (1, 2, 3):
        Ts = subsets(n)
        for N in (n + 1, n + 2, n + 3):
            for _ in range(trials):
                v = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(N, n)))
                x = rng.uniform(0.3, 1.5, size=N)
                X = chi(v, Ts)
                d = dual_min(x, X)
                p = primal_min(x, X)
                if not np.isfinite(p):
                    continue
                prod = d * p
                worst = max(worst, abs(prod - 1.0))
                rows.append({'n': n, 'N': N, 'dual': d, 'primal': p,
                             'product': prod})
    return worst, rows


# ---------------------------------------------------------------- block M
def check_M(rng, trials=300, grid=64):
    """Mahler measure of a multi-affine Lambda >= |lambda_empty| (= 1 here)."""
    worst = np.inf
    for _ in range(trials):
        n = rng.integers(1, 4)
        Ts = subsets(n)
        lam = np.concatenate([[1.0 + 0j],
                              rng.standard_normal(len(Ts) - 1)
                              + 1j * rng.standard_normal(len(Ts) - 1)])
        lam[1:] *= rng.uniform(0.2, 3.0)
        ang = 2 * np.pi * np.arange(grid) / grid
        mesh = np.meshgrid(*([ang] * n), indexing='ij')
        vals = np.zeros(mesh[0].shape, dtype=complex)
        for t, T in enumerate(Ts):
            term = np.ones(mesh[0].shape, dtype=complex)
            for j in T:
                term = term * np.exp(1j * mesh[j])
            vals = vals + lam[t] * term
        M = np.exp(np.mean(np.log(np.abs(vals) + 1e-300)))
        worst = min(worst, M / abs(lam[0]))
    return worst


# ---------------------------------------------------------------- block O
def check_O(rng, trials=200, grid=96):
    """sup_v E_pi |Lambda(v)|^r >= 1 for random finite measures pi."""
    worst = np.inf
    for _ in range(trials):
        n = int(rng.integers(1, 4))
        r = float(rng.uniform(0.2, 1.0))
        Ts = subsets(n)
        m = int(rng.integers(1, 6))
        w = rng.dirichlet(np.ones(m))
        ang = 2 * np.pi * np.arange(grid) / grid
        mesh = np.meshgrid(*([ang] * n), indexing='ij')
        acc = np.zeros(mesh[0].shape)
        for a in range(m):
            lam = np.concatenate([[1.0 + 0j],
                                  rng.standard_normal(len(Ts) - 1)
                                  + 1j * rng.standard_normal(len(Ts) - 1)])
            lam[1:] *= rng.uniform(0.0, 2.5)
            vals = np.zeros(mesh[0].shape, dtype=complex)
            for t, T in enumerate(Ts):
                term = np.ones(mesh[0].shape, dtype=complex)
                for j in T:
                    term = term * np.exp(1j * mesh[j])
                vals = vals + lam[t] * term
            acc += w[a] * np.abs(vals) ** r
        worst = min(worst, acc.max())
    return worst


# ---------------------------------------------------------------- block C
def cramer_obj(ang, r):
    """n = 2, N = 4.  ang = 6 angles (alpha_2..alpha_4, beta_2..beta_4)."""
    al = np.concatenate([[0.0], ang[:3]])
    be = np.concatenate([[0.0], ang[3:]])
    a = np.exp(1j * al)
    b = np.exp(1j * be)
    M = np.vstack([np.ones(4), a, b, a * b])
    try:
        c = np.linalg.solve(M, np.array([1.0, 0, 0, 0], dtype=complex))
    except np.linalg.LinAlgError:
        return 1e6
    if not np.all(np.isfinite(c)):
        return 1e6
    return float(np.sum(np.abs(c) ** r))


def search_C(r, rng, starts=4000):
    best = np.inf
    bestang = None
    for _ in range(starts):
        a0 = rng.uniform(0, 2 * np.pi, size=6)
        res = minimize(cramer_obj, a0, args=(r,), method='Nelder-Mead',
                       options={'maxiter': 6000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best:
            best, bestang = res.fun, res.x
    return best, bestang


# ---------------------------------------------------------------- block G
def direct_min(n, N, r, rng, starts=200):
    """min sum |c_l|^r over the exact (KL) system, free phases."""
    Ts = subsets(n)
    nT = len(Ts)

    def split(z):
        c = z[:N] + 1j * z[N:2 * N]
        v = np.exp(1j * z[2 * N:].reshape(N, n))
        return c, v

    def f(z):
        c, _ = split(z)
        return float(np.sum((np.abs(c) ** 2 + 1e-30) ** (r / 2)))

    def con(z):
        c, v = split(z)
        X = chi(v, Ts)
        C = X.T @ c
        out = [C[0].real - 1.0, C[0].imag]
        for t in range(1, nT):
            out += [C[t].real, C[t].imag]
        return np.array(out)

    best = np.inf
    for s in range(starts):
        z0 = np.concatenate([rng.standard_normal(2 * N) / N,
                             rng.uniform(0, 2 * np.pi, size=N * n)])
        try:
            res = minimize(f, z0, constraints=[{'type': 'eq', 'fun': con}],
                           method='SLSQP', options={'maxiter': 500, 'ftol': 1e-16})
        except Exception:
            continue
        if res.success and np.max(np.abs(con(res.x))) < 1e-10:
            best = min(best, f(res.x))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=str, default='DMOCG')
    ap.add_argument('--starts', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', type=str,
                    default='heracles/FINAL_RUNS/GOPT4/s1.json')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {}

    if 'D' in args.blocks:
        worst, rows = check_D(rng)
        print(f"[D] Proposition D1: max |dual*primal - 1| = {worst:.3e} "
              f"over {len(rows)} random (n, N, x, v)")
        out['D_worst'] = worst
        out['D_rows'] = rows[:20]

    if 'M' in args.blocks:
        w = check_M(rng)
        print(f"[M] Lemma M: min over 300 random Lambda of "
              f"Mahler/|lambda_empty| = {w:.6f}   (must be >= 1)")
        out['M_min_ratio'] = w

    if 'O' in args.blocks:
        w = check_O(rng)
        print(f"[O] averaging obstruction: min over 200 random pi of "
              f"sup_v E_pi|Lambda(v)|^r = {w:.6f}   (must be >= 1)")
        out['O_min_sup'] = w

    if 'C' in args.blocks:
        for q in (2.0, 2.5):
            r = q / 3.0
            tgt = 3.0 ** (1 - r)
            best, ang = search_C(r, rng, starts=args.starts)
            print(f"[C] n=2 N=4 q={q}: global min sum|c|^r = {best:.12f}  "
                  f"target 3^(1-r) = {tgt:.12f}   ratio = {best/tgt:.12f}")
            out[f'C_q{q}'] = {'min': best, 'target': tgt, 'ratio': best / tgt,
                              'angles': ang.tolist()}

    if 'G' in args.blocks:
        for (n, q) in ((2, 2.0), (2, 2.5), (3, 2.0)):
            r = q / (n + 1)
            tgt = (n + 1) ** (1 - r)
            for N in (n + 2, n + 3, n + 4):
                v = direct_min(n, N, r, rng, starts=120)
                print(f"[G] n={n} q={q} N={N}: min = {v:.12f}  "
                      f"target = {tgt:.12f}  ratio = {v/tgt:.12f}")
                out[f'G_n{n}_q{q}_N{N}'] = {'min': v, 'target': tgt,
                                            'ratio': v / tgt}

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
