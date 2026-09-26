"""arch_ls.py -- the least-squares chain probe against the Christoffel function.
For a population mu (weights x_l, phases v_l), M = M_+(mu), C = M^{-1}, d = det M,
    K(v)     = m(v)^* C m(v)                      (Christoffel function of the chain space; int K dmu = D)
    Lam_LS   = e_0^* C m(v) / C_00                (LS-optimal chain polynomial with constant term 1)
Test (a) Lam_LS is a moment functional of a probability measure (LP on a torus grid);
     (b) |Lam_LS(v)| <= d^{1/D} K(v)/D  for all v on the torus (pointwise slack);
     (c) the weaker mu-integrated consequence  int |Lam_LS| dmu <= d^{1/D}   (i.e. F(mu) <= det^{1/D});
     (d) m (signed aimed output, max over amplitude phases by Nelder-Mead) vs int|Lam_LS| dmu vs det^{1/D}.
usage: arch_ls.py n ntrials seed
"""
import numpy as np, sys, json
from scipy.optimize import minimize, linprog
from itertools import combinations
from arch_core2 import chain, Mplus, aimed

n = int(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3])
D = n + 1; rng = np.random.default_rng(seed)
G = {1: 4000, 2: 120, 3: 24}[n]
th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)
V = np.exp(1j * grid); CH = chain(V)
subsets = [T for k in range(n + 1) for T in combinations(range(n), k)]
MON = np.stack([np.prod(V[:, list(T)], axis=1) if T else np.ones(len(V)) for T in subsets], 1)

def in_body_lam(lam_coeffs_chain):
    """chain coefficients (psi_k, k=0..n) with psi_0 = 1 -> omega over all subsets (chain intervals [k] only)."""
    om = np.zeros(len(subsets), dtype=complex); om[0] = 1
    for k in range(1, D):
        om[subsets.index(tuple(range(k)))] = lam_coeffs_chain[k]
    Aeq = np.concatenate([MON[:, 1:].real.T, MON[:, 1:].imag.T, np.ones((1, len(V)))], 0)
    beq = np.concatenate([om[1:].real, om[1:].imag, [1.0]])
    r = linprog(np.zeros(len(V)), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    return r.status == 0

def population(kind):
    if kind == 'block':
        N = D; x = np.ones(N) / D + 0.03 * rng.normal(size=N); v = np.exp(1j * (2 * np.pi * np.arange(N)[:, None] / D + 0.15 * rng.normal(size=(N, n))))
    elif kind == 'blockN':
        N = D + rng.integers(1, 4); x = np.concatenate([np.ones(D) / D, 0.1 * rng.uniform(size=N - D)]) + 0.03 * rng.normal(size=N)
        v = np.exp(1j * (2 * np.pi * (np.arange(N) % D)[:, None] / D + 0.25 * rng.normal(size=(N, n))))
    elif kind == 'random':
        N = rng.integers(D, 2 * D + 3); x = rng.uniform(0.05, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
    elif kind == 'collapse':
        N = D + 1; x = rng.uniform(0.2, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n))); v[1] = v[0] * np.exp(0.05j * rng.normal(size=n))
    elif kind == 'twist':
        N = D; x = np.ones(N) / D; ph = 2 * np.pi * np.arange(N)[:, None] / D * np.ones((1, n)); ph[:, 0] *= -1
        v = np.exp(1j * (ph + 0.2 * rng.normal(size=(N, n))))
    return np.abs(x), v

def best_m(x, v, nst=6):
    """max over amplitude phases of m = min Re h (Nelder-Mead); returns a LOWER bound on sup_phi m."""
    N = len(x); best = -1e9
    for s in range(nst):
        f = lambda ph: -aimed(x * np.exp(1j * ph), v, G={1: None, 2: 240, 3: 24}[n])
        ph0 = np.zeros(N) if s == 0 else rng.uniform(-1, 1, N) * (0.3 if s < 3 else 3)
        r = minimize(f, ph0, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-9, fatol=1e-13))
        best = max(best, -r.fun)
    return best

recs = []
for t in range(ntr):
    kind = ['block', 'blockN', 'random', 'collapse', 'twist'][t % 5]
    x, v = population(kind)
    M = Mplus(x, v); C = np.linalg.inv(M); d = float(np.linalg.det(M).real)
    Kv = np.einsum('pk,kl,pl->p', CH.conj(), C, CH).real
    psi = C[:, 0] / C[0, 0]                      # Lam_LS(v) = psi^* m(v)?  Use e_0^* C m(v)/C_00 = sum_k C_{0k} m_k / C_00
    lam = (CH @ C[0, :]) / C[0, 0]
    slack_b = float((d ** (1 / D) * Kv / D - np.abs(lam)).min())
    # population-side quantities
    chp = chain(v); lam_pop = (chp @ C[0, :]) / C[0, 0]
    F_upper = float((x * np.abs(lam_pop)).sum())
    admissible = in_body_lam(C[0, :] / C[0, 0])
    mstar = best_m(x, v) if t % 5 in (0, 2) else np.nan
    rec = dict(kind=kind, N=int(len(x)), det13=d ** (1 / D), slack_b=slack_b, int_lam_mu=F_upper, ratio_c=F_upper / d ** (1 / D),
               admissible=bool(admissible), m_star=float(mstar) if np.isfinite(mstar) else None,
               d0=1 / C[0, 0].real, S=float(x.sum()))
    recs.append(rec)
    print(f"[{t:2d} {kind:8s} N={len(x)}] det^(1/D)={d**(1/D):.4f} S={x.sum():.4f}  (b) min slack={slack_b:+.3e}  (c) int|Lam|dmu/det^(1/D)={rec['ratio_c']:.6f}  admissible={admissible}" +
          (f"  m*/det^(1/D)={mstar/d**(1/D):.6f}" if np.isfinite(mstar) else ''), flush=True)

sb = np.array([r['slack_b'] for r in recs]); rc = np.array([r['ratio_c'] for r in recs]); ad = np.array([r['admissible'] for r in recs])
print(f"n={n}: (b) pointwise min slack {sb.min():+.3e} [violations {int((sb < -1e-9).sum())}/{len(sb)}]; (c) max int|Lam|dmu/det^(1/D) = {rc.max():.6f}; admissible {ad.sum()}/{len(ad)}")
json.dump(dict(n=n, seed=seed, G=G, recs=recs), open(f'FINAL_RUNS/ARCH/ls_n{n}_s{seed}.json', 'w'))
