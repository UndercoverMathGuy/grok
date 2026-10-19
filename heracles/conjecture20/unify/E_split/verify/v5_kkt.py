"""Referee check of E6 (cost shares = margin shares at rescaling-stationary maximisers) and the share floor.
(1) IF maximisers built from y* at (5,2,b=2.5), (7,2,b=2.2): LP for q on the argmin set.
(2) p=3 certified counterexample networks (n=11, 13; b=2): LP on the near-argmin set (they are numerical optima).
(3) A deliberately non-stationary network (one block rescaled): LP must fail.
(4) Share floor w_i >= (nu^{nu/2} gamma/2)^{1/(s-1)}."""
import math, importlib.util
import numpy as np
from scipy.optimize import linprog
from vlib import *
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def kkt_lp(Gi, h, w):
    """Gi: (num_active, m) per-neuron gaps on active set. min eps s.t. |sum_a q_a Gi[a]/h - w| <= eps, q in simplex."""
    A_n, m = Gi.shape
    c = np.zeros(A_n + 1); c[-1] = 1
    Aub = []; bub = []
    for i in range(m):
        row = np.concatenate([Gi[:, i] / h, [-1]]); Aub.append(row); bub.append(w[i])
        row = np.concatenate([-Gi[:, i] / h, [-1]]); Aub.append(row); bub.append(-w[i])
    Aeq = [np.concatenate([np.ones(A_n), [0]])]
    r = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=[1],
                bounds=[(0, None)] * A_n + [(0, None)], method='highs')
    return r.fun


def Gamma_s(p, s, iters=4000):
    """programme value by projected ascent on a smooth surrogate + exhaustive support LP polish (small p)."""
    import itertools
    from scipy.optimize import minimize
    F = F_of(p); best = (-1, None)
    for k in range(1, len(F) + 1):
        for S in itertools.combinations(F, k):
            def neg(v):
                y = np.abs(v); y = y / (y ** (1 / s)).sum() ** s
                return -min(sum(y[q] * kappa(j, d, p) for q, j in enumerate(S)) for d in range(1, p))
            for st in range(6):
                r = minimize(neg, np.random.default_rng(st).uniform(0.2, 1, k), method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 20000})
                if -r.fun > best[0]:
                    y = np.abs(r.x); y = y / (y ** (1 / s)).sum() ** s
                    best = (-r.fun, dict(zip(S, y)))
    return best


for (p, n, b) in [(5, 2, 2.5), (7, 2, 2.2), (5, 3, 3.2)]:
    nu = n + 1; s = nu / b
    G, y = Gamma_s(p, s)
    Us, Ws = [], []
    for j, yj in y.items():
        if yj > 1e-9:
            U, W = block(p, n, j, 1.0, 0.123 * j)
            beta, _, _ = atoms(U, W); lam = (yj / abs(beta[j])) ** (1 / nu)
            Us.append(U * lam); Ws.append(W * lam)
    U = np.concatenate(Us); W = np.concatenate(Ws)
    f = table(U, W); Gm = gaps(f, p, n); h = Gm.min()
    c, w = shares(U, W, s)
    gam = h / c.sum() ** s
    gIF = nu ** (1 - s) * Bp(p, n) * G
    act = np.argwhere(Gm <= h + 1e-10 * abs(h))
    Gi = np.array([[gaps(table(U[i:i + 1], W[i:i + 1]), p, n)[a0, a1] for i in range(len(W))] for (a0, a1) in act])
    eps = kkt_lp(Gi, h, w)
    say(f"(p,n,b)=({p},{n},{b}) IF network: gamma/gamma_IF = {gam/gIF:.8f}, |A| = {len(act)}, KKT LP residual = {eps:.2e}")
    # deliberately non-stationary: rescale the first block by 1.3
    U2 = U.copy(); W2 = W.copy(); U2[:nu] *= 1.3; W2[:nu] *= 1.3
    Gm2 = gaps(table(U2, W2), p, n); h2 = Gm2.min(); c2, w2 = shares(U2, W2, s)
    act2 = np.argwhere(Gm2 <= h2 + 1e-10 * abs(h2))
    Gi2 = np.array([[gaps(table(U2[i:i + 1], W2[i:i + 1]), p, n)[a0, a1] for i in range(len(W2))] for (a0, a1) in act2])
    say(f"   same network with block 1 rescaled x1.3 (not stationary): KKT LP residual = {kkt_lp(Gi2, h2, w2):.2e}")
    floor = (nu ** (nu / 2) * gam / 2) ** (1 / (s - 1))
    say(f"   share floor check: min w = {w.min():.4f} >= floor {floor:.2e}: {w.min() >= floor}")

say("== p=3 certified networks (b=2): KKT on near-argmin set ==")
spec = importlib.util.spec_from_file_location("excert", "/home/user/grok/heracles/conjecture20/exact_certificate.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
for n, net in ex.NETWORKS.items():
    Uf = np.array([[float(a) for a in r] for r in net["u"]]); Wf = np.array([[float(a) for a in r] for r in net["w"]])
    m = len(Uf); nu = n + 1; s = nu / 2
    rows = []
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; yv = (N1 + 2 * N2) % 3
            z = Uf[:, 0] ** N0 * Uf[:, 1] ** N1 * Uf[:, 2] ** N2
            for cc in range(3):
                if cc != yv:
                    rows.append((Wf[:, yv] - Wf[:, cc]) * z)
    Gi_all = np.array(rows); g = Gi_all.sum(1); h = g.min()
    sq = n * (Uf ** 2).sum(1) + (Wf ** 2).sum(1)
    w = sq / sq.sum()  # b=2: c_i = ||omega_i||^2
    gam = h / sq.sum() ** s
    for tol in [1e-6, 1e-4, 1e-3]:
        act = np.where(g <= h * (1 + tol))[0]
        eps = kkt_lp(Gi_all[act], h, w)
        say(f"  n={n}: tol {tol:.0e}: |A_tol| = {len(act)}, KKT LP residual max_i|E_q g_i/h - w_i| = {eps:.2e}  (w_i range {w.min():.4f}-{w.max():.4f})")
    floor = (nu ** (nu / 2) * gam / 2) ** (1 / (s - 1))
    say(f"  n={n}: share floor (if it were a maximiser) {floor:.3e} <= min w {w.min():.4f}: {w.min() >= floor}")
open(__file__.replace('.py', '.out'), 'w').write("\n".join(out) + "\n")
