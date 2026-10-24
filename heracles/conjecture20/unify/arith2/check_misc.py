# Sanity checks for three small PROVED statements.
# (1) KKT identity at programme maximisers: there is a probability mu on the binding set B with
#        K_l(mu) = Gamma_s * y_l^{r-1}  for l in spt y   (y normalised sum y^r = 1).
#     Checked by an LP feasibility problem (mu >= 0 on B, sum mu = 1, equalities); reports the residual.
# (2) First-order label rows: with A = y (clean) and small interference v (phases th), the label-row cost
#        C(w) = sum_l (y_l + v_l cos(2 pi l w/p + th_l))^r  satisfies  C(w) - sum y^r = (r/Gamma_s) psi_mu(w) + O(|v|^2),
#     psi_mu(w) = sum_l v_l K_l(mu) cos(2 pi l w/p + th_l)  (the mu-averaged second difference of phi).
# (3) Ablation inequality (any network): h(theta minus a sub-network N) >= h(theta) - max_{x,d} g_N(x,d),
#     checked on random product networks at small p, n=2.
import numpy as np, itertools
from scipy.optimize import linprog
from lib import *

s = 1.5; r = 1/s
print("(1) KKT identity K_l(mu*) = Gamma y_l^{r-1}:")
for p in [31, 61, 101, 211, 401, 1009]:
    G, y = load_support(p); K = kmat(p); S = np.nonzero(y)[0]
    yn = y/np.sum(y[S]**r)**s
    cov = K @ yn; m = cov.min(); B = np.nonzero(cov <= m*(1+1e-6))[0]
    A_eq = np.vstack([K[B][:, S].T, np.ones(len(B))]); b_eq = np.concatenate([m*yn[S]**(r-1), [1.0]])
    res = linprog(np.zeros(len(B)), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    if res.status == 0:
        resid = np.abs(A_eq @ res.x - b_eq).max()
        print(f"   p={p}: |B|={len(B)} K*={len(S)} feasible, residual {resid:.2e}, Gamma(from file)={G:.5f} vs m={m:.5f}")
    else:
        # least-squares fallback with mu>=0: minimise L1 residual
        n = len(B); k = len(S)+1
        c_ = np.concatenate([np.zeros(n), np.ones(2*k)])
        Aeq = np.hstack([A_eq, np.eye(k), -np.eye(k)])
        res2 = linprog(c_, A_eq=Aeq, b_eq=b_eq, bounds=(0, None), method='highs')
        print(f"   p={p}: |B|={len(B)} K*={len(S)} exact KKT infeasible at tol; min L1 residual {res2.fun:.2e}")

print("(2) first-order label rows:")
rng = np.random.default_rng(3)
for p in [61, 211, 1009]:
    G, y = load_support(p); K = kmat(p); S = np.nonzero(y)[0]; yn = y/np.sum(y[S]**r)**s
    cov = K @ yn; m = cov.min(); B = np.nonzero(cov <= m*(1+1e-6))[0]
    A_eq = np.vstack([K[B][:, S].T, np.ones(len(B))]); b_eq = np.concatenate([m*yn[S]**(r-1), [1.0]])
    mu = linprog(np.zeros(len(B)), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs').x
    Kmu = mu @ K[B]                                    # K_l(mu) for all l (mu symmetric via kappa(d)=kappa(-d))
    F = (p-1)//2; l = np.arange(1, F+1); w = np.arange(p)
    for eps in [1e-2, 1e-3]:
        v = np.zeros(F); v[S] = eps*yn[S]*rng.uniform(0.5, 1, len(S)); th = rng.uniform(0, 2*np.pi, F)
        cosm = np.cos(2*np.pi*np.outer(l, w)/p + th[:, None])
        C = ((yn[:, None] + v[:, None]*cosm)**r)[S].sum(0) if True else None
        psi = (v[:, None]*Kmu[:, None]*cosm).sum(0)
        err = np.abs((C - np.sum(yn[S]**r)) - (r/m)*psi).max()
        print(f"   p={p} eps={eps}: max |C(w)-sum y^r - (r/Gamma) psi_mu(w)| = {err:.2e}   (scale of psi term {np.abs(r/m*psi).max():.2e})")

print("(3) ablation inequality h(theta - N) >= h(theta) - max g_N on random networks:")
worst = -1e9
for trial in range(200):
    p = int(rng.choice([5, 7, 11])); n = 2; width = int(rng.integers(3, 8))
    U = rng.normal(size=(width, n, p)); W = rng.normal(size=(width, p))
    X = list(itertools.product(range(p), repeat=n))
    def table(idx):
        T = np.zeros((len(X), p))
        for k, x in enumerate(X):
            for i in idx: T[k] += W[i]*np.prod([U[i, t, x[t]] for t in range(n)])
        return T
    allidx = list(range(width)); Nset = list(rng.choice(width, int(rng.integers(1, width)), replace=False))
    rest = [i for i in allidx if i not in Nset]
    Tf, TN, TR = table(allidx), table(Nset), table(rest)
    def gaps(T):
        G_ = []
        for k, x in enumerate(X):
            yv = sum(x) % p
            for d in range(1, p): G_.append(T[k, yv] - T[k, (yv-d) % p])
        return np.array(G_)
    gf, gN, gR = gaps(Tf), gaps(TN), gaps(TR)
    worst = max(worst, (gf.min() - gN.max()) - gR.min())
print(f"   max over 200 trials of [h(theta) - max g_N] - h(theta - N) = {worst:.3e}  (must be <= 0)")
