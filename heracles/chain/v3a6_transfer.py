# 3a.6 transfer lemma: (3V) by independent sampling, and the end-to-end inequality h(z) >= A'(2+A')/3 sum_k |z_k|_2.
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(4)
lhs = lambda a, A: 2 * a.sum() + sum(abs(a[i] - A * a[j]) for i in range(3) for j in range(3) if i != j)
worst = 9
for _ in range(200000):
    a = rng.dirichlet(np.ones(3) * rng.choice([0.2, 1, 5])); A = rng.uniform(0, 0.47)
    worst = min(worst, lhs(a, A) - 2 * (2 + A) * np.linalg.norm(a))
A = 0.5; a = np.array([1, A, A])
print(f"(3V) min slack on [0,0.47]: {worst:.3e} (>= 0 needed);  at A'=0.5, a=(1,.5,.5): {lhs(a, A) - 2 * (2 + A) * np.linalg.norm(a):+.4f} (fails, as stated)")

def hval(zc, S, p):
    # zc: complex array (3, K); h = max over (a,b) of Re sum_k z1 zeta^a + z2 zeta^b + z3 zeta^(a+b), zeta_k = w^{-2k}
    x = np.arange(p)
    ze = np.exp(-2j * np.pi * 2 * np.outer(S, x) / p)          # (K, p)
    f1 = np.real(zc[0] @ ze); f2 = np.real(zc[1] @ ze); f3 = np.real(zc[2] @ ze)
    F = f1[:, None] + f2[None, :] + f3[(x[:, None] + x[None, :]) % p]
    return F.max()

def ratio(v, S, p, K):
    zc = (v[:3 * K] + 1j * v[3 * K:]).reshape(3, K)
    return hval(zc, S, p) / np.linalg.norm(zc, axis=0).sum()

def astar_est(S, p, K):
    x = np.arange(p); ze = np.exp(-2j * np.pi * 2 * np.outer(S, x) / p)
    f = lambda v: np.real((v[:K] + 1j * v[K:]) @ ze).max() / np.abs(v[:K] + 1j * v[K:]).sum()
    return min(minimize(f, rng.normal(size=2 * K), method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-10, fatol=1e-12)).fun for _ in range(60))

for p, S in ((13, [1]), (13, [1, 2]), (13, [1, 3, 4]), (31, [1, 5, 7])):
    K = len(S); S = np.array(S)
    Ast = astar_est(S, p, K); Ap = min(Ast, 0.47)
    best = min(minimize(ratio, rng.normal(size=6 * K), args=(S, p, K), method='Nelder-Mead',
                        options=dict(maxiter=20000, xatol=1e-10, fatol=1e-12)).fun for _ in range(40))
    print(f"p={p} S={list(S)}: A* ~ {Ast:.4f} (search), guaranteed A_ball >= {Ap * (2 + Ap) / 3:.4f}, searched min h/sum|z_k| = {best:.4f}")
