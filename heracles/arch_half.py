"""arch_half.py -- the 'half' criterion:  min_{v in T^n} Re Lambda_Y(v) >= 1/2  ==>  rho = 2 Re Lambda_Y(conj u) - 1 is a
probability density with Lambda_rho = Lambda_Y  (admissibility for free, every n).
Tests: (a) the (UT)-optimal Y*;  (b) extreme feasible Y with Re trY >= D;  (c) the optimal Y* for near-block / ill B.
Reports min_v Re Lambda and also min_v [Re Lambda - |Lambda - 1|]-type quantities, and the n=1 closed form.
usage: arch_half.py n ntrials seed
"""
import numpy as np, sys, json, cvxpy as cp
from scipy.optimize import minimize

n = int(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3])
D = n + 1; rng = np.random.default_rng(seed)
G = {1: 4000, 2: 160, 3: 40, 4: 16}[n]
th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)
V = np.exp(1j * grid); CH = np.concatenate([np.ones((len(V), 1)), np.cumprod(V, 1)], 1)

def randB(kind):
    if kind == 'wishart':
        A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); B = A @ A.conj().T + 1e-2 * np.eye(D)
    elif kind == 'ill':
        U, _ = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))); lam = np.exp(rng.uniform(-3, 3, D)); B = (U * lam) @ U.conj().T
    elif kind == 'chain':
        N = rng.integers(D, 2 * D + 2); x = rng.uniform(0.1, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-8 * np.eye(D))
    elif kind == 'nearblock':
        N = D; x = np.ones(N) / D + 0.05 * rng.normal(size=N); v = np.exp(1j * (2 * np.pi * np.arange(N)[:, None] / D + 0.3 * rng.normal(size=(N, n))))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * np.abs(x)) @ ch.conj(); B = np.linalg.inv(M)
    elif kind == 'collapse':
        N = D + 1; x = rng.uniform(0.2, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n))); v[1] = v[0] * np.exp(0.03j * rng.normal(size=n))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-8 * np.eye(D))
    elif kind == 'corr1':   # n=1 style strongly correlated, general D: B = I + rho*(J - I) scaled
        r = rng.uniform(0.5, 0.98); B = (1 - r) * np.eye(D) + r * np.ones((D, D)); ph = np.exp(1j * rng.uniform(0, 2 * np.pi, D)); B = (ph[:, None] * B) * ph.conj()[None, :]
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D)

def solve(B, Xi=None):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)] + [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0]
    if Xi is None: obj = cp.real(cp.trace(Y))
    else: cons += [cp.real(cp.trace(Y)) >= D]; obj = cp.real(cp.trace(Xi @ Y))
    cp.Problem(cp.Maximize(obj), cons).solve(solver=cp.CLARABEL)
    return Y.value

def min_re(Y):
    t = np.trace(Y); Y = Y / t
    lam = np.einsum('pk,kl,pl->p', CH.conj(), Y, CH)
    i = int(lam.real.argmin())
    def f(a):
        ch = np.concatenate([[1], np.cumprod(np.exp(1j * a))]); return float((ch.conj() @ Y @ ch).real)
    r = minimize(f, grid[i], method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-14))
    return min(float(lam.real[i]), r.fun), float(np.abs(lam).max())

worst_opt = 1e9; worst_ext = 1e9; recs = []
kinds = ['wishart', 'ill', 'chain', 'nearblock', 'collapse', 'corr1']
for t in range(ntr):
    kind = kinds[t % len(kinds)]; B = randB(kind)
    Y = solve(B); mo, mx = min_re(Y); worst_opt = min(worst_opt, mo)
    me = 1e9
    for s in range(3):
        Ye = solve(B, rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))); m2, _ = min_re(Ye); me = min(me, m2)
    worst_ext = min(worst_ext, me)
    recs.append(dict(kind=kind, minRe_opt=mo, minRe_ext=me, cond=float(np.linalg.cond(B))))
    print(f"[{t:2d} {kind:9s} cond={np.linalg.cond(B):8.2f}] optimal Y*: min Re Lambda = {mo:.4f} (max|Lambda| {mx:.3f});  extreme Y: min Re Lambda = {me:.4f}", flush=True)
print(f"n={n}: min over trials of min_v Re Lambda:  optimal Y* {worst_opt:.5f};  extreme feasible Y {worst_ext:.5f}   (criterion needs >= 0.5)")
json.dump(dict(n=n, seed=seed, worst_opt=worst_opt, worst_ext=worst_ext, recs=recs), open(f'FINAL_RUNS/ARCH/half_n{n}_s{seed}.json', 'w'))
