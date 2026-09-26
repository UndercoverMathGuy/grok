"""arch_ut2.py -- (UT) with the admissibility side-condition:
    T_adm(B) := max { Re tr Y : Y UT, [[B,Y],[Y^*,B]] >= 0,  exists real diag d with sum d = Re tr Y and
                                       diag(d) + Y_up + Y_up^* >= 0 }        (Y_up = strictly upper part of Y)
The last condition makes Lambda_Y = m^*Ym/trY the moment functional of the density |psi|^2-type  m(u)^* Gamma m(u)
(Gamma PSD, interval-supported), hence admissible with all non-interval moments zero. Need T_adm(B) >= D.
Also reports plain T(B) and, for n<=3, an LP admissibility check of the plain-(UT) probe.
usage: arch_ut2.py n ntrials seed
"""
import numpy as np, sys, json, cvxpy as cp
from scipy.optimize import linprog
from itertools import combinations

n = int(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3])
D = n + 1; rng = np.random.default_rng(seed)

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
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D)

def solve(B, adm):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)]
    cons += [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0]
    if adm:
        d = cp.Variable(D)
        Yup = Y - cp.diag(cp.diag(Y))
        cons += [cp.diag(d) + Yup + cp.conj(Yup.T) >> 0, cp.sum(d) == cp.real(cp.trace(Y))]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Y))), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value, Y.value

if n <= 3:
    G = {1: 2000, 2: 72, 3: 20}[n]
    th = 2 * np.pi * np.arange(G) / G
    grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)
    subsets = [T for k in range(1, n + 1) for T in combinations(range(n), k)]
    MON = np.stack([np.exp(1j * grid[:, list(T)].sum(1)) for T in subsets], 1)
    def admissible(om):
        target = np.array([om.get(T, 0) for T in subsets])
        Aeq = np.concatenate([MON.real.T, MON.imag.T, np.ones((1, len(grid)))], 0)
        beq = np.concatenate([target.real, target.imag, [1.0]])
        r = linprog(np.zeros(len(grid)), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
        return r.status == 0

recs = []
for t in range(ntr):
    kind = ['wishart', 'ill', 'chain', 'nearblock', 'collapse'][t % 5]
    B = randB(kind)
    T0, Y0 = solve(B, False)
    T1, Y1 = solve(B, True)
    adm0 = None
    if n <= 3:
        om = {}
        for j in range(D):
            for k in range(j + 1, D):
                om[tuple(range(j, k))] = Y0[j, k] / np.trace(Y0)
        adm0 = admissible(om)
    recs.append(dict(kind=kind, T=float(T0), T_adm=float(T1), cond=float(np.linalg.cond(B)), plain_probe_admissible=adm0))
    print(f"[{t:2d} {kind:9s} cond={np.linalg.cond(B):9.2f}]  T(B)={T0:.6f}  T_adm(B)={T1:.6f}  (need >= {D})  {'OK' if T1 >= D - 1e-6 else 'FAIL'}   plain probe admissible(LP)={adm0}", flush=True)
Ta = np.array([r['T_adm'] for r in recs])
print(f"n={n}: min T_adm(B) = {Ta.min():.6f} vs D={D}; fails {int((Ta < D - 1e-6).sum())}/{ntr}")
json.dump(dict(n=n, seed=seed, recs=recs), open(f'FINAL_RUNS/ARCH/ut2_n{n}_s{seed}.json', 'w'))
