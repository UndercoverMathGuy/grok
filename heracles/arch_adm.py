"""arch_adm.py -- is admissibility of Lambda_Y = m^*Ym/trY special to the (UT)-optimal Y?
For random B (det 1) at n=2,3:
  (a) optimal Y* (SDP)  -> check Y* = B^{1/2} U B^{1/2} with U unitary (polar structure), LP admissibility, and
      the 'square-density' sufficient condition;
  (b) random feasible UT Y (random UT matrix scaled to the boundary of [[B,Y],[Y^*,B]] >= 0) -> LP admissibility;
  (c) random UT Y with W = B^{-1/2} Y B^{-1/2} unitary?  (constructed: take U unitary, Y = B^{1/2} U B^{1/2}; UT only if
      U is special, so instead sample U = polar(B^{1/2} K B^{1/2}) for random unit-LT K and test UT-ness/admissibility)
usage: arch_adm.py n ntrials seed
"""
import numpy as np, sys, json, cvxpy as cp
from scipy.optimize import linprog
from scipy.linalg import sqrtm, polar
from itertools import combinations

n = int(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3])
D = n + 1; rng = np.random.default_rng(seed)
G = {2: 72, 3: 20}[n]
th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)
subsets = [T for k in range(1, n + 1) for T in combinations(range(n), k)]
MON = np.stack([np.exp(1j * grid[:, list(T)].sum(1)) for T in subsets], 1)
Aeq = np.concatenate([MON.real.T, MON.imag.T, np.ones((1, len(grid)))], 0)

def admissible_Y(Y):
    om = {}
    for j in range(D):
        for k in range(j + 1, D):
            om[tuple(range(j, k))] = Y[j, k] / np.trace(Y)
    target = np.array([om.get(T, 0) for T in subsets])
    beq = np.concatenate([target.real, target.imag, [1.0]])
    r = linprog(np.zeros(len(grid)), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    return r.status == 0

def randB():
    kind = rng.choice(['wishart', 'chain', 'ill'])
    if kind == 'wishart':
        A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); B = A @ A.conj().T + 1e-2 * np.eye(D)
    elif kind == 'ill':
        U, _ = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))); lam = np.exp(rng.uniform(-2, 2, D)); B = (U * lam) @ U.conj().T
    else:
        N = rng.integers(D, 2 * D + 2); x = rng.uniform(0.1, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-8 * np.eye(D))
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D), kind

def solve(B):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)] + [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Y))), cons); prob.solve(solver=cp.CLARABEL)
    return prob.value, Y.value

stats = dict(opt_adm=0, opt_unitary=0, rand_adm=0, rand_tot=0, polarK_ut=0, polarK_adm=0, polarK_tot=0)
for t in range(ntr):
    B, kind = randB(); Bh = sqrtm(B); Bhi = np.linalg.inv(Bh)
    val, Y = solve(B)
    W = Bhi @ Y @ Bhi
    unit_err = np.linalg.norm(W.conj().T @ W - np.eye(D))
    adm = admissible_Y(Y)
    stats['opt_adm'] += adm; stats['opt_unitary'] += (unit_err < 1e-3)
    # (b) random feasible UT Y
    for s in range(4):
        R = np.triu(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) + 2 * np.eye(D)
        R = R * np.exp(-1j * np.angle(np.trace(R)))            # trace real positive
        Wr = Bhi @ R @ Bhi; R = R / np.linalg.norm(Wr, 2)        # on the boundary
        stats['rand_tot'] += 1; stats['rand_adm'] += admissible_Y(R)
    # (c) polar of B^{1/2} K B^{1/2} for random unit-LT K
    for s in range(4):
        K = np.eye(D) + np.tril(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)), -1) * 0.7
        U, _ = polar(Bh @ K @ Bh); Yk = Bh @ U @ Bh
        ut = np.linalg.norm(np.tril(Yk, -1)) / np.linalg.norm(Yk) < 1e-6
        stats['polarK_tot'] += 1; stats['polarK_ut'] += ut
        if ut: stats['polarK_adm'] += admissible_Y(Yk)
    print(f"[{t:2d} {kind:8s}] T(B)={val:.4f}  W unitary err={unit_err:.2e}  Y* admissible={adm}", flush=True)
print(f"n={n}: optimal Y*: admissible {stats['opt_adm']}/{ntr}, W unitary {stats['opt_unitary']}/{ntr};  random feasible UT Y: admissible {stats['rand_adm']}/{stats['rand_tot']};  polar(B^1/2 K B^1/2) UT in {stats['polarK_ut']}/{stats['polarK_tot']}")
json.dump(dict(n=n, seed=seed, stats={k: int(v) for k, v in stats.items()}), open(f'FINAL_RUNS/ARCH/adm_n{n}_s{seed}.json', 'w'))
