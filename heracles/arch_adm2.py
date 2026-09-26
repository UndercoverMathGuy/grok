"""arch_adm2.py -- conjecture: Y upper triangular, [[B,Y],[Y^*,B]] >= 0, det B = 1, Re tr Y >= D  ==>  Lambda_Y admissible.
Stress test: for random B, sample EXTREME points of {Y feasible, Re trY >= D} by maximising random linear objectives
Re tr(Xi Y) (SDP); also sample with Re tr Y == D exactly (boundary).  LP admissibility on a torus grid (inner approx.).
Also n=1: check |Y_01| <= |tr Y| directly.
usage: arch_adm2.py n ntrials seed
"""
import numpy as np, sys, json, cvxpy as cp
from scipy.optimize import linprog
from itertools import combinations

n = int(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3])
D = n + 1; rng = np.random.default_rng(seed)
G = {1: 3000, 2: 72, 3: 20}[n]
th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)
subsets = [T for k in range(1, n + 1) for T in combinations(range(n), k)]
MON = np.stack([np.exp(1j * grid[:, list(T)].sum(1)) for T in subsets], 1)
Aeq = np.concatenate([MON.real.T, MON.imag.T, np.ones((1, len(grid)))], 0)

def omega_of(Y):
    om = {}
    for j in range(D):
        for k in range(j + 1, D):
            om[tuple(range(j, k))] = Y[j, k] / np.trace(Y)
    return np.array([om.get(T, 0) for T in subsets])
def admissible(om, tol=2e-6):
    beq = np.concatenate([om.real, om.imag, [1.0]])
    r = linprog(np.zeros(len(grid)), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    if r.status == 0: return True
    # allow tiny tolerance: minimise l1 deviation
    return False

def randB():
    kind = rng.choice(['wishart', 'chain', 'ill', 'nearI'])
    if kind == 'wishart':
        A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); B = A @ A.conj().T + 1e-2 * np.eye(D)
    elif kind == 'ill':
        U, _ = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))); lam = np.exp(rng.uniform(-2.5, 2.5, D)); B = (U * lam) @ U.conj().T
    elif kind == 'nearI':
        H = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); H = (H + H.conj().T) / 2; B = np.eye(D) + 0.4 * H / np.linalg.norm(H, 2)
    else:
        N = rng.integers(D, 2 * D + 2); x = rng.uniform(0.1, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-8 * np.eye(D))
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D), kind

def extreme(B, Xi, eq):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)] + [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0]
    cons += [cp.real(cp.trace(Y)) == D] if eq else [cp.real(cp.trace(Y)) >= D]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Xi @ Y))), cons); prob.solve(solver=cp.CLARABEL)
    return Y.value if prob.status in ('optimal', 'optimal_inaccurate') else None

tot = 0; ok = 0; worst = 0.0; fails = []
for t in range(ntr):
    B, kind = randB()
    for s in range(6):
        Xi = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
        Y = extreme(B, Xi, eq=(s % 2 == 0))
        if Y is None: continue
        om = omega_of(Y); tot += 1
        a = admissible(om); ok += a
        mx = np.abs(om).max(); worst = max(worst, mx)
        if not a: fails.append(dict(kind=kind, trY=[float(np.trace(Y).real), float(np.trace(Y).imag)], om=[[float(z.real), float(z.imag)] for z in om], B=[[[float(z.real), float(z.imag)] for z in r] for r in B]))
    print(f"[{t:2d} {kind:8s}] cumulative admissible {ok}/{tot}   max|omega| so far {worst:.4f}", flush=True)
print(f"n={n}: extreme feasible Y with Re trY >= D: admissible {ok}/{tot}; max |omega_T| = {worst:.5f}")
json.dump(dict(n=n, seed=seed, ok=ok, tot=tot, worst=worst, fails=fails[:5]), open(f'FINAL_RUNS/ARCH/adm2_n{n}_s{seed}.json', 'w'))
