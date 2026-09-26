"""arch_ut.py -- test of (UT):  for B > 0 with det B = 1,
      T(B) := max { Re tr Y : Y upper triangular (chain order), [[B, Y],[Y^*, B]] >= 0 }   >=   D ?
If so, Lambda_Y(v) := (1/D) m(v)^* Y m(v) = (trY)/D + (1/D) sum_{j<k} Y_jk v^{(j,k]} is an interval polynomial with
|Lambda_Y| <= q_B/D pointwise (Cauchy-Schwarz with the contraction W = G^{-*} Y G^{-1}), and after dividing by trY/D >= 1
it is a candidate probe; admissibility (moments of a probability measure) is then checked by LP on a torus grid.
usage: arch_ut.py n ntrials seed
"""
import numpy as np, sys, json, cvxpy as cp
from scipy.optimize import linprog
from itertools import combinations, product

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
    elif kind == 'nearI':
        H = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); H = (H + H.conj().T) / 2; B = np.eye(D) + 0.3 * H / np.linalg.norm(H, 2)
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D)

def solve_UT(B):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)]
    Big = cp.bmat([[B, Y], [cp.conj(Y.T), B]])
    cons += [Big >> 0]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Y))), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value, Y.value

# admissibility LP on a torus grid
G = {1: 2000, 2: 72, 3: 22}[n]
th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)
subsets = [T for k in range(1, n + 1) for T in combinations(range(n), k)]
MON = np.stack([np.exp(1j * grid[:, list(T)].sum(1)) for T in subsets], 1)
def admissible(om):   # om: dict subset->complex (non-listed = 0)
    target = np.array([om.get(T, 0) for T in subsets])
    Aeq = np.concatenate([MON.real.T, MON.imag.T, np.ones((1, len(grid)))], 0)
    beq = np.concatenate([target.real, target.imag, [1.0]])
    r = linprog(np.zeros(len(grid)), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    return r.status == 0

recs = []
for t in range(ntr):
    kind = ['wishart', 'ill', 'chain', 'nearblock', 'nearI'][t % 5]
    B = randB(kind)
    val, Y = solve_UT(B)
    om = {}
    for j in range(D):
        for k in range(j + 1, D):
            om[tuple(range(j, k))] = Y[j, k] / np.trace(Y)   # v^{(j,k]} = slots j..k-1 (0-based)
    adm = admissible(om) if val >= D - 1e-6 else None
    recs.append(dict(kind=kind, T=float(val), cond=float(np.linalg.cond(B)), admissible=adm))
    print(f"[{t:2d} {kind:9s} cond={np.linalg.cond(B):8.2f}]  T(B) = {val:.6f}  (need >= {D})   {'OK' if val >= D - 1e-6 else 'FAIL'}   admissible={adm}", flush=True)
Ts = np.array([r['T'] for r in recs])
print(f"n={n}: min T(B) over {ntr} trials = {Ts.min():.6f}  vs D = {D};  fails: {int((Ts < D - 1e-6).sum())};  inadmissible: {sum(1 for r in recs if r['admissible'] is False)}")
json.dump(dict(n=n, seed=seed, recs=recs), open(f'FINAL_RUNS/ARCH/ut_n{n}_s{seed}.json', 'w'))
