"""arch_adm3.py -- can admissibility of the (UT) probe be certified by an SOS density on the full multi-affine
monomial space?  rho(u) = sum_{alpha,beta in {0,1}^n} Gamma_{alpha beta} conj(u^alpha) u^beta,  Gamma >= 0 (2^n x 2^n),
moments  rhohat_T = sum_{alpha superset T} Gamma_{alpha, alpha - T}  must equal  omega_T (Y data; 0 off intervals), rhohat_0 = 1.
Also the degree-2 version (monomials {0,1,2}^n, 3^n x 3^n) if the first fails.
usage: arch_adm3.py n ntrials seed
"""
import numpy as np, sys, json, cvxpy as cp
from itertools import product

n = int(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3])
D = n + 1; rng = np.random.default_rng(seed)

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

def solveUT(B, Xi=None):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)] + [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0]
    if Xi is None:
        obj = cp.real(cp.trace(Y))
    else:
        cons += [cp.real(cp.trace(Y)) >= D]; obj = cp.real(cp.trace(Xi @ Y))
    cp.Problem(cp.Maximize(obj), cons).solve(solver=cp.CLARABEL)
    return Y.value

def omega_dict(Y):
    om = {}
    for j in range(D):
        for k in range(j + 1, D):
            T = tuple(1 if j <= s < k else 0 for s in range(n)); om[T] = Y[j, k] / np.trace(Y)
    return om

def sos_feasible(om, deg):
    alphas = list(product(range(deg + 1), repeat=n)); idx = {a: i for i, a in enumerate(alphas)}
    m = len(alphas)
    Gm = cp.Variable((m, m), hermitian=True)
    cons = [Gm >> 0]
    for T in product([0, 1], repeat=n):
        terms = []
        for a in alphas:
            b = tuple(a[s] - T[s] for s in range(n))
            if b in idx: terms.append(Gm[idx[a], idx[b]])
        target = 1.0 if sum(T) == 0 else om.get(T, 0)
        cons += [sum(terms) == target]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(Gm))), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve(solver=cp.SCS, eps=1e-8, max_iters=20000)
    return prob.status in ('optimal', 'optimal_inaccurate')

res = dict(deg1=0, deg2=0, tot=0)
for t in range(ntr):
    B, kind = randB()
    for s in range(3):
        Y = solveUT(B) if s == 0 else solveUT(B, rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
        om = omega_dict(Y); res['tot'] += 1
        f1 = sos_feasible(om, 1); res['deg1'] += f1
        f2 = f1 or sos_feasible(om, 2); res['deg2'] += f2
    print(f"[{t:2d} {kind:8s}] SOS-density certificates: degree-1 {res['deg1']}/{res['tot']}, degree<=2 {res['deg2']}/{res['tot']}", flush=True)
print(f"n={n}: degree-1 SOS density feasible {res['deg1']}/{res['tot']}; degree-2 {res['deg2']}/{res['tot']}")
json.dump(dict(n=n, seed=seed, res=res), open(f'FINAL_RUNS/ARCH/adm3_n{n}_s{seed}.json', 'w'))
