"""arch_markov.py -- explicit measures for the (UT) probe at n=2 (Markov-chain construction).
u_1 ~ Poisson(z1) (E u_1 = z1),  u_2 | u_1 ~ Poisson(alpha + beta conj(u_1))  (legal iff |alpha|+|beta| <= 1).
Moments: E u_1 = z1, E u_2 = alpha + beta conj(z1), E u_1u_2 = alpha z1 + beta.
Given the probe (w1, w2, w12) = (Y01, Y12, Y02)/trY, solve z1 = w1 and the 2x2 system for (alpha, beta);
report s := |alpha| + |beta| (need <= 1). Also the reversed chain (u_2 first). Extreme feasible Y with Re trY >= D.
usage: arch_markov.py ntrials seed
"""
import numpy as np, sys, json, cvxpy as cp

ntr = int(sys.argv[1]); seed = int(sys.argv[2]); n = 2; D = 3
rng = np.random.default_rng(seed)

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

def extreme(B, Xi):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)] + [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0]
    if Xi is None: obj = cp.real(cp.trace(Y))
    else: cons += [cp.real(cp.trace(Y)) >= D]; obj = cp.real(cp.trace(Xi @ Y))
    cp.Problem(cp.Maximize(obj), cons).solve(solver=cp.CLARABEL)
    return Y.value

def markov(w1, w2, w12):
    z1 = w1
    if abs(z1) >= 1: return np.inf
    a = (w2 - np.conj(z1) * w12) / (1 - abs(z1) ** 2); b = (w12 - z1 * w2) / (1 - abs(z1) ** 2)
    return abs(a) + abs(b)

worst = 0; worst_rev = 0; worst_min = 0; tot = 0; bad = 0
for t in range(ntr):
    B, kind = randB()
    for s in range(5):
        Y = extreme(B, None if s == 0 else rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
        tr = np.trace(Y); w1, w2, w12 = Y[0, 1] / tr, Y[1, 2] / tr, Y[0, 2] / tr
        s1 = markov(w1, w2, w12); s2 = markov(w2, w1, w12)   # reversed slot order
        tot += 1; worst = max(worst, s1); worst_rev = max(worst_rev, s2); worst_min = max(worst_min, min(s1, s2)); bad += (min(s1, s2) > 1)
    print(f"[{t:2d} {kind:8s}] worst |a|+|b|: forward {worst:.4f}, reversed {worst_rev:.4f}, min-of-two {worst_min:.4f}   (need <= 1)", flush=True)
print(f"n=2: forward Markov worst {worst:.5f}; reversed {worst_rev:.5f}; best-of-two worst {worst_min:.5f}; failures {bad}/{tot}")
json.dump(dict(worst=float(worst), worst_rev=float(worst_rev), worst_min=float(worst_min), bad=int(bad), tot=int(tot)), open(f'FINAL_RUNS/ARCH/markov_s{seed}.json', 'w'))
