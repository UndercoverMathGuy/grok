"""arch_ellip.py -- the elliptope criterion:  R_Y := I + (Y_up + Y_up^*)/trY  >= 0  (Y_up = strictly upper part).
Necessary for admissibility at every n (it is the chain matrix of the would-be probe measure), and SUFFICIENT at n=2
(complex 3x3 elliptope = convex hull of unimodular rank-ones, since extreme points have rank r with r^2 <= 3).
Report lambda_min(R_Y) for (UT)-optimal Y* and for extreme feasible Y with Re trY >= D; also with Re trY = D exactly.
usage: arch_ellip.py n ntrials seed
"""
import numpy as np, sys, json, cvxpy as cp

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
    elif kind == 'corr1':
        r = rng.uniform(0.5, 0.99); B = (1 - r) * np.eye(D) + r * np.ones((D, D)); ph = np.exp(1j * rng.uniform(0, 2 * np.pi, D)); B = (ph[:, None] * B) * ph.conj()[None, :]
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D)

def solve(B, Xi=None, eq=False):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)] + [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0]
    if Xi is None: obj = cp.real(cp.trace(Y))
    else:
        cons += [cp.real(cp.trace(Y)) == D] if eq else [cp.real(cp.trace(Y)) >= D]; obj = cp.real(cp.trace(Xi @ Y))
    cp.Problem(cp.Maximize(obj), cons).solve(solver=cp.CLARABEL)
    return Y.value

def lam_min_R(Y):
    t = np.trace(Y); Y = Y * np.exp(-1j * np.angle(t)); t = np.trace(Y).real
    Yup = np.triu(Y, 1); R = np.eye(D) + (Yup + Yup.conj().T) / t
    return float(np.linalg.eigvalsh(R).min())

worst = dict(opt=1e9, ext=1e9, exteq=1e9); recs = []
kinds = ['wishart', 'ill', 'chain', 'nearblock', 'collapse', 'corr1']
for tI in range(ntr):
    kind = kinds[tI % len(kinds)]; B = randB(kind)
    lo = lam_min_R(solve(B)); worst['opt'] = min(worst['opt'], lo)
    le = min(lam_min_R(solve(B, rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))) for _ in range(3)); worst['ext'] = min(worst['ext'], le)
    lq = min(lam_min_R(solve(B, rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)), eq=True)) for _ in range(3)); worst['exteq'] = min(worst['exteq'], lq)
    recs.append(dict(kind=kind, opt=lo, ext=le, exteq=lq))
    print(f"[{tI:2d} {kind:9s}] lambda_min(R_Y): optimal {lo:+.4f}   extreme(trY>=D) {le:+.4f}   extreme(trY=D) {lq:+.4f}", flush=True)
print(f"n={n}: worst lambda_min(R_Y): optimal {worst['opt']:+.5f}, extreme(>=D) {worst['ext']:+.5f}, extreme(=D) {worst['exteq']:+.5f}   (need >= 0)")
json.dump(dict(n=n, seed=seed, worst=worst, recs=recs), open(f'FINAL_RUNS/ARCH/ellip_n{n}_s{seed}.json', 'w'))
