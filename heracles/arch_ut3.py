"""arch_ut3.py -- the admissibility-constrained (UT) program.
   T_ell(B) := max { tr Y : Y UT, tr Y real, [[B,Y],[Y^*,B]] >= 0,  (tr Y) I + Y_up + Y_up^* >= 0 }
The last LMI says the probe's chain matrix R_Y = I + (Y_up+Y_up^*)/trY is PSD: NECESSARY for admissibility at every n,
SUFFICIENT at n = 2 (complex 3x3 elliptope = conv of unimodular rank-ones). Need T_ell(B) >= D det(B)^{1/D} = D.
Adversarial: many B kinds incl. very ill-conditioned; then Nelder-Mead on B (8 params at D=3) to minimise T_ell(B).
usage: arch_ut3.py n ntrials seed [adv_steps]
"""
import numpy as np, sys, json, cvxpy as cp
from scipy.optimize import minimize

n = int(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3]); adv = int(sys.argv[4]) if len(sys.argv) > 4 else 0
D = n + 1; rng = np.random.default_rng(seed)

def randB(kind):
    if kind == 'wishart':
        A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); B = A @ A.conj().T + 1e-3 * np.eye(D)
    elif kind == 'ill':
        U, _ = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))); lam = np.exp(rng.uniform(-4, 4, D)); B = (U * lam) @ U.conj().T
    elif kind == 'chain':
        N = rng.integers(D, 2 * D + 2); x = rng.uniform(0.05, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-9 * np.eye(D))
    elif kind == 'nearblock':
        N = D; x = np.ones(N) / D + 0.05 * rng.normal(size=N); v = np.exp(1j * (2 * np.pi * np.arange(N)[:, None] / D + 0.3 * rng.normal(size=(N, n))))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * np.abs(x)) @ ch.conj(); B = np.linalg.inv(M)
    elif kind == 'collapse':
        N = D + 1; x = rng.uniform(0.2, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n))); v[1] = v[0] * np.exp(0.01j * rng.normal(size=n))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-9 * np.eye(D))
    elif kind == 'corr1':
        r = rng.uniform(0.5, 0.995); B = (1 - r) * np.eye(D) + r * np.ones((D, D)); ph = np.exp(1j * rng.uniform(0, 2 * np.pi, D)); B = (ph[:, None] * B) * ph.conj()[None, :]
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D)

def T_ell(B):
    Y = cp.Variable((D, D), complex=True); t = cp.Variable()
    Yup = Y - cp.diag(cp.diag(Y))
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)] + [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0,
            cp.imag(cp.trace(Y)) == 0, t == cp.real(cp.trace(Y)), t * np.eye(D) + Yup + cp.conj(Yup.T) >> 0]
    prob = cp.Problem(cp.Maximize(t), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve(solver=cp.SCS, eps=1e-9, max_iters=50000)
    return prob.value, Y.value

def T_plain(B):
    Y = cp.Variable((D, D), complex=True)
    cons = [Y[j, k] == 0 for j in range(D) for k in range(j)] + [cp.bmat([[B, Y], [cp.conj(Y.T), B]]) >> 0]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Y))), cons); prob.solve(solver=cp.CLARABEL); return prob.value

worst = 1e9; wB = None; recs = []
kinds = ['wishart', 'ill', 'chain', 'nearblock', 'collapse', 'corr1']
for tI in range(ntr):
    kind = kinds[tI % 6]; B = randB(kind)
    te, Y = T_ell(B); tp = T_plain(B)
    recs.append(dict(kind=kind, T_ell=float(te), T=float(tp), cond=float(np.linalg.cond(B))))
    if te < worst: worst = te; wB = B
    print(f"[{tI:2d} {kind:9s} cond={np.linalg.cond(B):10.1f}]  T(B)={tp:.5f}   T_ell(B)={te:.5f}   (need >= {D})  {'OK' if te >= D - 1e-5 else 'FAIL'}", flush=True)
print(f"n={n}: min T_ell(B) over {ntr} random B = {worst:.6f} vs D = {D}")

if adv > 0:
    # adversarial descent on B: parametrise B = L L^* / det^{1/D}, L lower triangular complex (real diag)
    def unpack(z):
        L = np.zeros((D, D), complex); i = 0
        for r in range(D):
            for c in range(r + 1):
                if c == r: L[r, c] = np.exp(z[i]); i += 1
                else: L[r, c] = z[i] + 1j * z[i + 1]; i += 2
        B = L @ L.conj().T; return B / np.linalg.det(B).real ** (1 / D)
    npar = D + D * (D - 1)
    best = 1e9; bestB = None
    for s in range(adv):
        z0 = rng.normal(size=npar) * rng.choice([0.3, 1.0, 2.0])
        r = minimize(lambda z: T_ell(unpack(z))[0], z0, method='Nelder-Mead', options=dict(maxiter=150, xatol=1e-4, fatol=1e-7))
        if r.fun < best: best = r.fun; bestB = unpack(r.x)
        print(f"  adversary start {s}: min T_ell so far {best:.6f}", flush=True)
    print(f"n={n}: ADVERSARIAL min T_ell(B) = {best:.6f} vs D = {D}   cond(B) = {np.linalg.cond(bestB):.2f}")
    recs.append(dict(kind='adversarial', T_ell=float(best), cond=float(np.linalg.cond(bestB)), B=[[[float(z.real), float(z.imag)] for z in r] for r in bestB]))
json.dump(dict(n=n, seed=seed, recs=recs), open(f'FINAL_RUNS/ARCH/ut3_n{n}_s{seed}.json', 'w'))
