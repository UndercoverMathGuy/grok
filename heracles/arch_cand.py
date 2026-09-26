"""arch_cand.py -- explicit admissible probe candidates for (PW) at n=2 (elliptope = moment body):
   R_omega must be a unit-diagonal PSD 3x3 matrix; candidates built from B (det 1):
     corr(B)_alpha : R = (1-alpha) I + alpha * diag(B)^{-1/2} B diag(B)^{-1/2}      (alpha = 2/D forced at 1st order)
     corr(Binv)    : R = (1-alpha) I + alpha * conj(corr(B^{-1}))  (sign-flipped)
     cholesky      : R from the unit-lower Cholesky factor structure
   For each: pointwise slack  min_v [q_B(v)/3 - |Lambda_omega(v)|]  (need >= 0), and the best alpha by 1-D search.
   Also the LP-free 'exact' answer: maximise slack over omega in the elliptope (SDP-free: Nelder-Mead over 6 params
   with a penalty for leaving the elliptope), to see how much room there is per B.
usage: arch_cand.py ntrials seed
"""
import numpy as np, sys, json
from scipy.optimize import minimize, minimize_scalar

ntr = int(sys.argv[1]); seed = int(sys.argv[2]); n = 2; D = 3
rng = np.random.default_rng(seed)
G = 160; th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(th, th, indexing='ij'), -1).reshape(-1, 2); V = np.exp(1j * grid)
CH = np.concatenate([np.ones((len(V), 1)), np.cumprod(V, 1)], 1)
MON = np.stack([V[:, 0], V[:, 1], V[:, 0] * V[:, 1]], 1)   # v1, v2, v1v2  <->  omega_(0,1], omega_(1,2], omega_(0,2]

def randB(kind):
    if kind == 'wishart':
        A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); B = A @ A.conj().T + 1e-2 * np.eye(D)
    elif kind == 'ill':
        U, _ = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))); lam = np.exp(rng.uniform(-3, 3, D)); B = (U * lam) @ U.conj().T
    elif kind == 'chain':
        N = rng.integers(D, 2 * D + 2); x = rng.uniform(0.05, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-9 * np.eye(D))
    elif kind == 'nearblock':
        N = D; x = np.ones(N) / D + 0.05 * rng.normal(size=N); v = np.exp(1j * (2 * np.pi * np.arange(N)[:, None] / D + 0.3 * rng.normal(size=(N, n))))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * np.abs(x)) @ ch.conj(); B = np.linalg.inv(M)
    elif kind == 'collapse':
        N = D + 1; x = rng.uniform(0.2, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n))); v[1] = v[0] * np.exp(0.01j * rng.normal(size=n))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-9 * np.eye(D))
    elif kind == 'degenerate':   # one chain direction nearly null in B (population concentrated near one point + rest)
        v0 = np.exp(1j * rng.uniform(0, 2 * np.pi, n)); m0 = np.concatenate([[1], np.cumprod(v0)])
        Q, _ = np.linalg.qr(np.column_stack([m0, rng.normal(size=(D, 2)) + 1j * rng.normal(size=(D, 2))]))
        lam = np.array([10 ** rng.uniform(-5, -2), 10 ** rng.uniform(0, 2), 10 ** rng.uniform(0, 2)]); B = (Q * lam) @ Q.conj().T
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D)

def qB(B): return np.einsum('pk,kl,pl->p', CH.conj(), B, CH).real
def slack(B, om):   # om = (omega_(0,1], omega_(1,2], omega_(0,2])
    lam = 1 + MON @ om
    return float((qB(B) / D - np.abs(lam)).min())
def om_from_R(R):   # R = would-be chain matrix of the probe: R_{kj} = omega_{(j,k]} for k>j
    return np.array([R[1, 0], R[2, 1], R[2, 0]])
def in_ellip(om):
    R = np.array([[1, np.conj(om[0]), np.conj(om[2])], [om[0], 1, np.conj(om[1])], [om[2], om[1], 1]])
    return float(np.linalg.eigvalsh(R).min())

def corr(A):
    d = np.sqrt(np.diag(A).real); return A / np.outer(d, d)

def best_alpha(B, base):   # R(alpha) = (1-alpha) I + alpha*base  (elliptope for alpha in [0,1] since base is unit-diag PSD)
    f = lambda a: -slack(B, om_from_R((1 - a) * np.eye(D) + a * base))
    r = minimize_scalar(f, bounds=(0, 1), method='bounded', options=dict(xatol=1e-6))
    return -r.fun, r.x

def optimise(B, nst=4):
    """max slack over omega in the elliptope (penalised Nelder-Mead)."""
    def f(z):
        om = z[:3] + 1j * z[3:]; e = in_ellip(om)
        return -slack(B, om) + (1e3 * (-e) if e < 0 else 0)
    best = 1e9; bz = None
    for s in range(nst):
        z0 = 0.2 * rng.normal(size=6) if s else np.zeros(6)
        r = minimize(f, z0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-8, fatol=1e-12))
        if r.fun < best: best = r.fun; bz = r.x
    return -best

recs = []; kinds = ['wishart', 'ill', 'chain', 'nearblock', 'collapse', 'degenerate']
for t in range(ntr):
    kind = kinds[t % 6]; B = randB(kind); Bi = np.linalg.inv(B)
    # candidates: note R_{kj} for k>j should be omega_{(j,k]}; B's lower entries B_{kj} (k>j) carry v^{(j,k]} in q_B via conj(m_k) B_{kj} m_j?? q_B = sum conj(m_k)B_{kl}m_l -> term B_{kl} conj(m_k) m_l = B_{kl} v^{(k,l]} for l>k. So omega_{(k,l]} ~ B_{kl} (UPPER entry). Use R = corr(B)^T-consistent: omega_(j,k] = B_{jk}/sqrt(B_jj B_kk).
    cB = corr(B); omB = np.array([cB[0, 1], cB[1, 2], cB[0, 2]])
    baseB = np.array([[1, np.conj(omB[0]), np.conj(omB[2])], [omB[0], 1, np.conj(omB[1])], [omB[2], omB[1], 1]])
    cI = corr(Bi); omI = -np.array([cI[0, 1], cI[1, 2], cI[0, 2]])
    baseI = np.array([[1, np.conj(omI[0]), np.conj(omI[2])], [omI[0], 1, np.conj(omI[1])], [omI[2], omI[1], 1]])
    sB23 = slack(B, om_from_R((1 - 2 / 3) * np.eye(D) + (2 / 3) * baseB))
    sBa, aB = best_alpha(B, baseB)
    sIa, aI = best_alpha(B, baseI) if np.linalg.eigvalsh(baseI).min() >= -1e-9 else (np.nan, np.nan)
    sopt = optimise(B)
    recs.append(dict(kind=kind, cond=float(np.linalg.cond(B)), corrB_a23=sB23, corrB_best=sBa, alphaB=float(aB), corrBinv_best=sIa, opt=sopt))
    print(f"[{t:2d} {kind:10s} cond={np.linalg.cond(B):9.1f}] corr(B) a=2/3: {sB23:+.4f}; best a={aB:.3f}: {sBa:+.4f}; -corr(Binv) best: {sIa:+.4f}; optimised over elliptope: {sopt:+.4f}", flush=True)
json.dump(dict(seed=seed, recs=recs), open(f'FINAL_RUNS/ARCH/cand_s{seed}.json', 'w'))
