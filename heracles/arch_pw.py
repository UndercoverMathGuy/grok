"""arch_pw.py -- test of the pointwise statement (PW) at n=2 (and n=1 as control):
   for every B > 0 (DxD) with det B = 1 there is a probability measure rho on T^n with
        |Lambda_rho(v)| <= q_B(v)/D   for all v in T^n,
   q_B(v) = m(v)^* B m(v)  (m = chain vector),  Lambda_rho(v) = sum_T rhohat_T v^T  (multi-affine probe).
(PW) <=> (F): det M_+ >= F(mu)^D (Sion), and (F) => (E+).
Candidate probe: Lambda_B := (analytic part of q_B)/tr B  (this is the n=1 proof).
Fallback: optimise rho (K atoms on the torus, weights softmax) to maximise min_v [q_B/D - |Lambda_rho|].
usage: arch_pw.py n ntrials seed [K]
"""
import numpy as np, sys, json
from scipy.optimize import minimize, linprog
from itertools import combinations

n = int(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 6
D = n + 1; rng = np.random.default_rng(seed)
G = {1: 2000, 2: 96, 3: 20}[n]
th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)   # angles
V = np.exp(1j * grid)                                                          # (P, n)
subsets = [T for k in range(n + 1) for T in combinations(range(n), k)]        # all patterns, () first
def monos(W):  # (P,n) -> (P, 2^n) multi-affine monomials w^T
    return np.stack([np.prod(W[:, list(T)], axis=1) if T else np.ones(len(W)) for T in subsets], 1)
MON = monos(V)
CH = np.concatenate([np.ones((len(V), 1)), np.cumprod(V, axis=1)], 1)       # chain vectors (P, D)

def qB(B):
    return np.einsum('pk,kl,pl->p', CH.conj(), B, CH).real

def randB(kind):
    if kind == 'wishart':
        A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); B = A @ A.conj().T + 1e-3 * np.eye(D)
    elif kind == 'ill':
        U, _ = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
        lam = np.exp(rng.uniform(-4, 4, D)); B = (U * lam) @ U.conj().T
    elif kind == 'chain':   # B ~ inverse of an M_+ of a random population (the minimiser of tr(M_+ B))
        N = rng.integers(D, 2 * D + 2); x = rng.uniform(0.1, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * x) @ ch.conj(); B = np.linalg.inv(M + 1e-6 * np.eye(D))
    elif kind == 'nearblock':
        N = D; x = np.ones(N) / D + 0.05 * rng.normal(size=N); v = np.exp(1j * (2 * np.pi * np.arange(N)[:, None] / D + 0.3 * rng.normal(size=(N, n))))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * np.abs(x)) @ ch.conj(); B = np.linalg.inv(M + 1e-6 * np.eye(D))
    B = (B + B.conj().T) / 2
    return B / np.linalg.det(B).real ** (1 / D)

def lam_from_omega(om):   # om: (2^n,) with om[0]=1
    return MON @ om

def in_body(om, tol=1e-7):
    """Is om (om[0]=1) in the moment body? LP feasibility on the grid."""
    Aeq = np.concatenate([MON[:, 1:].real.T, MON[:, 1:].imag.T, np.ones((1, len(V)))], 0)
    beq = np.concatenate([om[1:].real, om[1:].imag, [1.0]])
    r = linprog(np.zeros(len(V)), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    return r.status == 0

def candidate(B):
    """analytic part of q_B / tr B, as an omega vector."""
    om = np.zeros(len(subsets), dtype=complex); om[0] = 1.0
    # q_B = sum_{k,k'} conj(m_k) B_{kk'} m_{k'};  conj(m_k) m_{k'} = v^{(k,k']} for k'>k  -> analytic monomials
    for k in range(D):
        for kp in range(k + 1, D):
            T = tuple(range(k, kp))
            om[subsets.index(T)] += 2 * B[k, kp] / np.trace(B).real
    return om

def slack(B, om):
    return float((qB(B) / D - np.abs(lam_from_omega(om))).min())

def optimise(B, K, nst=3):
    """max over rho = sum_i w_i delta_{u_i}  of  min_v [q_B/D - |Lambda_rho|]."""
    q = qB(B) / D
    def f(z):
        w = np.exp(z[:K]); w /= w.sum(); U = np.exp(1j * z[K:].reshape(K, n))
        om = (monos(U) * w[:, None]).sum(0)
        return -float((q - np.abs(MON @ om)).min())
    best = 1e9; bz = None
    for s in range(nst):
        z0 = np.concatenate([rng.normal(size=K), rng.uniform(0, 2 * np.pi, K * n)])
        r = minimize(f, z0, method='Nelder-Mead', options=dict(maxiter=6000, maxfev=6000, xatol=1e-8, fatol=1e-12))
        r = minimize(f, r.x, method='Powell', options=dict(maxiter=3000, xtol=1e-8, ftol=1e-12))
        if r.fun < best: best = r.fun; bz = r.x
    return -best, bz

out = []
for t in range(ntr):
    kind = ['wishart', 'ill', 'chain', 'nearblock'][t % 4]
    B = randB(kind)
    om = candidate(B)
    s_cand = slack(B, om); inb = in_body(om)
    rec = dict(kind=kind, cand_slack=s_cand, cand_in_body=bool(inb), cond=float(np.linalg.cond(B)))
    if s_cand < -1e-9 or not inb:
        val, z = optimise(B, K)
        rec['opt_slack'] = float(val)
    out.append(rec)
    print(f"[{t:2d} {kind:9s} cond={rec['cond']:.2e}] candidate: slack={s_cand:+.3e} in_body={inb}" +
          (f"   optimised: slack={rec['opt_slack']:+.3e}" if 'opt_slack' in rec else ''), flush=True)

cs = np.array([r['cand_slack'] for r in out]); ib = np.array([r['cand_in_body'] for r in out])
os_ = np.array([r.get('opt_slack', np.nan) for r in out])
print(f"n={n}: candidate min slack {cs.min():+.3e}; in body {ib.sum()}/{len(ib)}; " +
      f"optimised min slack (where run) {np.nanmin(os_) if np.isfinite(os_).any() else 'n/a'}")
json.dump(dict(n=n, seed=seed, K=K, G=G, recs=out), open(f'FINAL_RUNS/ARCH/pw_n{n}_s{seed}.json', 'w'))
