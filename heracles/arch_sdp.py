"""arch_sdp.py -- SDP certificate for (PW) at n=2 for a given B (det 1):
   find omega (moments of a probability measure on a torus grid) and P >= 0 (8x8) with
      [[q_B/D - s, Lam_omega],[conj Lam_omega, q_B/D - s]] (v)  ==  W(v)^* P W(v),   W = I_2 (x) z(v),  z = (1, v1, v2, v1 v2),
   maximising s.  s >= 0 certifies (PW) at B (up to the grid-measure admissibility, which is exact in the limit).
Prints s*, the rank / structure of P, omega, and compares omega with simple functions of B.
usage: arch_sdp.py kind seed [eps]      kind in {nearI, wishart, chain, nearblock, worstH}
"""
import numpy as np, sys, json, cvxpy as cp
from itertools import product

kind = sys.argv[1]; seed = int(sys.argv[2]); eps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
rng = np.random.default_rng(seed); n = 2; D = 3

def randB():
    if kind == 'nearI':
        H = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); H = (H + H.conj().T) / 2; B = np.eye(D) + eps * H
    elif kind == 'worstH':   # the tight second-order direction found by arch_local2 (n=2)
        H = np.zeros((D, D), complex); H[0, 1] = 0.1875 + 0.5461j; H[0, 2] = 0.1875 - 0.5461j; H[1, 2] = 0.5187 + 0.2536j
        H = H + H.conj().T; B = np.eye(D) + eps * H
    elif kind == 'wishart':
        A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)); B = A @ A.conj().T + 0.05 * np.eye(D)
    elif kind in ('chain', 'nearblock'):
        if kind == 'chain':
            N = rng.integers(D, 2 * D + 2); x = rng.uniform(0.1, 1, N); v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
        else:
            N = D; x = np.ones(N) / D + 0.05 * rng.normal(size=N); v = np.exp(1j * (2 * np.pi * np.arange(N)[:, None] / D + eps * rng.normal(size=(N, n))))
        ch = np.concatenate([np.ones((N, 1)), np.cumprod(v, 1)], 1); M = (ch.T * np.abs(x)) @ ch.conj(); B = np.linalg.inv(M)
    B = (B + B.conj().T) / 2
    ev = np.linalg.eigvalsh(B); assert ev.min() > 0, ev
    return B / np.linalg.det(B).real ** (1 / D)

B = randB()
# q_B modes: gamma in {-1,0,1}^2 ; q = trB + 2 Re(B01 v1 + B02 v1v2 + B12 v2)
qhat = {(0, 0): np.trace(B).real / D + 0j, (1, 0): B[0, 1] / D, (0, 1): B[1, 2] / D, (1, 1): B[0, 2] / D}
for g in list(qhat):
    if g != (0, 0): qhat[(-g[0], -g[1])] = np.conj(qhat[g])
alphas = list(product([0, 1], repeat=n))            # z monomials
idx = {a: i for i, a in enumerate(alphas)}
Gg = 24; th = 2 * np.pi * np.arange(Gg) / Gg
U = np.stack(np.meshgrid(th, th, indexing='ij'), -1).reshape(-1, 2)
theta = cp.Variable(len(U), nonneg=True)
P = cp.Variable((8, 8), hermitian=True)
s = cp.Variable()
cons = [P >> 0, cp.sum(theta) == 1]
# omega_T = sum_u theta_u u^T  (T in {(1,0),(0,1),(1,1)})
def omega(T):
    ph = U @ np.array(T); return theta @ np.cos(ph) + 1j * (theta @ np.sin(ph))
om = {(0, 0): 1.0, (1, 0): omega((1, 0)), (0, 1): omega((0, 1)), (1, 1): omega((1, 1))}
# blocks: P00 (0:4), P11 (4:8), P01 (0:4, 4:8) ; entry (a,b) block sum over beta-alpha = gamma of P_{alpha,beta}
def block_mode(bi, bj, gamma):
    terms = []
    for a in alphas:
        b = (a[0] + gamma[0], a[1] + gamma[1])
        if b in idx: terms.append(P[4 * bi + idx[a], 4 * bj + idx[b]])
    return sum(terms) if terms else 0
for gamma in product([-1, 0, 1], repeat=n):
    target = qhat.get(gamma, 0) - (s if gamma == (0, 0) else 0)
    cons += [block_mode(0, 0, gamma) == target, block_mode(1, 1, gamma) == target]
    # Lambda has only gamma >= 0
    lam = om.get(gamma, 0) if (gamma[0] >= 0 and gamma[1] >= 0) else 0
    cons += [block_mode(0, 1, gamma) == lam]
prob = cp.Problem(cp.Maximize(s), cons)
prob.solve(solver=cp.CLARABEL)
Pv = P.value; ev = np.linalg.eigvalsh(Pv)
omv = np.array([om[(1, 0)].value, om[(0, 1)].value, om[(1, 1)].value])
print(f"kind={kind} seed={seed} eps={eps}: status={prob.status}  s* = {s.value:+.6e}")
print(f"  eig(P) = {np.round(ev, 5)}   (rank ~ {int((ev > 1e-6 * ev.max()).sum())})")
print(f"  omega (v1, v2, v1v2)   = {np.round(omv, 5)}")
naive = np.array([2 * B[0, 1], 2 * B[1, 2], 2 * B[0, 2]]) / np.trace(B).real
print(f"  naive 2B_I/trB          = {np.round(naive, 5)}")
Bi = np.linalg.inv(B)
print(f"  B_I / B_kk (row-normal.) = {np.round([B[0,1]/B[0,0], B[1,2]/B[1,1], B[0,2]/B[0,0]], 5)}")
print(f"  -Binv_I / Binv_kk        = {np.round([-Bi[0,1]/Bi[0,0], -Bi[1,2]/Bi[1,1], -Bi[0,2]/Bi[0,0]], 5)}")
print(f"  B = \n{np.round(B, 4)}")
json.dump(dict(kind=kind, seed=seed, eps=eps, s=float(s.value), B=[[[float(z.real), float(z.imag)] for z in r] for r in B],
               omega=[[float(z.real), float(z.imag)] for z in omv], eigP=ev.tolist(),
               P=[[[float(z.real), float(z.imag)] for z in r] for r in Pv]),
          open(f'FINAL_RUNS/ARCH/sdp_{kind}_s{seed}_e{eps}.json', 'w'))
