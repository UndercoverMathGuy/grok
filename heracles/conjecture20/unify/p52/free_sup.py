"""Sanity check of the free relaxation and its supremum over atom systems at (5,2).
ratio = h_free(A,B,C,D) / (Gamma_s 3^{1-s} Q^s)  with Q = sum |beta_i|^r (exact atom cost).
ratio < 1 for all atom systems would prove W_1 < gamma_IF (IF value with both features).
Usage: python free_sup.py b natoms restarts seed
"""
import sys
import numpy as np
from scipy.optimize import minimize
from p52lib import *

b = float(sys.argv[1]); m = int(sys.argv[2]); R = int(sys.argv[3]); seed = int(sys.argv[4])
s = 3.0 / b; r = 1.0 / s
G = Gamma(s)
rng = np.random.default_rng(seed)

# --- sanity: h_free >= h for random networks with beta_2 = 0 forced by removing ... (just check h <= h_free on random nets)
viol = 0
for t in range(20):
    U = rng.normal(size=(4, 2, 5)); W = rng.normal(size=(4, 5))
    bt = betas(U, W)
    # kill clean beta_2 by adding a compensating... simply test with outputs at freq 1 + arbitrary: compare h with h_free only
    # when beta_2 = 0; instead check the weaker identity on the LP: actual g is feasible for LP if beta_2 = 0.
    # Make beta_2 = 0: project W's frequency-2 part orthogonally so that sum conj(What(2)) prod Uh = 0
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    v = Uh[:, 0, 2] * Uh[:, 1, 2]      # beta_2 = 2p^-3 sum conj(Wh2) v ; remove component of Wh2 along v
    Wh2 = Wh[:, 2]
    Wh2 = Wh2 - (np.vdot(v, Wh2) / np.vdot(v, v)).conjugate() * v * 0  # placeholder
    # direct: solve for complex scalar c with sum conj(Wh2 - c v) v = 0  ->  c = conj(sum conj(Wh2) v)/|v|^2 ... (conj(c) |v|^2 = sum conj(Wh2) v)
    cc = np.conj(np.sum(np.conj(Wh[:, 2]) * v) / np.sum(np.abs(v) ** 2))
    Wh[:, 2] = Wh[:, 2] - cc * v
    Wh[:, 3] = np.conj(Wh[:, 2])
    W2 = np.real(np.fft.ifft(Wh, axis=1))
    bt = betas(U, W2)
    assert abs(np.sum(bt[2])) < 1e-12
    Gp = gaps(U, W2); h = Gp.min()
    Phi = fields(U, W2)
    A = np.conj(fcoef(Phi[1], (0, 0)))
    bT = [np.conj(fcoef(Phi[1], chi)) for chi in OWN1]
    hf = h_free(A, bT)
    if h > hf + 1e-9:
        viol += 1
print(f"sanity: h <= h_free on 20 random beta_2=0 networks: violations {viol}", flush=True)


def unpack(v):
    beta = v[0:m] * np.exp(1j * v[m:2 * m])
    xi = np.exp(1j * v[2 * m:4 * m].reshape(m, 2))
    return beta, xi


def ratio(v):
    beta, xi = unpack(v)
    Q = np.sum(np.abs(beta) ** r)
    if Q < 1e-9:
        return 0.0
    A, bT = own_from_atoms(beta, xi)
    try:
        hf = h_free(A, bT)
    except Exception:
        return 0.0
    return hf / (G * 3 ** (1 - s) * Q ** s)


best = (-1, None)
for k in range(R):
    v0 = np.concatenate([rng.uniform(0.2, 1, m), rng.uniform(0, 2 * np.pi, m), rng.uniform(0, 2 * np.pi, 2 * m)])
    if k == 0:   # IF block
        v0 = np.concatenate([np.ones(m) / m, np.zeros(m), np.array([[2 * np.pi * i / 3] * 2 for i in range(m)]).ravel()])
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 3000 * m, 'xatol': 1e-9, 'fatol': 1e-12})
    val = -res.fun
    if val > best[0]:
        best = (val, res.x)
    print(f"  start {k}: {val:.6f}   best {best[0]:.6f}", flush=True)
beta, xi = unpack(best[1])
A, bT = own_from_atoms(beta, xi)
print(f"b={b} s={s:.4f} Gamma={G:.5f}: best free ratio {best[0]:.6f}")
print("  |beta_i| =", np.round(np.abs(beta), 4), " A=", np.round(A, 4), " |B,C,D|/|A| =", np.round(np.abs(bT) / abs(A), 4))
print("  single-feature IF ratio would be", KA / G)
