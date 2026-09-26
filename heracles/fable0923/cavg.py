"""cavg.py -- rigorous transfer from the certified K-mode depth A* to the l2-POLYBALL depth of the 3K body.

Setting (n=2). For z = (z^1, z^2, z^3) in (C^K)^3 (patterns {1},{2},{12}), the support function of the
3K-mode body is  h(z) = max_{a,b} F(a,b),  F = f_1(a) + f_2(b) + f_3(a+b),  f_i(t) = Re sum_k z^i_k zeta_k^t.
Lemma (proved in JOURNAL 2.2): for each of the 6 role assignments sigma = (outer, full, coupled) of the
three patterns and every unit vector d in T^K,
    h(z) >= A* [ Re <d, z^full> + sum_k | z^outer_k + A* d_k conj(z^coupled_k) | ]        (*)
(A* = certified K-mode polydisc depth of S). Hence h(z) >= A* sum_k beta_sigma(z^{(k)}) with
    beta_sigma(x,y,w) = max_{|d|=1} [ Re(conj(d) x) + |y + A* d conj(w)| ],  x=full, y=outer, w=coupled.
Averaging over the 6 assignments:  h(z) >= A* c(A*) sum_k ||z^{(k)}||_2  with
    c(A) = min over unit (x,y,w) in C^3 of (1/6) sum_sigma beta_sigma.
This script computes c(A).  Output: A_ball(S) >= A*(S) * c(A*(S)).
"""
import numpy as np, itertools, sys
from scipy.optimize import minimize

TH = np.linspace(0, 2*np.pi, 4001)[:-1]; D = np.exp(1j*TH)

def beta(x, y, w, A):
    v = (np.conj(D) * x).real + np.abs(y + A * D * np.conj(w))
    i = v.argmax()
    # refine locally with a fine grid
    th = np.linspace(TH[i] - 2e-3, TH[i] + 2e-3, 401); d = np.exp(1j*th)
    return ((np.conj(d) * x).real + np.abs(y + A * d * np.conj(w))).max()

def avg_beta(v, A):
    z = v[:3] + 1j*v[3:]; z = z / np.linalg.norm(z)
    tot = 0.0
    for perm in itertools.permutations(range(3)):     # (full, outer, coupled)
        tot += beta(z[perm[0]], z[perm[1]], z[perm[2]], A)
    return tot / 6

def c_of(A, starts=60, seed=0):
    rng = np.random.default_rng(seed); best = np.inf; bz = None
    for s in range(starts):
        v0 = rng.standard_normal(6)
        if s % 4 == 1: v0 = np.array([1, 0, 0, 0, 0, 0.]) + 0.3*rng.standard_normal(6)
        if s % 4 == 2: v0 = np.array([1, 1, 1, 0, 0, 0.]) + 0.3*rng.standard_normal(6)
        res = minimize(lambda v: avg_beta(v, A), v0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-8, fatol=1e-10))
        if res.fun < best: best, bz = res.fun, res.x
    z = bz[:3] + 1j*bz[3:]; z /= np.linalg.norm(z)
    return best, z

if __name__ == "__main__":
    As = [float(a) for a in sys.argv[1:]] or [0.30, 0.35, 0.38, 0.40, 0.43, 0.46, 0.48, 0.50]
    for A in As:
        c, z = c_of(A)
        print(f"A*={A:.3f}: c(A*)={c:.5f}  -> A_ball >= {A*c:.4f}   worst dir |z|=({np.abs(z[0]):.3f},{np.abs(z[1]):.3f},{np.abs(z[2]):.3f}) phases={np.round(np.angle(z),2)}", flush=True)
