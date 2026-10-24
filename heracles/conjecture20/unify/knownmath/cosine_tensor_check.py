"""Checks for KM1 (pinching threshold b=2) and KM3 (l_r quasi-nuclear norm of the cosine tensor).
C_nu in (R^2)^{(x) nu}: C[a_1..a_nu] = coefficient table of cos(theta_1+...+theta_nu) = Re prod_t (cos th_t + i sin th_t).
"""
import numpy as np, itertools
from scipy.optimize import least_squares
rng = np.random.default_rng(2)

def cosine_tensor(nu):
    C = np.zeros((2,)*nu)
    for idx in itertools.product([0, 1], repeat=nu):
        # Re prod (x_t + i y_t): picks y in slots with idx=1, factor i^{#y}
        k = sum(idx)
        C[idx] = np.real(1j**k)
    return C

def rank1(vs):
    T = vs[0]
    for v in vs[1:]: T = np.multiply.outer(T, v)
    return T

print("== KM3 identity: equally spaced block ==")
for nu in [2, 3, 4, 5, 6]:
    C = cosine_tensor(nu)
    S = np.zeros_like(C)
    for l in range(nu):
        ph = [np.pi*l/nu]*(nu-1) + [-(nu-1)*np.pi*l/nu]
        S += rank1([np.array([np.cos(f), -np.sin(f)]) for f in ph])
    S *= 2**(nu-1)/nu
    fro2 = np.sum(C**2)
    # spectral norm: max over unit vectors of Re prod z_t = 1 (check by random sampling)
    best = 0
    for _ in range(20000):
        th = rng.uniform(0, 2*np.pi, nu)
        best = max(best, np.cos(th.sum()))
    print(f"nu={nu}: ||C - block||_max = {np.max(np.abs(C-S)):.1e}; ||C||_F^2 = {fro2:.0f} = 2^(nu-1)={2**(nu-1)}; sampled spectral norm {best:.6f}; nuclear lower bound F^2/sigma = {fro2:.0f} = block cost {2**(nu-1)}")

print("== KM3 lower bound: random exact decompositions with N > nu terms ==")
def fit(nu, N, C):
    def res(x):
        V = x.reshape(N, nu, 2)
        S = sum(rank1(list(V[i])) for i in range(N))
        return (S - C).ravel()
    x0 = rng.normal(size=N*nu*2)
    r = least_squares(res, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=5000)
    return r.x.reshape(N, nu, 2), np.max(np.abs(r.fun))
for nu in [3, 4]:
    C = cosine_tensor(nu)
    for r_exp in [2/nu, 0.75, 1.0]:
        bound = nu**(1-r_exp) * 2**((nu-1)*r_exp)
        worst = np.inf; cnt = 0
        for trial in range(40):
            N = nu + rng.integers(0, 4)
            V, err = fit(nu, N, C)
            if err > 1e-9: continue
            cnt += 1
            norms = np.prod(np.linalg.norm(V, axis=2), axis=1)
            worst = min(worst, np.sum(norms**r_exp) / bound)
        print(f"nu={nu} r={r_exp:.3f}: {cnt} exact decompositions, min cost/bound = {worst:.6f} (must be >= 1)")

print("== KM1 pinching: sum_k prod_t ||P_k v_t||^r <= prod_t ||v_t||^r iff r*nu >= 2 ==")
for nu in [2, 3, 4]:
    for b in [1.5, 2.0, 2.5, nu]:
        r_exp = b/nu
        worst = 0
        for _ in range(20000):
            K = rng.integers(2, 5)
            # slot vectors split into K blocks: record block norms
            a = np.abs(rng.normal(size=(K, nu))) * (rng.random((K, nu)) < 0.8)
            a = a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-300)
            lhs = np.sum(np.prod(a**r_exp, axis=1))
            worst = max(worst, lhs)
        # extremal example: equal split over 2 blocks in every slot
        eq = 2 * (2**-0.5)**(nu*r_exp)
        print(f"nu={nu} b={b}: max pinched/original = {worst:.4f}; equal-split example = {eq:.4f}")
