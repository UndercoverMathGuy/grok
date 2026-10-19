"""Referee check of KM3 (l_r quasi-nuclear norm of the cosine tensor C_nu).

(1) The claimed minimiser family with RANDOM phi_t (sum phi_t = 0) and random pair sign flips reproduces C_nu exactly,
    and its cost equals nu^{1-r} 2^{(nu-1) r}.  (Agent only tested phi_t = 0.)
(2) Translation to (JF): for random in-plane decompositions, the character coefficients equal 2^{-nu} beta_i prod_{T} xi_it.
(3) Hard search for a cheaper exact decomposition: penalty continuation, N = nu..nu+3 terms, out-of-plane slots allowed
    (ambient R^3 per slot), r in {2/nu, 0.6, 0.85, 1}; report min feasible cost / bound (must be >= 1 - tol).
(4) Rank: nu = 3 hyperdeterminant of C_3 is negative (real rank 3); best (nu-1)-term fits for nu = 3, 4 (residual).
"""
import itertools
import numpy as np
from scipy.optimize import minimize, least_squares
rng = np.random.default_rng(4242)


def cosine_tensor(nu):
    C = np.zeros((2,) * nu)
    for idx in itertools.product([0, 1], repeat=nu):
        C[idx] = np.real(1j ** sum(idx))
    return C


def rank1(vs):
    T = vs[0]
    for v in vs[1:]:
        T = np.multiply.outer(T, v)
    return T


print("(1) general minimiser family")
for nu in [2, 3, 4, 5, 6]:
    n = nu - 1
    C = cosine_tensor(nu)
    worst = 0
    for _ in range(50):
        phi = rng.uniform(0, 2 * np.pi, nu); phi[-1] = -phi[:-1].sum()
        S = np.zeros_like(C)
        for l in range(nu):
            ph = [phi[t] + np.pi * l / nu for t in range(n)] + [phi[-1] - n * np.pi * l / nu]
            vs = [np.array([np.cos(f), -np.sin(f)]) for f in ph]
            # random sign flips of pairs of factors
            for _k in range(2):
                a, b = rng.choice(nu, 2, replace=False)
                vs[a] = -vs[a]; vs[b] = -vs[b]
            S += rank1(vs)
        S *= 2 ** (nu - 1) / nu
        worst = max(worst, np.max(np.abs(S - C)))
    for r in [0.5, 1.0]:
        cost = nu * (2 ** (nu - 1) / nu) ** r
        bound = nu ** (1 - r) * 2 ** ((nu - 1) * r)
        assert abs(cost - bound) < 1e-9 * bound
    print(f"  nu={nu}: max |C - family| over 50 random (phi, sign flips) = {worst:.1e}; cost == bound: yes")

print("(2) character expansion of in-plane rank-one terms")
worst = 0
for _ in range(100):
    nu = rng.integers(2, 6)
    V = rng.normal(size=(nu, 2))
    z = V[:, 0] + 1j * V[:, 1]
    beta = np.prod(np.conj(z)); xi = z / np.conj(z)
    th = rng.uniform(0, 2 * np.pi, (64, nu))
    val = np.prod(V[None, :, 0] * np.cos(th) + V[None, :, 1] * np.sin(th), axis=1)
    rec = np.zeros(64, dtype=complex)
    for eps in itertools.product([1, -1], repeat=nu):
        T = [t for t in range(nu) if eps[t] == -1]
        coef = 2.0 ** (-nu) * beta * np.prod(xi[T]) if T else 2.0 ** (-nu) * beta
        rec += coef * np.exp(1j * th @ np.array(eps))
    worst = max(worst, np.max(np.abs(rec - val)))
print(f"  max reconstruction error = {worst:.1e}")


def search(nu, N, r, amb, ntr):
    C = cosine_tensor(nu)
    Cpad = np.zeros((amb,) * nu); Cpad[(slice(0, 2),) * nu] = C
    bound = nu ** (1 - r) * 2 ** ((nu - 1) * r)

    def unpack(x):
        return x.reshape(N, nu, amb)

    def obj(x, mu):
        V = unpack(x)
        S = sum(rank1(list(V[i])) for i in range(N))
        res = np.sum((S - Cpad) ** 2)
        norms2 = np.prod(np.sum(V ** 2, axis=2), axis=1)
        return np.sum((norms2 + 1e-14) ** (r / 2)) + mu * res, res

    best = np.inf
    for _ in range(ntr):
        x = rng.normal(size=N * nu * amb) * 0.8
        for mu in [0.3, 3, 30, 300, 3e3, 3e4, 3e5]:
            x = minimize(lambda y: obj(y, mu)[0], x, method="L-BFGS-B", options={"maxiter": 4000}).x
        c, res = obj(x, 0)
        if res < 1e-9:
            best = min(best, c / bound)
    return best


print("(3) search for cheaper exact decompositions (cost/bound; must be >= 1 up to penalty slack ~1e-4)")
for nu in [3, 4]:
    for r in [2 / nu, 0.6, 0.85, 1.0]:
        out = []
        for N, amb in [(nu, 2), (nu + 1, 2), (nu + 2, 3), (nu + 3, 2)]:
            out.append(search(nu, N, r, amb, 12 if nu == 3 else 6))
        print(f"  nu={nu} r={r:.3f}: min cost/bound by (N,amb) = " + ", ".join(f"{v:.5f}" for v in out))

print("(4) rank checks")
C3 = cosine_tensor(3)
a = C3
# Cayley hyperdeterminant of 2x2x2
Det = (a[0,0,0]**2*a[1,1,1]**2 + a[0,0,1]**2*a[1,1,0]**2 + a[0,1,0]**2*a[1,0,1]**2 + a[1,0,0]**2*a[0,1,1]**2
       - 2*(a[0,0,0]*a[0,0,1]*a[1,1,0]*a[1,1,1] + a[0,0,0]*a[0,1,0]*a[1,0,1]*a[1,1,1] + a[0,0,0]*a[1,0,0]*a[0,1,1]*a[1,1,1]
            + a[0,0,1]*a[0,1,0]*a[1,0,1]*a[1,1,0] + a[0,0,1]*a[1,0,0]*a[0,1,1]*a[1,1,0] + a[0,1,0]*a[1,0,0]*a[0,1,1]*a[1,0,1])
       + 4*(a[0,0,0]*a[0,1,1]*a[1,0,1]*a[1,1,0] + a[0,0,1]*a[0,1,0]*a[1,0,0]*a[1,1,1]))
print(f"  hyperdeterminant of C_3 = {Det} (negative => real rank 3, complex rank 2)")
for nu in [3, 4]:
    C = cosine_tensor(nu); N = nu - 1
    best = np.inf
    for _ in range(60):
        def res(x):
            V = x.reshape(N, nu, 2)
            return (sum(rank1(list(V[i])) for i in range(N)) - C).ravel()
        rr = least_squares(res, rng.normal(size=N * nu * 2), max_nfev=4000)
        best = min(best, np.sum(rr.fun ** 2))
    print(f"  nu={nu}: best squared residual with nu-1 = {N} real terms = {best:.4e} (bounded away from 0 => no exact {N}-term decomposition found)")
