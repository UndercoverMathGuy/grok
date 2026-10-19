"""Referee-3 Part 4: high-ratio networks at (5,2) with beta_2 = 0 vs the certified point bound.

Networks: IF block of feature 1 (3 neurons, xi = eta = omega^k) + 1..3 extra random neurons (DC, every frequency) + noise,
then locally optimised by SLSQP (maximise h at cost 1, beta_2 = 0) at s = 3/2.  For each final network compute the directly
evaluated gamma/gamma_IF at s = 1.5, 1.25, 1 and the certified bound at its normalised own data
(payment on the point box, trivial bound, rigorous evaluation of a degree-2 certificate solved at that point).
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/p52')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/p52/verify')
from cert_hs import cert_data, eval_cells
from hs2 import elim_ratio
from pay_adapt import payment

p = 5; rho = np.exp(2j * np.pi / p); xs = np.arange(p)
X1, X2 = np.meshgrid(xs, xs, indexing='ij'); Y = (X1 + X2) % p
Fm = rho ** (-np.outer(xs, xs))
B5 = 2 * 30 ** -1.5; KA = 1 - np.cos(2 * np.pi / 5)
rng = np.random.default_rng(99)


def gaps(U1, U2, W):
    f = np.einsum('ic,ia,ib->abc', W, U1, U2)
    return np.stack([f[X1, X2, Y] - f[X1, X2, (Y - d) % p] for d in (1, 2, 3, 4)], -1)


def own(U1, U2, W, j=1):
    w = (W @ Fm); a = (U1 @ Fm); b = (U2 @ Fm); c = 2 / 125
    return (np.sum(c * np.conj(w[:, j]) * a[:, j] * b[:, j]), np.sum(c * np.conj(w[:, j]) * a[:, -j % p] * b[:, j]),
            np.sum(c * np.conj(w[:, j]) * a[:, j] * b[:, -j % p]), np.sum(c * np.conj(w[:, j]) * a[:, -j % p] * b[:, -j % p]))


def ratio(U1, U2, W, s):
    nrm2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
    P = np.sum(nrm2 ** (1.5 / s))
    return gaps(U1, U2, W).min() / P ** s / (3 ** (1 - s) * B5 * 2.5 * 2 ** -s)


def start():
    rows = []
    for k in range(3):
        u = np.cos(2 * np.pi * xs / p - np.pi * k / 3)
        uh = np.sum(u * rho ** (-xs)); t = np.conj(p ** 3 / (2 * uh * uh))
        rows.append((u, u, (2 / p) * np.real(t * rho ** xs)))
    for e in range(rng.integers(1, 4)):
        sc = rng.uniform(0.05, 0.5)
        rows.append(tuple(sc * rng.normal(size=p) for _ in range(3)))
    U1 = np.array([r[0] for r in rows]) + 0.03 * rng.normal(size=(len(rows), p))
    U2 = np.array([r[1] for r in rows]) + 0.03 * rng.normal(size=(len(rows), p))
    W = np.array([r[2] for r in rows]) + 0.03 * rng.normal(size=(len(rows), p))
    return U1, U2, W


def optimise(U1, U2, W, iters):
    m = U1.shape[0]
    unpack = lambda z: (z[:m * p].reshape(m, p), z[m * p:2 * m * p].reshape(m, p), z[2 * m * p:3 * m * p].reshape(m, p))
    cost = lambda z: np.sum(sum((M ** 2).sum(1) for M in unpack(z[:-1])))
    b2 = lambda z: (lambda v: np.array([v.real, v.imag]) * 1e3)(own(*unpack(z[:-1]), j=2)[0])
    z0 = np.concatenate([U1.ravel(), U2.ravel(), W.ravel(), [0.0]])
    z0[:-1] /= np.sqrt(cost(z0)); z0[-1] = gaps(*unpack(z0[:-1])).min()
    res = minimize(lambda z: -z[-1], z0, method='SLSQP', options=dict(maxiter=iters, ftol=1e-12),
                   constraints=[{'type': 'ineq', 'fun': lambda z: (gaps(*unpack(z[:-1])) - z[-1]).ravel()},
                                {'type': 'eq', 'fun': lambda z: np.array([1.0 - cost(z)])},
                                {'type': 'eq', 'fun': b2}])
    return unpack(res.x[:-1])


def point_bound(A, B, C, D):
    sc = abs(A); A, B, C, D = A / sc, B / sc, C / sc, D / sc
    al = np.angle(A)
    if al < 0:
        A, B, C, D = np.conj(A), np.conj(B), np.conj(C), np.conj(D); al = -al
    if abs(C) > abs(B):
        B, C = C, B
    if al > np.pi / 2:
        return 0.0, None
    ph = np.angle(A * D * np.conj(B * C)) if abs(B * C * D) > 0 else 0.0
    m = np.array([abs(B), abs(C), abs(D)])
    Gm = 2.5 * 2 ** -1.5
    if m.max() >= 1.4:
        return KA * np.cos(al) / (Gm * m.max() / np.sqrt(3)), m
    F = payment(m, m, ph, ph)
    triv = KA * np.cos(al) / (Gm * F)
    try:
        val, Gs = elim_ratio(al, m[0], m[1], m[2], ph - al, F, deg=2, ret=True)
        r = eval_cells(cert_data(Gs), np.array([al]), np.array([al]), m[None, :], m[None, :], np.array([ph - al]),
                       np.array([ph - al]), np.array([F]))[0]
    except Exception:
        r = np.inf
    return min(triv, r), m


import ref3_advlib52 as AL


def run_adv(perturb):
    """balanced IF start + extra neurons + noise (ref3_adversary), optionally SLSQP-optimised; returns (U1, U2, W)"""
    U1, U2, W = AL.start()
    if not perturb:
        m = U1.shape[0]
        z0 = np.concatenate([U1.ravel(), U2.ravel(), W.ravel(), [AL.gaps(U1, U2, W).min()]])
        return U1, U2, W
    m = U1.shape[0]
    z0 = np.concatenate([U1.ravel(), U2.ravel(), W.ravel(), [AL.gaps(U1, U2, W).min()]])
    b = AL.b
    cons = [{'type': 'ineq', 'fun': lambda z: (AL.gaps(*AL.unpack(z[:-1], m)) - z[-1]).ravel()},
            {'type': 'eq', 'fun': lambda z: np.array([1.0 - np.sum(sum((M ** 2).sum(1) for M in AL.unpack(z[:-1], m)) ** (b / 2))])},
            {'type': 'eq', 'fun': lambda z: (lambda v: np.array([v.real, v.imag]) * 1e3)(AL.beta(*AL.unpack(z[:-1], m), 2))}]
    res = minimize(lambda z: -z[-1], z0, method='SLSQP', constraints=cons, options=dict(maxiter=perturb, ftol=1e-12))
    return AL.unpack(res.x[:-1], m)


t0 = time.time()
rows = []
for k in range(16):
    U1, U2, W = run_adv([0, 5, 20, 500][k % 4])
    b2 = own(U1, U2, W, j=2)[0]
    if abs(b2) > 0:
        c = np.cos(4 * np.pi * xs / p); tt = np.conj(-b2 / ((2 / 125) * 6.25))
        wk = (2 / 5) * np.real(tt * rho ** (2 * xs))
        g = (np.linalg.norm(c) ** 2 * np.linalg.norm(wk)) ** (1 / 3)          # balanced factors, same product
        U1 = np.vstack([U1, c * g / np.linalg.norm(c)]); U2 = np.vstack([U2, c * g / np.linalg.norm(c)])
        W = np.vstack([W, wk * g / np.linalg.norm(wk)])
    A, B, C, D = own(U1, U2, W)
    bnd, m = point_bound(A, B, C, D)
    rs = [ratio(U1, U2, W, s) for s in (1.5, 1.25, 1.0)]
    rows.append((rs[0], bnd))
    flag = "VIOLATION" if max(rs) > bnd + 1e-9 else "ok"
    print(f"net {k:2d} iters={[0, 5, 20, 500][k % 4]:3d}: ratio(s=1.5,1.25,1) = {np.round(rs,5).tolist()}  |B,C,D|/|A| = "
          f"{np.round(m,4).tolist() if m is not None else None}  certified point bound = {bnd:.5f}  {flag}")
print(f"max ratio {max(r for r, _ in rows):.5f}; min(bound - ratio_{{3/2}}) = {min(b - r for r, b in rows):.5f}  ({time.time()-t0:.0f}s)")
