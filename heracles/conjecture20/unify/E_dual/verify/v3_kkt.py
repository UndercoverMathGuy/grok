"""Referee check of D3c (KKT dual tensor) at GENUINE local maximisers found from random starts (not the decoupled IF
optimum with a hand-built q).  Method: softmin L-BFGS, then sequential-LP polishing (trust region); the final LP's
duals give the KKT measure q on the gap constraints.  Then check
   t_i := E_q g(unit factors of neuron i)  ==  h nu^{nu/2} a_i^{r-1}   (unit b-cost),
   F = log gamma + (s-1) H(w) == E_w log t - (nu/2) log nu,   t_i <= ||E_q g||_sigma (lower bound by ALS)."""
import numpy as np, sys, os
from scipy.optimize import minimize, linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *

rng = np.random.default_rng(5)


def setup(p, n):
    idx = np.indices((p,) * n).reshape(n, -1).T
    y = idx.sum(1) % p
    return idx, y


def gaps_and_jac(U, W, p, n, idx, y):
    """gaps g[k] for k=(x,delta) flattened (x major), and Jacobian dg/dtheta (K, m*(n+1)*p), vectorised."""
    m = U.shape[0]; N = p ** n
    fac = np.stack([U[:, t, idx[:, t]] for t in range(n)], 1)          # (m, n, N)
    prodU = np.prod(fac, axis=1)                                        # (m, N)
    D = np.arange(1, p)
    c2 = (y[:, None] - D[None, :]) % p                                  # (N, p-1)
    f = W.T @ prodU                                                     # (p, N)
    g = (f[y, np.arange(N)][:, None] - f[c2, np.arange(N)[:, None]])    # (N, p-1)
    eye = np.eye(p)
    JW = prodU.T[:, None, :, None] * (eye[y][:, None, None, :] - eye[c2][:, :, None, :])   # (N, p-1, m, p)
    dw = W[:, y][:, :, None] - W[np.arange(m)[:, None, None], c2[None, :, :]]               # (m, N, p-1)
    JU = np.zeros((N, p - 1, m, n, p))
    for t in range(n):
        others = np.prod(np.delete(fac, t, axis=1), axis=1)                                 # (m, N)
        val = (dw * others[:, :, None]).transpose(1, 2, 0)                                  # (N, p-1, m)
        JU[:, :, :, t, :] = val[:, :, :, None] * eye[idx[:, t]][:, None, None, :]
    J = np.concatenate([JU, JW[:, :, :, None, :]], axis=3)              # (N, p-1, m, n+1, p)
    return g.ravel(), J.reshape(N * (p - 1), -1)


def pack(U, W):
    m, n, p = U.shape
    return np.concatenate([U, W[:, None, :]], axis=1).ravel()


def unpackv(v, m, n, p):
    A = v.reshape(m, n + 1, p); return A[:, :n].copy(), A[:, n].copy()


def cost_grad(v, m, n, p, b):
    A = v.reshape(m, -1); S = (A ** 2).sum(1)
    c = np.sum(S ** (b / 2)); gc = (b * S ** (b / 2 - 1))[:, None] * A
    return c, gc.ravel()


def find_local_max(p, n, b, m, seed_rng):
    s = (n + 1) / b; idx, y = setup(p, n)
    v = seed_rng.standard_normal(m * (n + 1) * p)
    for tau in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        def obj(v):
            U, W = unpackv(v, m, n, p)
            g, J = gaps_and_jac(U, W, p, n, idx, y)
            c, gc = cost_grad(v, m, n, p, b)
            gn = g / c ** s; gmin = gn.min(); T = tau * np.abs(gn).max()
            e = np.exp(-(gn - gmin) / T); Z = e.sum(); pi = e / Z
            sm = gmin - T * np.log(Z)
            dsm = (pi @ J) / c ** s - s * (pi @ g) / c ** (s + 1) * gc
            return -sm, -dsm
        res = minimize(obj, v, jac=True, method='L-BFGS-B', options={'maxiter': 3000})
        v = res.x; c, _ = cost_grad(v, m, n, p, b); v = v * c ** (-1 / b)
    # SLP polish at unit cost
    D = 0.05
    for it in range(400):
        U, W = unpackv(v, m, n, p)
        g, J = gaps_and_jac(U, W, p, n, idx, y)
        c, gc = cost_grad(v, m, n, p, b)
        K, nv = J.shape
        # vars (d, t): max t s.t. -J d + t <= g ; gc.d <= 1 - c ; |d| <= D
        A_ub = np.vstack([np.hstack([-J, np.ones((K, 1))]), np.concatenate([gc, [0]])[None, :]])
        b_ub = np.concatenate([g, [1 - c]])
        res = linprog(np.concatenate([np.zeros(nv), [-1]]), A_ub=A_ub, b_ub=b_ub,
                      bounds=[(-D, D)] * nv + [(None, None)], method='highs')
        d = res.x[:nv]
        vn = v + d; cn, _ = cost_grad(vn, m, n, p, b); vn = vn * cn ** (-1 / b)
        Un, Wn = unpackv(vn, m, n, p)
        hn = margin(Un, Wn); h = g.min()
        if hn > h:
            v = vn
        else:
            D *= 0.5
        if D < 1e-11: break
    # final LP at tiny trust region to read off duals
    U, W = unpackv(v, m, n, p)
    g, J = gaps_and_jac(U, W, p, n, idx, y)
    c, gc = cost_grad(v, m, n, p, b)
    return v, g, J, c, gc


def als_spectral_lower(Gt, n, p, iters=300, starts=20, rng=None):
    """lower bound for the spectral norm of the (n+1)-way tensor Gt (shape (p,)*(n+1)) by alternating maximisation."""
    best = 0
    for st in range(starts):
        vs = [rng.standard_normal(p) for _ in range(n + 1)]; vs = [x / np.linalg.norm(x) for x in vs]
        for it in range(iters):
            for k in range(n + 1):
                T = Gt
                # contract all but k
                for q in reversed(range(n + 1)):
                    if q == k: continue
                    T = np.tensordot(T, vs[q], axes=([q], [0])) if False else T
                # simple explicit contraction
                T = Gt.copy()
                order = [q for q in range(n + 1) if q != k]
                for q in sorted(order, reverse=True):
                    T = np.tensordot(T, vs[q], axes=([q], [0]))
                vs[k] = T / np.linalg.norm(T)
        val = Gt
        for q in reversed(range(n + 1)):
            val = np.tensordot(val, vs[q], axes=([q], [0]))
        best = max(best, abs(float(val)))
    return best


for (p, n, b, m, seed) in [(5, 2, 2.0, 6, 1), (5, 2, 2.5, 7, 2), (7, 2, 2.2, 8, 3), (5, 3, 2.4, 8, 4), (5, 2, 2.0, 9, 5),
                           (7, 2, 2.6, 6, 6)]:
    nu = n + 1; s = nu / b; r = 1 / s
    srng = np.random.default_rng(seed)
    v, g, J, c, gc = find_local_max(p, n, b, m, srng)
    U, W = unpackv(v, m, n, p)
    h = g.min()
    act = g <= h + 1e-7 * abs(h)
    # KKT multipliers by NNLS on active set: J_act^T q = lam * gc, sum q = 1
    from scipy.optimize import nnls
    Ja = J[act]
    Amat = np.vstack([np.hstack([Ja.T, -gc[:, None]]), np.concatenate([np.ones(act.sum()), [0]])[None, :] * 1e3])
    bvec = np.concatenate([np.zeros(J.shape[1]), [1e3]])
    sol, rn = nnls(Amat, bvec, maxiter=50000)
    q = sol[:-1]; lam = sol[-1]
    stat = np.linalg.norm(Ja.T @ q - lam * gc) / np.linalg.norm(lam * gc)
    a = acost(U, W); keep = a > 1e-8 * a.max()
    # t_i: E_q g of neuron i alone with unit factors
    t = []
    for i in np.nonzero(keep)[0]:
        Ui = U[i:i + 1] / np.linalg.norm(U[i], axis=1)[None, :, None]; Wi = W[i:i + 1] / np.linalg.norm(W[i])
        gi = gaps(table(Ui, Wi), p, n).ravel()
        t.append((q * gi[act]).sum())
    t = np.array(t); ak = a[keep]
    pred = h * nu ** (nu / 2) * ak ** (r - 1)
    w = ak ** r / (ak ** r).sum()
    Fv = np.log(h) + (s - 1) * shannon(w)          # unit cost: log gamma = log h
    # spectral norm lower bound of E_q g as a tensor in (u_1..u_n, w)
    Gt = np.zeros((p,) * (nu))
    idx, y = setup(p, n); kk = 0; qa = np.zeros(len(g)); qa[act] = q
    for xi in range(p ** n):
        for d in range(1, p):
            if qa[kk] > 0:
                c1 = y[xi]; c2 = (y[xi] - d) % p
                Gt[tuple(idx[xi]) + (c1,)] += qa[kk]; Gt[tuple(idx[xi]) + (c2,)] -= qa[kk]
            kk += 1
    sig = als_spectral_lower(Gt, n, p, iters=60, starts=10, rng=rng)
    sizes = ak / ak.max()
    print('p=%d n=%d b=%.2f m=%d: h=%.4e gamma/gIF? ; active=%d ; stationarity rel resid %.1e ; max|t/pred-1| = %.2e ;'
          ' F - [E_w log t - nu/2 log nu] = %.1e ; neuron size ratios min %.3f ; t/sigma_lb in [%.4f, %.4f]'
          % (p, n, b, m, h, act.sum(), stat, np.max(np.abs(t / pred - 1)), Fv - ((w * np.log(t)).sum() - nu / 2 * np.log(nu)),
             sizes.min(), t.min() / sig, t.max() / sig), flush=True)
