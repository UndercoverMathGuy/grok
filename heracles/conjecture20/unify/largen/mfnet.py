# Slot-symmetric product networks, arbitrary inputs u_i in R^p, outputs w_i in R^p on ALL frequencies, any b.
#   f(N)[c] = sum_i e^{l_i} w~_i(c) prod_a u~_i(a)^{N_a}   (u~, w~ unit vectors, N = count vector)
#   cost ||omega_i||^2 = nu e^{2 l_i/nu} (balanced),  P = sum_i (nu e^{2 l_i/nu})^{b/2},  gamma = h / P^{nu/b}
#   ratio = gamma / gamma_IF,  gamma_IF = nu^{1-s} B_p Gamma_s(p).
# Clean amplitudes beta_j = sum_i e^{l_i} 2 p^{-nu} conj(w^_i(j)) u^_i(j)^n ; we report eta_j = |beta_j| / (B_p P^{nu/b} nu^{1-s})
# (eta_j = 1 for every frequency of an interference-free block at the IF price).
import sys, math, itertools, json, time
import numpy as onp
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from gamma import Gamma

def counts(n, p):
    out = []
    def rec(prefix, left, k):
        if k == p - 1:
            out.append(prefix + [left]); return
        for v in range(left + 1):
            rec(prefix + [v], left - v, k + 1)
    rec([], n, 0)
    N = onp.array(out, dtype=float)
    y = (N @ onp.arange(p)).astype(int) % p
    return N, y

class MF:
    def __init__(self, p, n, b=2.0):
        self.p, self.n, self.b, self.nu = p, n, b, n + 1
        self.s = self.nu / b
        self.N, self.y = counts(n, p)
        K = len(self.y)
        # selection matrices for gaps: g[k,d] = f[k,y_k] - f[k,(y_k-d)%p], d=1..p-1
        self.Iy = onp.zeros((K, p)); self.Iy[onp.arange(K), self.y] = 1
        self.Id = [onp.zeros((K, p)) for d in range(1, p)]
        for d in range(1, p):
            self.Id[d - 1][onp.arange(K), (self.y - d) % p] = 1
        self.G = Gamma(p, self.s)
        self.Bp = 2 * (2 * p * self.nu) ** (-self.nu / 2)
        self.lgif = (1 - self.s) * math.log(self.nu) + math.log(self.Bp) + math.log(self.G)
        a = onp.arange(p)
        self.F = onp.exp(-2j * onp.pi * onp.outer(onp.arange(p), a) / p)   # DFT rows j: sum_a v(a) rho^{-ja}

    def unpack(self, z, m):
        p = self.p
        U = z[:m * p].reshape(m, p); W = z[m * p:2 * m * p].reshape(m, p); l = z[2 * m * p:2 * m * p + m]
        return U, W, l

    def gaps(self, z, m):
        p, n, nu, b = self.p, self.n, self.nu, self.b
        U, W, l = self.unpack(z, m)
        l = l - np.max(l)                      # scale invariance: centre the log-masses
        Un = U / np.sqrt(np.sum(U ** 2, axis=1, keepdims=True))
        Wn = W / np.sqrt(np.sum(W ** 2, axis=1, keepdims=True))
        logabs = np.log(np.abs(Un) + 1e-300)
        par = onp.dot(self.N, (onp.asarray(getattr(Un, '_value', Un)) < 0).T)   # parity counts (constant)
        sgn = 1 - 2 * onp.remainder(par, 2)
        lc = (b / 2) * (math.log(nu) + (2 / nu) * l)
        logP = np.log(np.sum(np.exp(lc - np.max(lc)))) + np.max(lc)
        lognorm = (nu / b) * logP + self.lgif
        Z = sgn * np.exp(np.dot(self.N, logabs.T) + l[None, :] - lognorm)     # K x m
        f = np.dot(Z, Wn)                                                      # K x p
        fy = np.sum(f * self.Iy, axis=1)
        g = np.concatenate([fy - np.sum(f * Id, axis=1) for Id in self.Id])
        return g

    def ratio(self, z, m):
        return float(onp.min(self.gaps(z, m)))

    def clean(self, z, m):
        p, n, nu, b = self.p, self.n, self.nu, self.b
        U, W, l = self.unpack(onp.asarray(z), m)
        l = l - l.max()
        Un = U / onp.linalg.norm(U, axis=1, keepdims=True); Wn = W / onp.linalg.norm(W, axis=1, keepdims=True)
        uh = Un @ self.F.T; wh = Wn @ self.F.T                    # m x p  (index j)
        lc = (b / 2) * (math.log(nu) + (2 / nu) * l); P = onp.exp(lc).sum()
        out = {}
        for j in range(1, (p - 1) // 2 + 1):
            beta = onp.sum(onp.exp(l) * 2 * p ** (-nu) * onp.conj(wh[:, j]) * uh[:, j] ** n)
            out[j] = abs(beta) / (self.Bp * P ** (nu / b) * nu ** (1 - self.s))
        return out

def seed(mf, m, kind, rng, eps=0.0):
    p, n, nu = mf.p, mf.n, mf.nu
    a = onp.arange(p)
    if kind == 'if':
        t = onp.pi * onp.arange(m) / nu
        U = onp.array([onp.cos(2 * onp.pi * a / p + ti) for ti in t])
        W = onp.array([onp.cos(2 * onp.pi * a / p + n * ti) for ti in t])
        l = onp.zeros(m)
        if m > nu: l[nu:] = -6.0
    elif kind == 'oned':
        e = onp.where(a < p / 2, 2 * a, 2 * a - p).astype(float); eps_s = onp.where(a < p / 2, 1.0, -1.0)
        lam = onp.linspace(-0.6, 0.6, m) * rng.uniform(0.6, 1.4)
        U = onp.array([eps_s * onp.exp(li * e) for li in lam])
        ph = rng.uniform(0, 2 * onp.pi, m)
        W = onp.array([onp.cos(2 * onp.pi * a / p + phi) + 0.05 * onp.cos(4 * onp.pi * a / p + rng.uniform(0, 6.3)) for phi in ph])
        l = onp.zeros(m)
    else:
        U = rng.standard_normal((m, p)); W = rng.standard_normal((m, p)); l = rng.uniform(-0.5, 0.5, m)
    U = U + eps * rng.standard_normal(U.shape); W = W + eps * rng.standard_normal(W.shape)
    return onp.concatenate([U.ravel(), W.ravel(), l])

def optimise(mf, z0, m, taus=(0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4), iters=600):
    z = z0.copy()
    for tau in taus:
        def obj(zz):
            g = mf.gaps(zz, m)
            return tau * np.log(np.sum(np.exp(-(g - np.min(g)) / tau))) - np.min(g)
        go = grad(obj)
        def fun(zz):
            v = obj(zz)
            return float(v), onp.asarray(go(zz), dtype=float)
        try:
            r = minimize(fun, z, jac=True, method='L-BFGS-B', options=dict(maxiter=iters))
            z = r.x
        except Exception as ex:
            pass
    return mf.ratio(z, m), z

if __name__ == "__main__":
    p = int(sys.argv[1]); ns = [int(v) for v in sys.argv[2].split(',')]
    kinds = sys.argv[3].split(',') if len(sys.argv) > 3 else ['if', 'oned', 'rand']
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    bval = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    rng = onp.random.default_rng(11)
    for n in ns:
        mf = MF(p, n, bval); nu = n + 1; t0 = time.time()
        res = []
        for kind in kinds:
            widths = {'if': [nu, nu + 2], 'oned': [nu - nu // p, nu], 'rand': [nu - 1, nu + 2]}[kind]
            for m in widths:
                for rep in range(reps):
                    eps = {'if': 0.15, 'oned': 0.02, 'rand': 0.0}[kind] * (1 + rep)
                    z0 = seed(mf, m, kind, rng, eps)
                    v, z = optimise(mf, z0, m)
                    cl = mf.clean(z, m)
                    res.append((v, kind, m, cl))
        res.sort(key=lambda t: -t[0])
        print(f"p={p} n={n} b={bval} K*-programme Gamma={mf.G:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        for v, kind, m, cl in res:
            top = max(cl.values())
            print(f"   ratio={v:+.5f}  seed={kind:5s} m={m:2d}  eta_j/IFprice: " +
                  " ".join(f"{j}:{cl[j]:.2e}" for j in cl) + f"  spurious/main={sorted(cl.values())[-2]/top if len(cl)>1 else 0:.2e}", flush=True)
