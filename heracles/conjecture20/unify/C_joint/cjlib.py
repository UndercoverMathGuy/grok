# C_joint library.
#  * programme Gamma_s(p) (exact vertex enumeration for small p; reweighted LP otherwise)
#  * single-T ("1D") family TL_n: feature l in S carries an IF block of clean amplitude alpha_l and an
#    n-atom single-T block of clean amplitude beta_l and interference W_l = beta_l e^{i phi_l} on one set T.
#    Field Phi_l(x) = alpha_l + beta_l + beta_l e^{i phi_l} rho^{2 l x_T}; with w = 2 x_T in Z_p:
#      g(w, d) = sum_l (alpha_l+beta_l) kappa_l(d) + beta_l Re(e^{i phi_l} rho^{l w} (1 - rho^{-l d})).
#    Cost (B_p units, balanced class-C neurons): Q = nu^{1-r} sum alpha^r + n^{1-r} sum beta^r.
#    IF reference value nu^{1-s} Gamma_s Q^s; ratio R = h / (Gamma_s (sum alpha^r + lam beta^r)^s), lam=(n/nu)^{1-r}.
#  * explicit class-C networks (U, W tables) for independent verification with E_width/elib.Net.
import numpy as np, itertools, sys, os
from scipy.optimize import minimize, linprog

def kappa_full(p):
    F = (p-1)//2
    d = np.arange(1, p)[:, None]; l = np.arange(1, F+1)[None, :]
    return 1 - np.cos(2*np.pi*d*l/p)            # (p-1, F)

def gamma_exact(p, s, S=None):
    """Exact Gamma^{(S)}_s(p) by vertex enumeration of {v>=0, K v >= 1}; returns (Gamma, maximisers y)."""
    F = (p-1)//2; r = 1.0/s
    S = list(range(1, F+1)) if S is None else list(S)
    K = kappa_full(p)[:F][:, [j-1 for j in S]]
    k = len(S)
    A = np.vstack([K, np.eye(k)]); bvec = np.concatenate([np.ones(F), np.zeros(k)])
    best = -1; sols = []
    for act in itertools.combinations(range(F+k), k):
        M = A[list(act)]
        if abs(np.linalg.det(M)) < 1e-12: continue
        v = np.linalg.solve(M, bvec[list(act)])
        if np.any(v < -1e-12) or np.any(K @ v < 1-1e-10): continue
        v = np.clip(v, 0, None)
        val = np.sum(v[v > 0]**r)**(-s)
        if val > best*(1+1e-12): best = val; sols = [v]
        elif val > best*(1-1e-12): sols.append(v)
    ys = []
    for v in sols:
        y = np.zeros(F); y[[j-1 for j in S]] = v/np.sum(v[v > 0]**r)**s
        if not any(np.allclose(y, z) for z in ys): ys.append(y)
    return best, ys

def gamma_lp(p, s, restarts=30, seed=0, S=None):
    """Reweighted-LP local optima of the programme (lower bound for Gamma)."""
    F = (p-1)//2; r = 1.0/s; K = kappa_full(p)[:F]
    S = list(range(1, F+1)) if S is None else list(S); idx = [j-1 for j in S]
    Ks = K[:, idx]; rng = np.random.default_rng(seed); best = (np.inf, None)
    for t in range(restarts):
        w = rng.uniform(0.2, 2.0, len(idx)) if t else np.ones(len(idx))
        for it in range(200):
            res = linprog(w, A_ub=-Ks, b_ub=-np.ones(F), bounds=(0, None), method='highs')
            v = res.x; wn = (v+1e-12)**(r-1)
            if np.allclose(wn, w, rtol=1e-9): break
            w = wn
        Q = np.sum(v[v > 1e-12]**r)
        if Q < best[0]: best = (Q, v.copy())
    Q, v = best; y = np.zeros(F); y[idx] = np.where(v > 1e-12, v, 0)/Q**s
    return Q**(-s), y

# ---------------- single-T family
class TL:
    def __init__(self, p, n, s, S):
        self.p, self.n, self.s = p, n, s; self.nu = n+1; self.r = 1.0/s
        self.S = np.array(S); F = (p-1)//2
        self.kap = kappa_full(p)[:, self.S-1]                         # (p-1, k)
        w = np.arange(p)[:, None, None]; d = np.arange(1, p)[None, :, None]; l = self.S[None, None, :]
        rho = lambda t: np.exp(2j*np.pi*t/p)
        self.E = rho(l*w)*(1 - rho(-l*d))                               # (p, p-1, k)
        self.lam = (n/self.nu)**(1-self.r)
    def gaps(self, al, be, ph):
        return self.kap[None]@(al+be) + np.real(self.E*(be*np.exp(1j*ph))[None, None, :]).sum(2)
    def cost(self, al, be):
        return np.sum(al**self.r) + self.lam*np.sum(be**self.r)
    def ratio(self, al, be, ph, Gam):
        return self.gaps(al, be, ph).min()/(Gam*self.cost(al, be)**self.s)

def tl_optimise(tl, Gam, x0, taus=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4), iters=3000, ifonly=False):
    """maximise softmin(G)/C^s (degree-0 homogeneous) with a scale-anchoring penalty."""
    k = len(tl.S); r, s = tl.r, tl.s
    def unpack(x):
        xa = np.clip(x[:k], -40, 40); xb = np.clip(x[k:2*k], -40, 40)
        be = np.zeros(k) + 1e-300 if ifonly else np.exp(xb)
        return np.exp(xa), be, x[2*k:]
    def f(x, tau):
        al, be, ph = unpack(x)
        C = tl.cost(al, be)
        G = tl.gaps(al, be, ph); scale = C**s
        g = G/scale
        m = g.min(); e = np.exp(-(g-m)/tau); Z = e.sum(); sm = m - tau*np.log(Z)     # softmin of G/C^s
        q = e/Z
        qd = q.sum(0)
        Eph = tl.E*np.exp(1j*ph)[None, None, :]
        dC_al = r*al**(r-1); dC_be = (0*be if ifonly else tl.lam*r*be**(r-1))
        g_al = (qd@tl.kap)/scale - s*sm*dC_al/C
        g_be = ((qd@tl.kap) + np.einsum('wd,wdk->k', q, np.real(Eph)))/scale - s*sm*dC_be/C
        g_ph = np.einsum('wd,wdk->k', q, -be[None, None, :]*np.imag(Eph))/scale
        pen = 1e-3*np.log(C)**2; dpen = 2e-3*np.log(C)/C
        val = sm/Gam - pen
        grad = np.concatenate([(g_al/Gam - dpen*dC_al)*al, (g_be/Gam - dpen*dC_be)*be, g_ph/Gam])
        return -val, -grad
    x = x0.copy()
    for tau in taus:
        res = minimize(f, x, args=(tau,), jac=True, method='L-BFGS-B', options={'maxiter': iters})
        x = res.x
    al, be, ph = unpack(x)
    return tl.ratio(al, be, ph, Gam), (al, be, ph)

# ---------------- explicit class-C networks (for verification)
def ifblock_tables(p, n, j, amp, phase=0.0):
    """Paper Prop. 9 IF block at frequency j with clean amplitude |beta| = amp (B_p-free normalisation:
    returns U,W with clean coefficient magnitude amp, computed numerically)."""
    nu = n+1; a = np.arange(p); U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
    for i in range(nu):
        al = np.pi*i/nu
        for t in range(n): U[i, t] = np.cos(2*np.pi*j*a/p + al + (phase if t == 0 else 0.0))
        W[i] = np.cos(2*np.pi*j*a/p + n*al + phase)
    return U, W

def atom_tables(p, n, j, cs, xis):
    """Class-C neurons at frequency j realising clean atoms c_i (complex) and xi_it (unimodular):
    u_it(x) = cos(2 pi j x/p + th_it) with e^{2 i th_it} = xi_it;  w_i(c) = |c_i|-scaled cos(2 pi j c/p + psi_i).
    Field of neuron i at feature j: (2/p) what_i(j) prod_t u_it(x_t) rho^{j x_t} = c_i prod_t (1 + xi_it rho^{2 j x_t}) up to
    the global constant; we only need atoms up to a common positive factor, fixed by matching numerically."""
    m = len(cs); a = np.arange(p); U = np.zeros((m, n, p)); W = np.zeros((m, p))
    for i in range(m):
        th = np.angle(xis[i])/2
        for t in range(n): U[i, t] = np.cos(2*np.pi*j*a/p + th[t])
        # field phase: (2/p) what(j) prod_t (1/2) e^{-i th_t} (1 + xi rho^{2jx}); what(j) = sum_c w(c) rho^{-jc}
        # w(c) = A cos(2 pi j c/p + psi) -> what(j) = (p A/2) e^{i psi}
        # => c_i = A 2^{-n} e^{i(psi - sum th)} ; choose psi = arg c_i + sum th, A = 2^n |c_i|
        psi = np.angle(cs[i]) + th.sum(); A = 2**n*abs(cs[i])
        W[i] = A*np.cos(2*np.pi*j*a/p + psi)
    return U, W
