# E_kkt library: general product networks on Z_p^n, KKT (Fritz-John) measures, critical values, injective norm.
#   neuron i: inputs U[i,t,:] in R^p (t=0..n-1), output W[i,:] in R^p
#   table f[x_1..x_n, c] = sum_i W[i,c] prod_t U[i,t,x_t];  label y(x) = sum_t x_t mod p
#   gap g(x,delta) = f[x,y] - f[x,y-delta]; h = min g; P = sum_i ||omega_i||^b; gamma = h / P^s, s = nu/b.
import numpy as np, itertools, math, json
from scipy.optimize import minimize, nnls, linprog

LET = 'abcdefghjklmnopqrstuvwxy'   # neuron index letter is 'z'


def Bp(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def kappa_mat(p):
    F = (p - 1) // 2
    d = np.arange(1, p)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * j / p)          # rows delta = 1..p-1, cols j = 1..F


def programme_exact(p, s):
    """Gamma_s(p) by enumeration of vertices of the coverage polyhedron {v >= 0 : K v >= 1} (small p).
    Gamma_s = max over vertices v of (sum v^{1/s})^{-s}.  Returns (Gamma, list of maximisers y (normalised), their z)."""
    F = (p - 1) // 2; r = 1.0 / s
    K = kappa_mat(p)[:F]                                    # offsets 1..F suffice (kappa(-d) = kappa(d))
    best = -1; sols = []
    # a vertex: choose a support S and a set of F-|S| ... use LP over all supports with reweighting is fuzzy;
    # here: enumerate basic solutions of K_S v_S = 1 on |S| tight rows.
    for k in range(1, F + 1):
        for S in itertools.combinations(range(F), k):
            for R in itertools.combinations(range(F), k):
                M = K[np.ix_(R, S)]
                if abs(np.linalg.det(M)) < 1e-12: continue
                v = np.linalg.solve(M, np.ones(k))
                if np.any(v <= 1e-12): continue
                full = np.zeros(F); full[list(S)] = v
                if np.any(K @ full < 1 - 1e-9): continue
                val = np.sum(v ** r) ** (-s)
                if val > best + 1e-10:
                    best = val; sols = [full]
                elif abs(val - best) <= 1e-10:
                    if not any(np.allclose(full, o) for o in sols): sols.append(full)
    ys = [v / np.sum(v ** r) ** s for v in sols]
    zs = [y ** r / np.sum(y ** r) for y in ys]
    return best, ys, zs


def entropy(w):
    w = np.asarray(w, float); w = w[w > 0]; w = w / w.sum()
    return float(-(w * np.log(w)).sum())


class Net:
    def __init__(self, U, W, b=2.0):
        self.U = np.array(U, float); self.W = np.array(W, float)
        self.m, self.n, self.p = self.U.shape
        self.b = b; self.nu = self.n + 1; self.s = self.nu / b
        p, n = self.p, self.n
        idx = np.indices((p,) * n).reshape(n, -1).T
        self.X = idx
        self.Y = idx.sum(1) % p
        self.N = p ** n

    def with_b(self, b):
        return Net(self.U, self.W, b)

    # ---- per-neuron input products U_i(x) = prod_t u_it(x_t), shape (m, N)
    def Uprod(self, U=None):
        U = self.U if U is None else U
        out = np.ones((U.shape[0], self.N))
        for t in range(self.n):
            out *= U[:, t, self.X[:, t]]
        return out

    def neuron_gaps(self, U=None, W=None):
        """(m, N, p-1): g_i(x, delta) = U_i(x) (w_i(y) - w_i(y - delta))."""
        U = self.U if U is None else U; W = self.W if W is None else W
        Up = self.Uprod(U); p = self.p
        top = W[:, self.Y]                                           # (m, N)
        out = np.empty((U.shape[0], self.N, p - 1))
        for d in range(1, p):
            out[:, :, d - 1] = Up * (top - W[:, (self.Y - d) % p])
        return out

    def gaps(self):
        return self.neuron_gaps().sum(0)

    def margin(self):
        return self.gaps().min()

    def sq(self):
        return (self.U ** 2).sum((1, 2)) + (self.W ** 2).sum(1)

    def cost(self):
        return np.sum(self.sq() ** (self.b / 2))

    def gamma(self):
        return self.margin() / self.cost() ** self.s

    def a(self):                       # a_i = ||omega_i||^nu
        return self.sq() ** (self.nu / 2)

    def shares(self, s=None):
        s = self.s if s is None else s
        c = self.a() ** (1.0 / s); return c / c.sum()

    def H(self, s=None):
        return entropy(self.shares(s))

    # ---- Fourier data
    def atoms(self):
        p, nu = self.p, self.nu; F = (p - 1) // 2
        Uh = np.fft.fft(self.U, axis=2); Wh = np.fft.fft(self.W, axis=1)
        js = np.arange(1, F + 1)
        return 2 * p ** (-nu) * np.conj(Wh[:, js]) * np.prod(Uh[:, :, js], axis=1)

    def clean(self):
        return self.atoms().sum(0)

    def signflip(self):
        p, n, nu = self.p, self.n, self.nu; F = (p - 1) // 2
        Uh = np.fft.fft(self.U, axis=2); Wh = np.fft.fft(self.W, axis=1)
        out = {}
        for j in range(1, F + 1):
            for T in itertools.product([0, 1], repeat=n):
                prod = np.conj(Wh[:, j]).copy()
                for t in range(n):
                    prod = prod * (np.conj(Uh[:, t, j]) if T[t] else Uh[:, t, j])
                out[(j, T)] = 2 * p ** (-nu) * prod.sum()
        return out

    # ---- gradients
    def grad_cost(self):
        S = self.sq(); c = self.b * S ** (self.b / 2 - 1)
        return c[:, None, None] * self.U, c[:, None] * self.W

    def jac_active(self, act):
        """Jacobian columns of g at flattened active indices act (k = x_index*(p-1) + (delta-1)).
        Returns array (dim, |act|), dim = m*n*p + m*p (U then W, C-order)."""
        p, n, m = self.p, self.n, self.m
        xs = act // (p - 1); ds = act % (p - 1) + 1
        X = self.X[xs]; y = self.Y[xs]; ym = (y - ds) % p
        dw = self.W[:, y] - self.W[:, ym]                            # (m, A)
        facs = np.stack([self.U[:, t, X[:, t]] for t in range(n)], 0)  # (n, m, A)
        A = len(act)
        gU = np.zeros((m, n, p, A)); gW = np.zeros((m, p, A))
        for t in range(n):
            others = np.prod(np.delete(facs, t, axis=0), axis=0) if n > 1 else np.ones((m, A))
            val = others * dw                                        # (m, A)
            for k in range(p):
                mask = (X[:, t] == k)
                gU[:, t, k, :] += val * mask[None, :]
        Up = np.prod(facs, axis=0)                                   # (m, A)
        for k in range(p):
            gW[:, k, :] += Up * ((y == k)[None, :]) - Up * ((ym == k)[None, :])
        return np.concatenate([gU.reshape(-1, A), gW.reshape(-1, A)], 0)


def pack(U, W): return np.concatenate([U.ravel(), W.ravel()])


def unpack(x, m, n, p):
    return x[:m * n * p].reshape(m, n, p), x[m * n * p:].reshape(m, p)


def grad_lin(net, lam):
    """gradient of sum_{x,delta} lam[x,delta] g(x,delta) (lam shape (N,p-1))."""
    p, n = net.p, net.n
    Lam = np.zeros((net.N, p)); r = np.arange(net.N)
    np.add.at(Lam, (r, net.Y), lam.sum(1))
    for d in range(1, p):
        np.add.at(Lam, (r, (net.Y - d) % p), -lam[:, d - 1])
    Lam = Lam.reshape((p,) * (n + 1))
    fac = [net.U[:, t] for t in range(n)] + [net.W]
    grads = []
    for k in range(n + 1):
        others = [fac[q] for q in range(n + 1) if q != k]
        subs = ','.join('z' + LET[q] for q in range(n + 1) if q != k)
        grads.append(np.einsum(subs + ',' + LET[:n + 1] + '->z' + LET[k], *others, Lam, optimize=True))
    return np.stack(grads[:n], 1), grads[n]


def softgamma(x, m, n, p, b, tau):
    U, W = unpack(x, m, n, p)
    net = Net(U, W, b); c = net.cost(); s = net.s
    g = net.gaps() * c ** (-s)
    gmin = g.min(); z = np.exp(-(g - gmin) / tau); Z = z.sum(); lam = z / Z
    Fv = gmin - tau * np.log(Z)
    gU, gW = grad_lin(net, lam)
    cU, cW = net.grad_cost()
    Eg = (lam * g).sum()
    dU = c ** (-s) * gU - s * Eg / c * cU
    dW = c ** (-s) * gW - s * Eg / c * cW
    return Fv, pack(dU, dW)


def optimise(net, taus=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6), iters=3000):
    m, n, p, b = net.m, net.n, net.p, net.b
    x = pack(net.U, net.W)
    scale = Bp(p, n) * (n + 1) ** (1 - (n + 1) / b)
    for tau in taus:
        t = tau * scale
        def fobj(z):
            Fv, gv = softgamma(z, m, n, p, b, t)
            return -Fv / scale, -gv / scale
        res = minimize(fobj, x, jac=True,
                       method='L-BFGS-B', options={'maxiter': iters, 'ftol': 1e-16, 'gtol': 1e-13})
        x = res.x
    U, W = unpack(x, m, n, p)
    out = Net(U, W, b)
    sc = out.cost() ** (-1.0 / b)
    return Net(U * sc, W * sc, b)


def balance(net):
    U = net.U.copy(); W = net.W.copy()
    for i in range(net.m):
        norms = np.concatenate([np.linalg.norm(U[i], axis=1), [np.linalg.norm(W[i])]])
        if np.any(norms == 0): continue
        g = np.exp(np.mean(np.log(norms)))
        U[i] *= (g / norms[:-1])[:, None]; W[i] *= g / norms[-1]
    return Net(U, W, net.b)


def prune(net, tol=1e-8):
    S = net.sq(); keep = S > tol * S.max()
    return Net(net.U[keep], net.W[keep], net.b)


def if_block(p, n, j, lam=1.0, phase0=0.0):
    nu = n + 1; a = np.arange(p); U = []; W = []
    for l in range(nu):
        al = np.pi * l / nu + phase0
        U.append([lam * np.cos(2 * np.pi * j * a / p + al) for t in range(n)])
        W.append(lam * np.cos(2 * np.pi * j * a / p + n * al))
    return np.array(U), np.array(W)


def if_network(p, n, y, b=2.0):
    nu = n + 1; Us = []; Ws = []
    for j, yj in enumerate(y, start=1):
        if yj <= 0: continue
        lam = (2 ** n * yj / nu) ** (1.0 / nu)
        U, W = if_block(p, n, j, lam); Us.append(U); Ws.append(W)
    return Net(np.concatenate(Us), np.concatenate(Ws), b)


def gamma_if(p, n, b, Gam):
    nu = n + 1; s = nu / b
    return nu ** (1 - s) * Bp(p, n) * Gam


def build1d(path):
    """slot-symmetric 1-D shared-sign network from a JSON file (Note's p=3 certified family)."""
    d = json.load(open(path)); p = d['p']; n = d['n']
    lam, lv, av = np.array(d['lam']), np.array(d['logV']), np.array(d['argV'])
    m = len(lam); lv = lv - lv.max()
    eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
    e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], dtype=float)
    rho = np.exp(2j * np.pi * np.arange(p) / p)
    U, W = [], []
    for i in range(m):
        u = eps * np.exp(lam[i] * e)
        w = np.real(np.exp(lv[i] + 1j * av[i]) * rho)
        nu_, nw = np.linalg.norm(u), np.linalg.norm(w)
        t = (nw * nu_ ** n) ** (1.0 / (n + 1))
        U.append([u / nu_ * t] * n); W.append(w / nw * t)
    return Net(np.array(U), np.array(W), 2.0), p, n


# ------------------------------------------------------------------ KKT machinery
def kkt_measure(net, rtol=1e-6, max_active=4000):
    """Fritz-John measure q on the (near-)active set: sum_a q_a grad g_a = (s h / P) grad P, q >= 0, sum q = 1.
    Returns (act indices, q, relative residual, h)."""
    g = net.gaps().ravel(); h = g.min()
    act = np.where(g <= h + rtol * abs(h))[0]
    if len(act) > max_active:
        act = act[np.argsort(g[act])[:max_active]]
    J = net.jac_active(act)
    cU, cW = net.grad_cost(); gP = pack(cU, cW)
    lam = net.s * h / net.cost()
    rhs = lam * gP
    Mw = 1e3 * np.linalg.norm(rhs)
    A = np.vstack([J, Mw * np.ones((1, len(act)))]); bb = np.concatenate([rhs, [Mw]])
    q, _ = nnls(A, bb, maxiter=50 * A.shape[1])
    res = np.linalg.norm(J @ q - rhs) / np.linalg.norm(rhs)
    return act, q, res, h


def q_full(net, act, q):
    Q = np.zeros(net.N * (net.p - 1)); Q[act] = q
    return Q.reshape(net.N, net.p - 1)


def dual_tensor(net, Q):
    """M_q[x_1..x_n, c] = sum_delta q(x,delta)([c=y] - [c=y-delta]); shape (p,)*(n+1)."""
    p, n = net.p, net.n
    M = np.zeros((net.N, p)); r = np.arange(net.N)
    np.add.at(M, (r, net.Y), Q.sum(1))
    for d in range(1, p):
        np.add.at(M, (r, (net.Y - d) % p), -Q[:, d - 1])
    return M.reshape((p,) * (n + 1))


def form_value(M, vecs):
    T = M
    for v in reversed(vecs):           # contract last axis first
        T = T @ v
    return float(T)


def contract_except(M, vecs, k):
    nu = M.ndim; T = M
    for t in range(nu - 1, -1, -1):
        if t == k: continue
        T = np.tensordot(T, vecs[t], axes=([t], [0]))
    return T


def critical_values(net, Q):
    M = dual_tensor(net, Q); out = []; resid = []
    for i in range(net.m):
        vecs = [net.U[i, t] / np.linalg.norm(net.U[i, t]) for t in range(net.n)] + [net.W[i] / np.linalg.norm(net.W[i])]
        tv = form_value(M, vecs); out.append(tv)
        rr = 0.0
        for k in range(net.nu):
            gk = contract_except(M, vecs, k)
            rr = max(rr, np.linalg.norm(gk - tv * vecs[k]) / max(abs(tv), 1e-300))
        resid.append(rr)
    return np.array(out), np.array(resid), M


def injective_norm(M, starts=200, iters=300, seed=0, init=()):
    """max of the multilinear form over unit vectors (alternating power method, random + given starts)."""
    rng = np.random.default_rng(seed); nu = M.ndim; p = M.shape[0]
    best = -np.inf; bestv = None
    cand = list(init) + [None] * starts
    for c in cand:
        vecs = [rng.standard_normal(p) for _ in range(nu)] if c is None else [v.copy() for v in c]
        vecs = [v / np.linalg.norm(v) for v in vecs]
        val = -np.inf
        for it in range(iters):
            for k in range(nu):
                gk = contract_except(M, vecs, k); ng = np.linalg.norm(gk)
                if ng == 0: break
                vecs[k] = gk / ng
            nv = form_value(M, vecs)
            if abs(nv - val) < 1e-15 * max(1, abs(nv)): val = nv; break
            val = nv
        if val > best: best = val; bestv = vecs
    return best, bestv
