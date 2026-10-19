# E_width library: general product networks on Z_p^n (any inputs, outputs on all frequencies).
#   neuron i: inputs U[i,t,:] (t=0..n-1), output W[i,:];  f[x,c] = sum_i W[i,c] prod_t U[i,t,x_t]
#   gap g(x,d) = f[x,y] - f[x,y-d];  h = min g;  cost P_b = sum_i ||omega_i||^b;  gamma(s) = h / P_b^s, s = nu/b
import numpy as np, itertools, math

LET = 'abcdefgh'


def Bp(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def kappa(p):
    F = (p - 1) // 2
    d = np.arange(1, p)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * j / p)          # rows delta=1..p-1, cols j=1..F


class Net:
    def __init__(self, U, W):
        self.U = np.array(U, float); self.W = np.array(W, float)
        self.m, self.n, self.p = self.U.shape
        self.nu = self.n + 1
        p, n = self.p, self.n
        idx = np.indices((p,) * n).reshape(n, -1).T
        self.X = idx
        self.Y = idx.sum(1) % p
        self.N = p ** n
        self.ein = ','.join('i' + LET[t] for t in range(n)) + ',i' + LET[n] + '->' + LET[:n + 1]

    def table(self):
        return np.einsum(self.ein, *[self.U[:, t] for t in range(self.n)], self.W, optimize=True)

    def gaps(self, f=None):
        p = self.p
        f = self.table() if f is None else f
        ff = f.reshape(self.N, p); r = np.arange(self.N)
        top = ff[r, self.Y]
        return np.stack([top - ff[r, (self.Y - d) % p] for d in range(1, p)], 1)  # (N, p-1)

    def margin(self):
        return self.gaps().min()

    def sq(self):  # ||omega_i||^2
        return (self.U ** 2).sum((1, 2)) + (self.W ** 2).sum(1)

    def mass(self):  # a_i = ||omega_i||^nu
        return self.sq() ** (self.nu / 2)

    def cost(self, b):
        return np.sum(self.sq() ** (b / 2))

    def gamma(self, s, h=None):
        b = self.nu / s
        h = self.margin() if h is None else h
        return h / self.cost(b) ** s

    # ---- Fourier data
    def clean(self):
        p, n, nu = self.p, self.n, self.nu; F = (p - 1) // 2; fh = np.fft.fftn(self.table())
        return np.array([2 * p ** (-nu) * fh[(j,) * n + ((-j) % p,)] for j in range(1, F + 1)])

    def signflip(self):
        p, n, nu = self.p, self.n, self.nu; F = (p - 1) // 2; fh = np.fft.fftn(self.table()); out = {}
        for j in range(1, F + 1):
            for T in itertools.product([0, 1], repeat=n):
                k = tuple(((-j) if T[t] else j) % p for t in range(n)) + ((-j) % p,)
                out[(j, T)] = 2 * p ** (-nu) * fh[k]
        return out

    def atoms(self):
        """beta_{ij} (m,F) and xi_{itj} (m,n,F) with beta_{ij,T} = beta_ij prod_{t in T} xi_itj (paper Lemma 6)."""
        p, nu = self.p, self.nu; F = (p - 1) // 2
        Uh = np.fft.fft(self.U, axis=2); Wh = np.fft.fft(self.W, axis=1)
        js = np.arange(1, F + 1)
        beta = 2 * p ** (-nu) * np.conj(Wh[:, js]) * np.prod(Uh[:, :, js], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = np.conj(Uh[:, :, js]) / Uh[:, :, js]
        xi = np.where(np.isfinite(xi), xi, 1.0)
        return beta, xi

    def fields(self):
        """phi[l-1, chi] = E_x Phi_l(x) rho^{-<chi,x>}, Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-lc}."""
        p, n = self.p, self.n; F = (p - 1) // 2
        f = self.table()
        fc = np.fft.fft(f, axis=n)                      # sum_c f[x,c] e^{-2 pi i l c/p}
        y = (np.indices((p,) * n).sum(0)) % p
        out = []
        for l in range(1, F + 1):
            Phi = (2.0 / p) * np.exp(2j * np.pi * l * y / p) * fc[..., l]
            out.append(np.fft.fftn(Phi) / p ** n)
        return np.array(out)


# ---------- entropies
def shannon(w):
    w = np.asarray(w, float); w = w[w > 0]; w = w / w.sum()
    return float(-(w * np.log(w)).sum())


def renyi(w, a):
    w = np.asarray(w, float); w = w[w > 0]; w = w / w.sum()
    if abs(a - 1) < 1e-14: return shannon(w)
    return float(np.log((w ** a).sum()) / (1 - a))


# ---------- chain matrices of one frequency's atom system
def chain_vectors(xi):
    """xi (N, n) unimodular -> m (N, nu) with m_l = (1, xi_l1, xi_l1 xi_l2, ...)."""
    N, n = xi.shape
    return np.concatenate([np.ones((N, 1), complex), np.cumprod(xi, axis=1)], 1)


def chain_matrix(beta, xi):
    """paper's H = sum_l beta_l m_l m_l^* (non-Hermitian)."""
    m = chain_vectors(xi)
    return np.einsum('l,la,lb->ab', beta, m, np.conj(m))


def twisted_matrix(beta, xi):
    """K = sum_l rho_l u_l u_l^*, u_{l,t} = p_l^t prod_{u<=t} xi_lu (Hermitian PSD); p_l = beta_l/|beta_l|."""
    rho = np.abs(beta); ph = np.where(rho > 0, beta / np.where(rho > 0, rho, 1), 1)
    m = chain_vectors(xi); nu = m.shape[1]
    u = m * ph[:, None] ** np.arange(nu)[None, :]
    return np.einsum('l,la,lb->ab', rho, u, np.conj(u))


def spec_chain(beta, xi):
    lam = np.linalg.eigvals(chain_matrix(beta, xi))
    return lam


def spec_twisted(beta, xi):
    return np.clip(np.linalg.eigvalsh(twisted_matrix(beta, xi)), 0, None)


# ---------- programme by exact vertex enumeration (small p)
def programme_vertices(p, s, S=None):
    """Gamma^{(S)}_s(p) = max min_delta sum_{j in S} y_j kappa_j(delta) s.t. sum y^{1/s} <= 1.
    Equivalent: min sum v^r over {v>=0 on S : K_S v >= 1}; concave => attained at a vertex. Returns
    (Gamma, list of maximisers y (full F vectors))."""
    F = (p - 1) // 2; r = 1.0 / s
    S = list(range(1, F + 1)) if S is None else list(S)
    K = kappa(p)[:F][:, [j - 1 for j in S]]            # delta = 1..F rows (delta, -delta identical)
    k = len(S)
    # constraints: K v >= 1 (F rows), v >= 0 (k rows); vertex = k linearly independent active constraints
    A = np.vstack([K, np.eye(k)]); bvec = np.concatenate([np.ones(F), np.zeros(k)])
    best = -1; sols = []
    for act in itertools.combinations(range(F + k), k):
        M = A[list(act)]
        if abs(np.linalg.det(M)) < 1e-12: continue
        v = np.linalg.solve(M, bvec[list(act)])
        if np.any(v < -1e-12) or np.any(K @ v < 1 - 1e-10): continue
        v = np.clip(v, 0, None)
        val = np.sum(v[v > 0] ** r) ** (-s)
        if val > best + 1e-12:
            best = val; sols = [v]
        elif abs(val - best) <= 1e-12:
            sols.append(v)
    ys = []
    for v in sols:
        y = np.zeros(F); y[[j - 1 for j in S]] = v / np.sum(v[v > 0] ** r) ** s
        if not any(np.allclose(y, z) for z in ys): ys.append(y)
    return best, ys


def zshares(y, s):
    r = 1.0 / s; z = np.where(y > 0, y, 0) ** r; return z / z.sum()


def if_block(p, n, j, lam=1.0, phase=0.0):
    """Paper Prop. 9 block for frequency j: nu neurons, alpha_l = pi l/nu."""
    nu = n + 1; a = np.arange(p)
    U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
    for l in range(nu):
        al = np.pi * l / nu
        for t in range(n):
            U[l, t] = lam * np.cos(2 * np.pi * j * a / p + al + (phase if t == 0 else 0.0))
        W[l] = lam * np.cos(2 * np.pi * j * a / p + n * al + phase)
    return U, W


def if_network(p, n, y, scale=1.0):
    """IF network with one block per j in spt(y), lambda_j^nu = 2^n y_j / nu (paper Prop. 9)."""
    nu = n + 1; Us = []; Ws = []
    for j, yj in enumerate(y, start=1):
        if yj <= 0: continue
        lam = (2 ** n * yj / nu) ** (1.0 / nu)
        U, W = if_block(p, n, j, lam)
        Us.append(U); Ws.append(W)
    return np.concatenate(Us) * scale, np.concatenate(Ws) * scale
