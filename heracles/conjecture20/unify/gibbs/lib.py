# Shared numerics for the GIBBS angle (unify/gibbs).  Conventions follow BRIEF.md / mechanism_note.tex.
# A network: U (m x n x p real inputs), W (m x p real outputs).  Logit f(x)[c] = sum_i W[i,c] prod_t U[i,t,x_t].
import numpy as np, itertools, json, os

def rho(p): return np.exp(2j*np.pi/p)

def table(U, W):
    """logit table f[x_1,...,x_n,c] (shape (p,)*n + (p,))."""
    m, n, p = U.shape
    f = np.zeros((p,)*n + (p,))
    for i in range(m):
        T = W[i]
        for t in reversed(range(n)):
            T = np.multiply.outer(U[i, t], T)
        f += T
    return f

def gaps(f):
    """g[x_1..x_n, delta] for delta=0..p-1 (delta=0 column is 0): g = f(x)[y] - f(x)[y-delta]."""
    n = f.ndim - 1; p = f.shape[0]
    idx = np.indices((p,)*n).reshape(n, -1)
    y = idx.sum(0) % p
    F = f.reshape(-1, p)
    rows = np.arange(F.shape[0])
    g = np.empty((F.shape[0], p))
    fy = F[rows, y]
    for d in range(p):
        g[:, d] = fy - F[rows, (y - d) % p]
    return g.reshape((p,)*n + (p,))

def margin(f):
    g = gaps(f)
    return g[..., 1:].min()

def fields(f):
    """Phi[l-1][x] for l in F=1..(p-1)/2:  Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-lc}."""
    n = f.ndim - 1; p = f.shape[0]; Fn = (p - 1)//2
    idx = np.indices((p,)*n).reshape(n, -1); y = idx.sum(0) % p
    F = f.reshape(-1, p)
    fh = np.fft.fft(F, axis=1)          # sum_c f[c] rho^{-kc}
    out = []
    for l in range(1, Fn + 1):
        out.append(((2.0/p) * rho(p)**(l*y) * fh[:, l]).reshape((p,)*n))
    return out

def coeffs(f):
    """clean beta_l and sign-flipped beta_{l,T} via FFT; returns dict (l, T tuple of 0/1) -> complex."""
    n = f.ndim - 1; p = f.shape[0]; Fn = (p - 1)//2
    Fh = np.fft.fftn(f)
    out = {}
    for l in range(1, Fn + 1):
        for T in itertools.product([0, 1], repeat=n):
            k = tuple(((-l) if T[t] else l) % p for t in range(n)) + ((-l) % p,)
            out[(l, T)] = 2 * p**(-(n+1)) * Fh[k]
    return out

def neuron_atoms(U, W):
    """per neuron i and l: beta_il = 2 p^-nu conj(what_i(l)) prod_t uhat_it(l), and xi_itl = conj(uhat)/uhat."""
    m, n, p = U.shape; Fn = (p - 1)//2; nu = n + 1
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    beta = np.zeros((m, Fn), complex); xi = np.ones((m, n, Fn), complex)
    for l in range(1, Fn + 1):
        prod = np.prod(Uh[:, :, l], axis=1)
        beta[:, l-1] = 2 * p**(-nu) * np.conj(Wh[:, l]) * prod
        u = Uh[:, :, l]
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(np.abs(u) > 0, np.conj(u)/u, 1.0)
        xi[:, :, l-1] = x
    return beta, xi

def cost(U, W, b):
    return sum((np.sum(U[i]**2) + np.sum(W[i]**2))**(b/2) for i in range(len(W)))

def Bp(p, nu): return 2 * (2*p*nu)**(-nu/2)

def kappa(l, d, p): return 1 - np.cos(2*np.pi*l*d/p)

def Kmat(p):
    Fn = (p - 1)//2; d = np.arange(1, p); j = np.arange(1, Fn + 1)
    return 1 - np.cos(2*np.pi*np.outer(d, j)/p)   # rows delta=1..p-1, cols l

def IF_prices(U, W, s):
    """a_l = nu^{1-s} Q_l^s, Q_l = sum_i |beta_il|^r."""
    m, n, p = U.shape; nu = n + 1; r = 1/s
    beta, _ = neuron_atoms(U, W)
    Q = np.sum(np.abs(beta)**r, axis=0)
    return nu**(1 - s) * Q**s, Q, beta

def if_block(p, n, j, lam):
    """paper Prop 9 block for frequency j: nu neurons, table nu lam^nu 2^-n cos(2 pi j (sum a - c)/p)."""
    nu = n + 1; a = np.arange(p)
    U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
    for l in range(nu):
        al = np.pi*l/nu
        for t in range(n): U[l, t] = lam*np.cos(2*np.pi*j*a/p + al)
        W[l] = lam*np.cos(2*np.pi*j*a/p + n*al)
    return U, W

HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.join(HERE, '..', '..', 'mechanism_scripts', 'relations3', 'supports.json')

def load_supports():
    with open(SUPP) as fh: return json.load(fh)

def programme_value(y, p):
    """min_delta sum_l y_l kappa_l(delta) for y indexed by l=1..F (array of length F)."""
    return (Kmat(p) @ y).min()
