# Independent referee library (does not import elib.py or global/lib.py).
import numpy as np, itertools

def nu_of(n): return n + 1

def Bp(p, n):
    nu = n + 1; return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)

def table(U, W):
    # U: (m, n, p), W: (m, p) -> f[x1..xn, c]
    m, n, p = U.shape
    f = W.copy()  # (m, p)
    out = np.zeros((p,) * n + (p,))
    for i in range(m):
        T = W[i]
        for t in reversed(range(n)):
            T = np.multiply.outer(U[i, t], T)
        out += T
    return out

def labels(p, n):
    idx = np.indices((p,) * n).reshape(n, -1).T
    return idx.sum(1) % p

def gaps(f, p, n):
    ff = f.reshape(-1, p); Y = labels(p, n); r = np.arange(ff.shape[0])
    top = ff[r, Y]
    return np.stack([top - ff[r, (Y - d) % p] for d in range(1, p)], 1)  # (N, p-1), d = 1..p-1

def Phi(f, p, n, l):
    ff = f.reshape(-1, p); Y = labels(p, n); rho = np.exp(2j * np.pi / p)
    c = np.arange(p)
    return (2.0 / p) * rho ** (l * Y) * (ff * rho ** (-l * c)[None, :]).sum(1)

def Gl(f, p, n, l):
    ph = Phi(f, p, n, l); rho = np.exp(2j * np.pi / p)
    d = np.arange(1, p)
    return np.real(ph[:, None] * (1 - rho ** (-l * d))[None, :])

def beta(f, p, n):
    nu = n + 1; F = (p - 1) // 2; fh = np.fft.fftn(f)
    return np.array([2 * p ** (-nu) * fh[(j,) * n + ((-j) % p,)] for j in range(1, F + 1)])

def neuron_sq(U, W, mu=None, p=None):
    if mu is None:
        return (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
    p = W.shape[1]
    s = (U ** 2).sum((1, 2)) + ((W @ proj(p, 0)) ** 2).sum(1)
    for l in range(1, (p - 1) // 2 + 1):
        s = s + mu[l - 1] * ((W @ proj(p, l)) ** 2).sum(1)
    return s

def cost(U, W, b, mu=None):
    return np.sum(neuron_sq(U, W, mu) ** (b / 2))

def proj(p, k):
    a = np.arange(p)
    if k == 0: return np.full((p, p), 1.0 / p)
    c = np.cos(2 * np.pi * k * a / p); s = np.sin(2 * np.pi * k * a / p)
    return np.outer(c, c) / (c @ c) + np.outer(s, s) / (s @ s)

def shares(U, W, b):
    a = neuron_sq(U, W) ** (b / 2); return a / a.sum()

def H(w):
    w = np.asarray(w, float); w = w[w > 0]; w = w / w.sum(); return float(-(w * np.log(w)).sum())

def Evec(U, W, b):
    p = W.shape[1]; F = (p - 1) // 2; w = shares(U, W, b); sq = neuron_sq(U, W)
    return np.array([np.sum(w * ((W @ proj(p, l)) ** 2).sum(1) / np.where(sq > 0, sq, 1)) for l in range(1, F + 1)])

def kap_rows(p, lam=None, S=None):
    F = (p - 1) // 2
    d = np.arange(1, F + 1)[:, None]; j = np.arange(1, F + 1)[None, :]
    K = 1 - np.cos(2 * np.pi * d * j / p)
    if lam is not None: K = K * np.asarray(lam)[None, :]
    if S is not None:
        mask = np.zeros(F, bool); mask[[j - 1 for j in S]] = True; K = K * mask[None, :]
    return K

def m_lam(y, p, lam=None):
    K = kap_rows(p, lam); return float((K @ y).min())

def programme(p, s, lam=None, S=None, tol=1e-10):
    """Own vertex enumeration of 1/Gamma = min ||v||_{1/s} over {v>=0, K v >= 1, spt v in S}.
    Returns Gamma and the list of distinct maximisers y (normalised sum y^{1/s} = 1)."""
    F = (p - 1) // 2; r = 1.0 / s
    S = list(range(1, F + 1)) if S is None else list(S)
    K = kap_rows(p, lam)[:, [j - 1 for j in S]]
    k = len(S); vals = []
    # vertex: support T (subset of S cols), |T| tight rows; solve K[R,T] v_T = 1
    for t in range(1, k + 1):
        for T in itertools.combinations(range(k), t):
            KT = K[:, T]
            for R in itertools.combinations(range(F), t):
                A = KT[list(R)]
                if abs(np.linalg.det(A)) < 1e-12: continue
                v = np.linalg.solve(A, np.ones(t))
                if np.any(v <= tol): continue
                if np.any(KT @ v < 1 - 1e-9): continue
                val = np.sum(v ** r) ** (-s)
                y = np.zeros(F); y[[S[q] - 1 for q in T]] = v / np.sum(v ** r) ** s
                vals.append((val, y))
    vals.sort(key=lambda z: -z[0]); G = vals[0][0]; M = []
    for val, y in vals:
        if val >= G * (1 - 1e-9) and not any(np.allclose(y, z, atol=1e-7) for z in M): M.append(y)
    return G, M

def zsh(y, s):
    z = np.where(y > 0, y ** (1.0 / s), 0.0); return z / z.sum()
