# Independent referee library (does NOT import cclib/ccf/elib).
# Network: U (m, n, p) inputs, W (m, p) outputs; f[x, c] = sum_i W[i,c] prod_t U[i,t,x_t]
import itertools
import numpy as np

LET = 'abcdefgh'


def table(U, W):
    m, n, p = U.shape
    ein = ','.join('i' + LET[t] for t in range(n)) + ',i' + LET[n] + '->' + LET[:n + 1]
    return np.einsum(ein, *[U[:, t] for t in range(n)], W, optimize=True)


def gaps(U, W, f=None):
    """g[x_flat, d-1] = f[x, y] - f[x, y-d], d = 1..p-1."""
    m, n, p = U.shape
    f = table(U, W) if f is None else f
    ff = f.reshape(p ** n, p)
    X = np.indices((p,) * n).reshape(n, -1).T
    Y = X.sum(1) % p
    r = np.arange(p ** n)
    top = ff[r, Y]
    return np.stack([top - ff[r, (Y - d) % p] for d in range(1, p)], 1)


def sq(U, W):
    return (U ** 2).sum((1, 2)) + (W ** 2).sum(1)


def Bp(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def kap(p):
    """K[d-1, l-1] = 1 - cos(2 pi l d / p), d = 1..p-1, l = 1..F."""
    F = (p - 1) // 2
    d = np.arange(1, p)[:, None]; l = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * l / p)


def Mcov(v, p):
    return float((kap(p) @ np.asarray(v, float)).min())


def clean(f, p, n):
    """beta_l = 2 p^{-nu} sum_{x,c} f[x,c] exp(-2 pi i (l sum x - l c)/p), l = 1..F."""
    nu = n + 1; F = (p - 1) // 2
    X = np.indices((p,) * (n + 1)).reshape(n + 1, -1).T
    ff = f.reshape(-1)
    out = []
    for l in range(1, F + 1):
        ph = np.exp(-2j * np.pi * l * (X[:, :n].sum(1) - X[:, n]) / p)
        out.append(2 * p ** (-nu) * (ff * ph).sum())
    return np.array(out)


def signflip(f, p, n):
    """beta_{l,T}: input index -l on T, l elsewhere, output -l."""
    nu = n + 1; F = (p - 1) // 2
    X = np.indices((p,) * (n + 1)).reshape(n + 1, -1).T
    ff = f.reshape(-1); out = {}
    for l in range(1, F + 1):
        for T in itertools.product([0, 1], repeat=n):
            sg = np.array([-1 if T[t] else 1 for t in range(n)])
            ph = np.exp(-2j * np.pi * l * ((X[:, :n] * sg).sum(1) - X[:, n]) / p)
            out[(l, T)] = 2 * p ** (-nu) * (ff * ph).sum()
    return out


def Phi(f, p, n):
    """Phi[l-1, x_flat] = (2/p) rho^{l y} sum_c f[x,c] rho^{-lc}."""
    F = (p - 1) // 2
    ff = f.reshape(p ** n, p)
    X = np.indices((p,) * n).reshape(n, -1).T; Y = X.sum(1) % p
    c = np.arange(p)
    return np.array([(2.0 / p) * np.exp(2j * np.pi * l * Y / p) * (ff * np.exp(-2j * np.pi * l * c / p)[None, :]).sum(1)
                     for l in range(1, F + 1)])


def renyi(w, a):
    w = np.asarray(w, float); w = w[w > 0]; w = w / w.sum()
    if abs(a - 1) < 1e-14:
        return float(-(w * np.log(w)).sum())
    return float(np.log((w ** a).sum()) / (1 - a))


def rand_classC(p, n, widths, rng, amp=(0.3, 1.5), scale=None):
    """widths: dict l -> count; neurons purely at frequency l (no DC)."""
    a = np.arange(p); Us, Ws, lab = [], [], []
    for l, w in widths.items():
        for k in range(w):
            A = rng.uniform(*amp, n + 1) * (scale[l] if scale else 1.0)
            ph = rng.uniform(0, 2 * np.pi, n + 1)
            Us.append([A[t] * np.cos(2 * np.pi * l * a / p + ph[t]) for t in range(n)])
            Ws.append(A[n] * np.cos(2 * np.pi * l * a / p + ph[n]))
            lab.append(l)
    return np.array(Us), np.array(Ws), np.array(lab)


def if_block(p, n, l, lam=1.0):
    """nu neurons, phases alpha_k = pi k / nu on every input, n alpha_k on output (paper Prop 9)."""
    nu = n + 1; a = np.arange(p)
    U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
    for k in range(nu):
        al = np.pi * k / nu
        for t in range(n):
            U[k, t] = lam * np.cos(2 * np.pi * l * a / p + al)
        W[k] = lam * np.cos(2 * np.pi * l * a / p + n * al)
    return U, W


def programme(p, s, S=None):
    """Gamma^{(S)}_s by vertex enumeration of {v >= 0 on S: K_S v >= 1} minimising sum v^r (concave)."""
    F = (p - 1) // 2; r = 1.0 / s
    S = list(range(1, F + 1)) if S is None else list(S)
    K = kap(p)[:F][:, [l - 1 for l in S]]
    k = len(S)
    A = np.vstack([K, np.eye(k)]); bb = np.concatenate([np.ones(F), np.zeros(k)])
    best = np.inf; arg = None
    for act in itertools.combinations(range(F + k), k):
        Mm = A[list(act)]
        if abs(np.linalg.det(Mm)) < 1e-12: continue
        v = np.linalg.solve(Mm, bb[list(act)])
        if np.any(v < -1e-12) or np.any(K @ v < 1 - 1e-10): continue
        v = np.clip(v, 0, None); val = np.sum(v[v > 0] ** r)
        if val < best - 1e-13:
            best = val; arg = v
    y = np.zeros(F); y[[l - 1 for l in S]] = arg / best ** s
    return best ** (-s), y
