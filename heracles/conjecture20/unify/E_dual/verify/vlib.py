# Independent referee library (does not import the agent's elib / global lib).
# Network: neuron i has inputs U[i,t,:] (t=0..n-1) and output W[i,:]; f[x_1..x_n, c] = sum_i W[i,c] prod_t U[i,t,x_t].
import itertools, numpy as np

L = 'abcdefghij'


def table(U, W):
    m, n, p = U.shape
    subs = ','.join('z' + L[t] for t in range(n)) + ',z' + L[n] + '->' + L[:n + 1]
    return np.einsum(subs, *[U[:, t] for t in range(n)], W, optimize=True)


def gaps(f, p, n):
    idx = np.indices((p,) * n).reshape(n, -1).T
    y = idx.sum(1) % p
    ff = f.reshape(-1, p); r = np.arange(ff.shape[0])
    return np.stack([ff[r, y] - ff[r, (y - d) % p] for d in range(1, p)], 1)  # columns delta = 1..p-1


def margin(U, W):
    m, n, p = U.shape
    return gaps(table(U, W), p, n).min()


def acost(U, W):
    """a_i = ||omega_i||^nu"""
    nu = U.shape[1] + 1
    S = (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
    return S ** (nu / 2)


def clean(f, p, n):
    nu = n + 1; F = (p - 1) // 2; fh = np.fft.fftn(f)
    return np.array([2 * p ** (-nu) * fh[(j,) * n + ((-j) % p,)] for j in range(1, F + 1)])


def atoms(U, W):
    m, n, p = U.shape; nu = n + 1; F = (p - 1) // 2
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1); js = np.arange(1, F + 1)
    return 2 * p ** (-nu) * np.conj(Wh[:, js]) * np.prod(Uh[:, :, js], axis=1)


def Bp(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def kap(p):
    F = (p - 1) // 2
    d = np.arange(1, p)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * j / p)  # rows delta=1..p-1


def Mfun(yh, p):
    return (kap(p) @ yh).min()


def renyi(q, al):
    q = q[q > 0]; q = q / q.sum()
    if abs(al - 1) < 1e-14:
        return float(-(q * np.log(q)).sum())
    return float(np.log((q ** al).sum()) / (1 - al))


def shannon(q):
    return renyi(q, 1.0)


def programme_vertices(p, s, extra_rows=None):
    """Exact Gamma_s via vertex enumeration of {y>=0, K y >= 1, extra rows A y >= 0}: concave l_r minimisation
    attains its min at a vertex.  Returns (Gamma, best y)."""
    F = (p - 1) // 2; r = 1 / s
    K = kap(p)[:F]
    rows = [(K[d], 1.0) for d in range(F)] + [(np.eye(F)[j], 0.0) for j in range(F)]
    if extra_rows is not None:
        rows += [(a, 0.0) for a in extra_rows]
    best = (np.inf, None)
    for comb in itertools.combinations(range(len(rows)), F):
        A = np.array([rows[c][0] for c in comb]); b = np.array([rows[c][1] for c in comb])
        if abs(np.linalg.det(A)) < 1e-12: continue
        y = np.linalg.solve(A, b)
        if np.any(y < -1e-12): continue
        y = np.maximum(y, 0)
        if np.any(K @ y < 1 - 1e-10): continue
        if extra_rows is not None and np.any(np.array(extra_rows) @ y < -1e-12): continue
        Q = np.sum(y[y > 0] ** r)
        if Q < best[0]: best = (Q, y)
    Q, y = best
    return Q ** (-s), y


def if_block_net(p, n, y, scales=None, rng=None, noise=0.0, dc=0.0):
    """Union over features j (y_j > 0) of the KM2 block: nu neurons, inputs cos(2 pi j x/p - pi l/nu), output
    phase chosen so atoms are real positive.  Neuron norms scaled so beta_j proportional to y_j.  Optional
    per-neuron size factors 'scales' (breaks IF), Gaussian noise, and DC offsets."""
    nu = n + 1; F = (p - 1) // 2; x = np.arange(p)
    Us = []; Ws = []
    k = 0
    for j in range(1, F + 1):
        if y[j - 1] <= 0: continue
        for l in range(nu):
            ph = -np.pi * l / nu
            c = y[j - 1] ** (1.0 / nu)
            if scales is not None:
                c = c * scales[k] ** (1.0 / nu)
            u = np.array([np.cos(2 * np.pi * j * x / p + ph) for t in range(n)]) * c
            w = np.cos(2 * np.pi * j * x / p + n * ph) * c
            Us.append(u); Ws.append(w); k += 1
    U = np.array(Us); W = np.array(Ws)
    if rng is not None and noise > 0:
        sc = np.abs(U).mean()
        U = U + noise * sc * rng.standard_normal(U.shape); W = W + noise * sc * rng.standard_normal(W.shape)
    if rng is not None and dc > 0:
        U = U + dc * rng.standard_normal((U.shape[0], U.shape[1], 1)); W = W + dc * rng.standard_normal((W.shape[0], 1))
    return U, W
