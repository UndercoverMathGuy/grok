# Independent referee library (does NOT import sclib / global lib).
import numpy as np, itertools, math

def Bp(p, n):
    nu = n + 1; return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)

class Net:
    def __init__(self, U, W, b):
        self.U = np.asarray(U, float); self.W = np.asarray(W, float)
        self.m, self.n, self.p = self.U.shape; self.b = b; self.nu = self.n + 1; self.s = self.nu / b
        p, n = self.p, self.n
        self.X = np.indices((p,) * n).reshape(n, -1).T            # (N, n)
        self.Y = self.X.sum(1) % p; self.N = p ** n
    def table(self):
        # f[x, c] = sum_i W[i,c] prod_t U[i,t,x_t]
        P = np.ones((self.m, self.N))
        for t in range(self.n): P = P * self.U[:, t, self.X[:, t]]
        return P.T @ self.W                                        # (N, p)
    def gaps(self, f=None):
        f = self.table() if f is None else f; p = self.p; r = np.arange(self.N)
        top = f[r, self.Y]
        return np.stack([top - f[r, (self.Y - d) % p] for d in range(1, p)], 1)   # delta = 1..p-1
    def masses(self):
        return ((self.U ** 2).sum((1, 2)) + (self.W ** 2).sum(1)) ** (self.nu / 2)
    def cost(self):
        return (((self.U ** 2).sum((1, 2)) + (self.W ** 2).sum(1)) ** (self.b / 2)).sum()
    def gamma(self):
        return self.gaps().min() / self.cost() ** self.s
    def fields(self):
        p = self.p; F = (p - 1) // 2; f = self.table(); c = np.arange(p)
        return np.stack([(2.0 / p) * np.exp(2j * np.pi * l * self.Y / p) * (f @ np.exp(-2j * np.pi * l * c / p)) for l in range(1, F + 1)], 1)
    def atoms(self):
        p, nu = self.p, self.nu; F = (p - 1) // 2; js = np.arange(1, F + 1)
        Uh = np.fft.fft(self.U, axis=2); Wh = np.fft.fft(self.W, axis=1)
        return 2 * p ** (-nu) * np.conj(Wh[:, js]) * np.prod(Uh[:, :, js], axis=1)    # (m, F)
    def signflip_atoms(self, l, T):
        """beta_{il,T}: input index -l on slots in T (T a 0/1 tuple)."""
        p, nu = self.p, self.nu
        Uh = np.fft.fft(self.U, axis=2); Wh = np.fft.fft(self.W, axis=1)
        out = 2 * p ** (-nu) * np.conj(Wh[:, l])
        for t in range(self.n): out = out * Uh[:, t, (-l) % p if T[t] else l]
        return out

def kappa(p):
    F = (p - 1) // 2; d = np.arange(1, p)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * j / p)                     # (p-1, F)

def cvec(p):
    F = (p - 1) // 2; d = np.arange(1, p)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.exp(-2j * np.pi * d * j / p)                   # c_l(delta)

def renyi(m, a):
    m = np.asarray(m, float); m = m[m > 0]; m = m / m.sum()
    if abs(a - 1) < 1e-13: return float(-(m * np.log(m)).sum())
    return float(np.log((m ** a).sum()) / (1 - a))

def programme_S(p, s, S):
    """exact Gamma^{(S)}_s by vertex enumeration of {v>=0, spt in S, Kv>=1} (small p)."""
    F = (p - 1) // 2; r = 1 / s
    K = kappa(p)[:F][:, [l - 1 for l in S]]; k = len(S); best = 0
    rows = [('r', i) for i in range(F)] + [('c', j) for j in range(k)]
    for comb in itertools.combinations(rows, k):
        A = np.zeros((k, k)); bb = np.zeros(k)
        for q, (typ, idx) in enumerate(comb):
            if typ == 'r': A[q] = K[idx]; bb[q] = 1
            else: A[q, idx] = 1
        if abs(np.linalg.det(A)) < 1e-12: continue
        v = np.linalg.solve(A, bb)
        if np.any(v < -1e-10) or np.any(K @ v < 1 - 1e-10): continue
        v = np.clip(v, 0, None); best = max(best, np.sum(v[v > 0] ** r) ** (-s))
    return best

def classC(p, n, specs, b):
    """specs: list of (l, amps[nu], phis[n], psi): u_t = amp_t cos(2pi l x/p + phi_t), w = amp_n cos(2pi l c/p + psi).
       atom = 2^{1-nu} prod amps * exp(i(sum phi - psi)); xi_t = exp(-2 i phi_t)."""
    x = np.arange(p); U = []; W = []
    for (l, amps, phis, psi) in specs:
        U.append([amps[t] * np.cos(2 * np.pi * l * x / p + phis[t]) for t in range(n)])
        W.append(amps[n] * np.cos(2 * np.pi * l * x / p + psi))
    return Net(np.array(U), np.array(W), b)
