# Global search over general slot-symmetric networks (all output frequencies): CMA over inputs U, exact SOCP over
# outputs with l_r reweighting; reports the efficiency gamma/gamma_IF and the clean-amplitude support.
import math, sys, time, numpy as np, cma, warnings, cvxpy as cp
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from famp import counts, log_gamma_if
warnings.filterwarnings("ignore")

class Prob:
    def __init__(self, p, n, b=2.0):
        self.p, self.n, self.b, self.nu = p, n, b, n + 1
        N, y, mult = counts(n, p); self.N, self.y, self.mult = N.numpy(), y.numpy(), mult.numpy()
        cs = np.arange(p); Bs = []
        for l in range(1, (p - 1) // 2 + 1):
            for f in (np.cos, np.sin):
                v = f(2 * math.pi * l * cs / p); Bs.append(v / np.linalg.norm(v))
        self.B = np.array(Bs).T                                   # p x (p-1), orthonormal zero-mean basis
        self.D = [self.B[self.y] - self.B[(self.y - d) % p] for d in range(1, p)]
        self.lgif = log_gamma_if(p, n, b)

    def table(self, U):
        Un = U / np.linalg.norm(U, axis=1, keepdims=True)
        logP = self.N @ np.log(np.abs(Un) + 1e-300).T
        sgn = (-1.0) ** (self.N @ (Un < 0).T)
        return sgn, logP

    def solve(self, U, rew=3):
        p, nu = self.p, self.nu; r = self.b / nu; m = len(U)
        sgn, logP = self.table(U)
        rowmax = logP.max(1); G = sgn * np.exp(logP - rowmax[:, None]); sc = np.exp(-rowmax); sc = sc / sc.max()
        A = np.vstack([(G[:, :, None] * D[:, None, :]).reshape(len(self.y), -1) for D in self.D])
        cc = np.concatenate([sc] * (p - 1))
        Z = cp.Variable((m, p - 1)); t = cp.Variable(m); h = cp.Variable(); w = cp.Parameter(m, nonneg=True)
        cons = [A @ cp.reshape(Z, (m * (p - 1),), order='C') >= h * cc, cp.norm(Z, 2, axis=1) <= t, w @ t <= 1]
        prob = cp.Problem(cp.Maximize(h), cons)
        wv = np.ones(m); best = (-np.inf, None)
        for it in range(rew + 1):
            w.value = wv
            try: prob.solve(solver=cp.CLARABEL)
            except Exception: break
            if Z.value is None: break
            Zv = Z.value; W = Zv @ self.B.T
            f = (G * np.exp(rowmax[:, None])) @ W
            gaps = np.array([f[np.arange(len(self.y)), self.y] - f[np.arange(len(self.y)), (self.y - d) % p] for d in range(1, p)])
            htrue = gaps.min(); a = np.linalg.norm(Zv, axis=1)
            if htrue > 0 and a.max() > 0:
                keep = a > 1e-9 * a.max(); P = (nu * a[keep] ** (2 / nu)).sum()
                e = math.exp(math.log(htrue) - (nu / 2) * math.log(P) - self.lgif)
                if e > best[0]: best = (e, (U, W, keep))
            wv = (a / max(a.max(), 1e-300) + 1e-4) ** (r - 1); wv = wv / wv.min()
        return best

    def amplitudes(self, U, W):
        # clean amplitudes A_j = |beta_j|, beta_j = 2 p^{-(n+1)} sum_{x,c} f(x)[c] rho^{-j(y - c)}
        p = self.p; sgn, logP = self.table(U)
        f = (sgn * np.exp(logP)) @ W
        out = []
        for j in range(1, (p - 1) // 2 + 1):
            ph = np.exp(-2j * math.pi * j * (self.y[:, None] - np.arange(p)[None, :]) / p)
            out.append(abs(2 * p ** (-(self.n + 1)) * (self.mult[:, None] * f * ph).sum()))
        return np.array(out)

def run(p, n, m, R=6, iters=80, seed=0):
    pr = Prob(p, n); rng = np.random.default_rng(seed); best = (-1, None)
    a = np.arange(p)
    for r in range(R):
        if r == 0:   # IF-like start on random frequencies
            js = rng.integers(1, (p - 1) // 2 + 1, m)
            x0 = np.array([np.cos(2 * math.pi * j * a / p + rng.uniform(0, math.pi)) for j in js]).ravel()
        else:
            x0 = rng.standard_normal(m * p)
        es = cma.CMAEvolutionStrategy(x0, 0.5, {'seed': seed + r + 1, 'verbose': -9, 'popsize': 12 + m, 'maxiter': iters})
        while not es.stop():
            X = es.ask(); F = []
            for x in X:
                v = pr.solve(np.array(x).reshape(m, p), rew=1); F.append(v[0] if v[0] > -np.inf else -1.0)
                if v[0] > best[0]: best = v
            es.tell(X, [-f for f in F])
    if best[1] is not None:
        v = pr.solve(best[1][0], rew=6)
        if v[0] > best[0]: best = v
    return best, pr

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    for m in [int(v) for v in sys.argv[3].split(",")]:
        best, pr = run(p, n, m)
        if best[1] is None:
            print("p=%d n=%d m=%d: no positive margin found (%.0fs)" % (p, n, m, time.time() - t0), flush=True); continue
        U, W, keep = best[1]
        A = pr.amplitudes(U[keep], W[keep])
        sh = A ** (2 / (n + 1)); sh = sh / sh.sum()
        print("p=%d n=%d m=%d: best eff %.5f  active neurons %d  clean amplitude shares (A^r normalised) %s (%.0fs)"
              % (p, n, m, best[0], keep.sum(), np.round(sh, 4), time.time() - t0), flush=True)
        np.save("ge_p%d_n%d_m%d.npy" % (p, n, m), np.concatenate([U[keep], W[keep]], 1))
