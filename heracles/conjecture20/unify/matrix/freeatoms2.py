"""Scale-invariant search for the atomic free relaxation R_atom(S') (see freeatoms.py).
Objective: maximise ratio = h(beta, xi)/Q^s directly (no dead zone for h<0).  Optional slot-symmetric atoms (xi_it = xi_i).
Usage: python freeatoms2.py p n b restarts seed natoms symmetric(0/1) [S'...]
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
from freeatoms import FA

class FA2(FA):
    def __init__(self, p, n, s, S, natoms, sym):
        super().__init__(p, n, s, S, {j: natoms for j in S})
        self.sym = sym
    def expand(self, th):
        if not self.sym:
            return th
        out = []; k = 0
        for j in self.S:
            m = self.na[j]
            out += list(th[k:k + 2 * m]); k += 2 * m
            ph = th[k:k + m]; k += m
            out += list(np.repeat(ph, self.n))
        return np.array(out)
    def contract_grad(self, g):
        if not self.sym:
            return g
        out = []; k = 0
        for j in self.S:
            m = self.na[j]
            out += list(g[k:k + 2 * m]); k += 2 * m
            gp = g[k:k + m * self.n].reshape(m, self.n).sum(1); k += m * self.n
            out += list(gp)
        return np.array(out)
    def f(self, th):
        full = self.expand(th)
        out = self.ratio_and_grad(full)
        if out is None:
            return 10.0, np.zeros_like(th)
        ratio, h, Q, g = out
        # g is grad of -log h + s log Q ; we need grad of -ratio = -(h/Q^s)/(norm) ...
        # reconstruct: grad(-log h + s log Q) = -grad h/h + s grad Q/Q ; ratio = c h Q^{-s}
        # grad(-ratio) = -ratio*( grad h/h - s grad Q/Q ) = ratio * g   (valid for h != 0)
        return -ratio, self.contract_grad(ratio * g)
    def opt(self, th, iters=500):
        res = minimize(self.f, th, jac=True, method='L-BFGS-B', options={'maxiter': iters, 'maxcor': 30})
        return -res.fun, res.x

def run(p, n, b, restarts, seed, natoms, sym, S=None, verbose=True):
    s = (n + 1) / b; F = (p - 1) // 2; S = S or list(range(1, F + 1))
    A = FA2(p, n, s, S, natoms, sym)
    rng = np.random.default_rng(seed)
    best = (-np.inf, None); t0 = time.time()
    dimper = 2 * natoms + (natoms if sym else natoms * n)
    for k in range(restarts):
        th = []
        for j in S:
            y = max(A.ystar[j - 1], 0.03)
            beta = y / natoms * (rng.normal(size=natoms) + 1j * rng.normal(size=natoms) + (1.5 if k % 2 else 0))
            ph = rng.uniform(0, 2 * np.pi, natoms if sym else natoms * n)
            th += list(beta.real) + list(beta.imag) + list(ph)
        th = np.array(th)
        try:
            val, th2 = A.opt(th)
        except Exception as e:
            continue
        if val > best[0]:
            best = (val, th2)
        if verbose and (k % 10 == 9 or val > 1.0 + 1e-6):
            print(f"  restart {k}: {val:.6f}   best {best[0]:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    return A, best

if __name__ == '__main__':
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    restarts, seed, natoms, sym = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    S = [int(v) for v in sys.argv[8:]] or None
    print(f"R_atom search p={p} n={n} b={b} natoms={natoms} sym={sym} S'={S}", flush=True)
    A, (val, th) = run(p, n, b, restarts, seed, natoms, sym, S)
    np.save(f"fa2_p{p}_n{n}_b{b}_m{natoms}_sym{sym}_s{seed}.npy", th)
    print(f"BEST {val:.6f}")
