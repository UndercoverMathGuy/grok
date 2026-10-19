"""Value of the chain-matrix relaxation R(p,n,b; mode) by local search with restarts.

R = sup  h(a, free) / ( nu^{1-s} Gamma_s(p) (sum_j nu^{-r} min_U ||a_{j,0} I + L_j + U_j||_{S_r}^r)^s ).
mode 'free': every non-own Fourier coefficient of every field is free (all compensation / DC / mixed modes free).
mode 'C'   : class C (non-own coefficients zero).
Usage: python relax_opt.py p n b mode restarts seed [support...]
"""
import sys, time, json
import numpy as np
from scipy.optimize import minimize
from mrel import Model, chain_matrix, schatten_r_grad, upper_index, subsets, interval
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
from prog import solve as prog_solve


class Relax:
    def __init__(self, p, n, s, mode, S=None):
        self.p, self.n, self.s = p, n, s
        self.r = 1 / s; self.nu = n + 1
        F = (p - 1) // 2
        self.S = list(S) if S else list(range(1, F + 1))
        self.M = Model(p, n, self.S, mode=mode, s=s)
        self.Ts = subsets(n)
        self.up = upper_index(self.nu)
        self.G, self.ystar, _ = prog_solve(p, s, restarts=8)
        self.dim_a = 2 * len(self.Ts)
        self.dim_u = 2 * len(self.up)
        self.dim = len(self.S) * (self.dim_a + self.dim_u)

    def unpack(self, th):
        a = {}; U = {}
        k = 0
        for j in self.S:
            v = th[k:k + self.dim_a]; k += self.dim_a
            a[j] = {T: v[2 * i] + 1j * v[2 * i + 1] for i, T in enumerate(self.Ts)}
            w = th[k:k + self.dim_u]; k += self.dim_u
            U[j] = w[0::2] + 1j * w[1::2]
        return a, U

    def pack(self, a, U):
        out = []
        for j in self.S:
            for T in self.Ts:
                out += [a[j][T].real, a[j][T].imag]
            for z in U[j]:
                out += [z.real, z.imag]
        return np.array(out)

    def cost(self, a, U, grad=False):
        tot = 0.0; g = {}
        for j in self.S:
            H = chain_matrix(a[j], self.n, U[j])
            v, G = schatten_r_grad(H, self.r)
            tot += v
            if grad:
                g[j] = G
        return (tot, g) if grad else tot

    def ratio(self, a, U):
        h, _ = self.M.margin(a)
        c = self.cost(a, U)
        q = self.nu ** (-self.r) * c
        return h / (self.nu ** (1 - self.s) * self.G * q ** self.s)

    def fobj(self, th):
        a, U = self.unpack(th)
        h, gh = self.M.margin(a)
        if h is None or h <= 1e-12:
            return 1e3, np.zeros_like(th)
        c, gc = self.cost(a, U, grad=True)
        val = -np.log(h) + self.s * np.log(c)
        # gradient
        grad = np.zeros_like(th)
        k = 0
        for j in self.S:
            G = gc[j]
            for i, T in enumerate(self.Ts):
                gam_h = gh.get((j, T), 0j)
                # cost complex gradient wrt a_T
                if T == ():
                    gam_c = np.trace(G)
                else:
                    # T must be an interval (t', t]
                    gam_c = 0j
                    if T == tuple(range(T[0], T[-1] + 1)):
                        tp, t = T[0], T[-1] + 1
                        gam_c = G[t, tp]
                gam = -gam_h / h + self.s * gam_c / c
                grad[k + 2 * i] = gam.real; grad[k + 2 * i + 1] = gam.imag
            k += self.dim_a
            for i, (t, tp) in enumerate(self.up):
                gam = self.s * G[t, tp] / c
                grad[k + 2 * i] = gam.real; grad[k + 2 * i + 1] = gam.imag
            k += self.dim_u
        return val, grad

    def start_if(self, noise=0.0, rng=None, ufill=None):
        a = {}; U = {}
        F = (self.p - 1) // 2
        for j in self.S:
            y = self.ystar[j - 1]
            a[j] = {T: (y if T == () else 0.0) + 0j for T in self.Ts}
            U[j] = np.zeros(len(self.up), dtype=complex)
            if noise and rng is not None:
                for T in self.Ts:
                    a[j][T] += noise * (y + 0.05) * (rng.normal() + 1j * rng.normal())
                U[j] += noise * (y + 0.05) * (rng.normal(size=len(self.up)) + 1j * rng.normal(size=len(self.up)))
        return a, U

    def start_atoms(self, rng, natoms=1, spread=1.0):
        """each feature realised by a few random atoms; U = mirror completion"""
        a = {}; U = {}
        for j in self.S:
            y = self.ystar[j - 1] + 0.05 * rng.uniform()
            H = np.zeros((self.nu, self.nu), dtype=complex)
            beta = (y / natoms) * np.exp(1j * spread * rng.normal(size=natoms)) * rng.uniform(0.5, 1.5, natoms)
            xi = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(natoms, self.n)))
            for b, x in zip(beta, xi):
                mv = np.concatenate([[1], np.cumprod(x)])
                H += np.conj(b) * np.outer(np.conj(mv), mv)
            a[j] = {}
            for T in self.Ts:
                bT = sum(b * np.prod([x[t] for t in T]) for b, x in zip(beta, xi))
                a[j][T] = np.conj(bT)
            U[j] = np.array([H[t, tp] for (t, tp) in self.up])
        return a, U

    def optimise(self, a, U, maxiter=400):
        th0 = self.pack(a, U)
        res = minimize(self.fobj, th0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'maxcor': 30})
        a, U = self.unpack(res.x)
        return self.ratio(a, U), a, U


def main():
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    mode = sys.argv[4]; restarts = int(sys.argv[5]); seed = int(sys.argv[6])
    S = [int(v) for v in sys.argv[7:]] or None
    s = (n + 1) / b
    R = Relax(p, n, s, mode, S)
    rng = np.random.default_rng(seed)
    print(f"p={p} n={n} b={b} s={s:.4f} mode={mode} S={R.S} Gamma={R.G:.6f} y*={np.round(R.ystar,4)}", flush=True)
    a, U = R.start_if()
    print(f"  IF start ratio = {R.ratio(a, U):.6f}", flush=True)
    best = (0, None)
    t0 = time.time()
    for k in range(restarts):
        kind = k % 4
        if kind == 0:
            a, U = R.start_if(noise=0.3, rng=rng)
        elif kind == 1:
            a, U = R.start_atoms(rng, natoms=1)
        elif kind == 2:
            a, U = R.start_atoms(rng, natoms=2, spread=0.3)
        else:
            a, U = R.start_if(noise=1.0, rng=rng)
        try:
            val, a2, U2 = R.optimise(a, U)
        except Exception as e:
            print("  fail", e); continue
        if val > best[0]:
            best = (val, (a2, U2))
        print(f"  restart {k} kind {kind}: ratio {val:.6f}   best {best[0]:.6f}   ({time.time()-t0:.0f}s)", flush=True)
    val, (a, U) = best
    out = {j: {str(T): [a[j][T].real, a[j][T].imag] for T in R.Ts} for j in R.S}
    outU = {j: [[z.real, z.imag] for z in U[j]] for j in R.S}
    fn = f"relax_p{p}_n{n}_b{b}_{mode}_s{seed}.json"
    json.dump({'ratio': val, 'a': out, 'U': outU}, open(fn, 'w'), indent=1)
    print(f"BEST ratio {val:.6f}; saved {fn}")


if __name__ == '__main__':
    main()
