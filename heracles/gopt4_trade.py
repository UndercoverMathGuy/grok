"""gopt4_trade.py -- does JUNK ever pay on a K >= 2 support?  (statement S2)

Setting.  Pure single-frequency neurons (the equality class of the per-neuron
AM-GM and Parseval steps, so the only place (T) can be attained), an arbitrary
number N_k of them per frequency, arbitrary phases, JUNK ALLOWED.  For such a
network, with  u_j^{(k)}(a) = exp(-2i * 2 pi k a_j / p),  theta_kD = 2 pi k D/p,

    h^{(k)}(u) = sum_l c_lk prod_j (1 + u_j v_lkj),
    2^n * margin = Lambda := min over a in Z_p^n, D != 0 of
                   sum_k Re[ (1 - e^{i theta_kD}) h^{(k)}(u^{(k)}(a)) ],
    P_q          = ((n+1) p / 2)^{q/2} * sum_{k,l} |c_lk|^r,      r = q/nu.

With sum_{k,l}|c_lk|^r normalised to 1 one gets exactly gamma_q = B * Lambda,
B = 2 (2 p (n+1))^{-nu/2}, so the target (T) is the single number

    ratio := Lambda / [ (n+1)^{-(1-r)/r} * Gamma_{nu/q}(p) ]   <=  1 ?

ratio = 1 exactly at the (n+1)-block on a maximiser of the programme.  This
script maximises ratio over junk-carrying networks: block starts, block + the
cost-optimal junk perturbation, and random starts.  Anything above 1 refutes (T).

All float64, local CPU.
"""
import json
import argparse
import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------ the programme
def gmat(p):
    k = np.arange(1, (p - 1) // 2 + 1)
    d = np.arange(1, p)
    return 1.0 - np.cos(2 * np.pi * np.outer(k, d) / p)     # (nf, p-1)


def gamma_s(p, r, starts=400, seed=0):
    """Gamma_{1/r}(p) = max { min_D sum_k x_k g_k(D) : x >= 0, sum_k x_k^r <= 1 }."""
    g = gmat(p)
    nf = g.shape[0]
    rng = np.random.default_rng(seed)

    def negval(z):
        x = z ** 2
        s = np.sum(x ** r)
        if s <= 0:
            return 0.0
        x = x / s ** (1.0 / r)                      # project onto sum x^r = 1
        return -float((g.T @ x).min())

    best = 0.0
    bestx = None
    for s in range(starts):
        if s < nf:
            z0 = np.zeros(nf)
            z0[s] = 1.0
        else:
            K = rng.integers(1, nf + 1)
            z0 = np.zeros(nf)
            z0[rng.choice(nf, K, replace=False)] = 1.0
            z0 = z0 * rng.uniform(0.5, 1.5, nf)
        res = minimize(negval, z0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
        res = minimize(negval, res.x, method='Powell',
                       options={'maxiter': 20000, 'xtol': 1e-13, 'ftol': 1e-15})
        if -res.fun > best:
            x = res.x ** 2
            x = x / np.sum(x ** r) ** (1.0 / r)
            best, bestx = -res.fun, x
    return best, bestx


# ------------------------------------------------------------ the network
class Net:
    def __init__(self, p, n, ks, N):
        self.p, self.n, self.ks, self.N = p, n, np.array(ks), N
        self.K = len(ks)
        r = np.arange(p ** n)
        self.ops = np.stack([(r // p ** j) % p for j in range(n)], 1)   # (p^n, n)
        # U[k, a, j] = exp(-2i*2pi*k*a_j/p)
        self.U = np.exp(-2j * 2 * np.pi * np.einsum('k,aj->kaj',
                                                    self.ks, self.ops) / p)
        D = np.arange(1, p)
        self.W = 1.0 - np.exp(1j * 2 * np.pi * np.outer(self.ks, D) / p)  # (K, p-1)

    def nvar(self):
        return 2 * self.K * self.N + self.n * self.K * self.N

    def split(self, z):
        K, N, n = self.K, self.N, self.n
        c = (z[:K * N] + 1j * z[K * N:2 * K * N]).reshape(K, N)
        v = np.exp(1j * z[2 * K * N:].reshape(K, N, n))
        return c, v

    def hvals(self, c, v):
        """H[k, a] = h^{(k)}(u^{(k)}(a))."""
        K, N, n = self.K, self.N, self.n
        prod = np.ones((K, self.p ** self.n, N), dtype=complex)
        for j in range(n):
            prod *= 1.0 + self.U[:, :, j][:, :, None] * v[:, None, :, j]
        return np.einsum('kal,kl->ka', prod, c)

    def Lam(self, z):
        c, v = self.split(z)
        H = self.hvals(c, v)                       # (K, p^n)
        M = np.einsum('kd,ka->ad', self.W, H).real  # (p^n, p-1)
        return float(M.min()), M

    def cost(self, z, r):
        c, _ = self.split(z)
        return float(np.sum(np.abs(c) ** r))

    def ratio(self, z, r, denom):
        L, _ = self.Lam(z)
        cst = self.cost(z, r)
        if cst <= 0:
            return -1e9
        return (L / cst ** (1.0 / r)) / denom      # Lambda is 1-homog in c? no:
        # Lambda is 1-homogeneous in c, cost^(1/r) is 1-homogeneous in c -> scale free


def block_z(net, shares, rng, jitter=0.0):
    """(n+1)-block: N >= n+1 atoms per frequency, first n+1 carry A_k/(n+1)."""
    K, N, n = net.K, net.N, net.n
    nb = n + 1
    cr = np.zeros((K, N))
    ci = np.zeros((K, N))
    phi = rng.uniform(0, 2 * np.pi, size=(K, N, n))
    for a in range(K):
        cr[a, :nb] = shares[a] / nb
        for l in range(nb):
            phi[a, l, :] = -2.0 * np.pi * l / nb
    z = np.concatenate([cr.ravel(), ci.ravel(), phi.ravel()])
    if jitter:
        z = z + jitter * rng.standard_normal(z.size)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int, nargs='+', default=[11, 13, 17])
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--q', type=float, default=2.0)
    ap.add_argument('--extra', type=int, default=1,
                    help='atoms per frequency beyond n+1')
    ap.add_argument('--starts', type=int, default=40)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', type=str,
                    default='heracles/FINAL_RUNS/GOPT4/trade.json')
    args = ap.parse_args()

    n = args.n
    nu = n + 1
    r = args.q / nu
    rng = np.random.default_rng(args.seed)
    out = []

    for p in args.p:
        G, xstar = gamma_s(p, r)
        S = [int(k) for k in np.nonzero(xstar > 1e-8)[0] + 1]
        shares_prog = xstar[np.array(S) - 1]
        denom = (n + 1) ** (-(1 - r) / r) * G
        net = Net(p, n, S, n + 1 + args.extra)
        # block shares: A_k such that sum_k (n+1)^{1-r} A_k^r = 1 and A propto xstar
        A = shares_prog * (n + 1) ** (-(1 - r) / r)
        z_blk = block_z(net, A, rng)
        rb = net.ratio(z_blk, r, denom)
        print(f"\np={p}  Gamma_{{1/r}}={G:.10f}  support={S} (K={len(S)})  "
              f"block ratio={rb:.12f}")

        best, bz = rb, z_blk
        for s in range(args.starts):
            if s < 12:
                z0 = block_z(net, A, rng, jitter=10.0 ** (-(s % 6) - 1))
            else:
                z0 = np.concatenate([
                    rng.standard_normal(net.K * net.N) / (net.K * net.N),
                    rng.standard_normal(net.K * net.N) / (net.K * net.N),
                    rng.uniform(0, 2 * np.pi, size=net.K * net.N * n)])
            # soft-min annealed ascent then exact polish
            z = z0.copy()
            for beta in (30.0, 100.0, 400.0, 2000.0):
                def neg(zz):
                    c, v = net.split(zz)
                    H = net.hvals(c, v)
                    M = np.einsum('kd,ka->ad', net.W, H).real
                    m0 = M.min()
                    soft = m0 - np.log(np.exp(-beta * (M - m0)).sum()) / beta
                    cst = np.sum((np.abs(c) ** 2 + 1e-30) ** (r / 2))
                    return -(soft / cst ** (1.0 / r)) / denom
                res = minimize(neg, z, method='L-BFGS-B',
                               options={'maxiter': 3000, 'ftol': 1e-16,
                                        'gtol': 1e-12})
                z = res.x
            res = minimize(lambda zz: -net.ratio(zz, r, denom), z,
                           method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-11,
                                    'fatol': 1e-13})
            val = net.ratio(res.x, r, denom)
            if val > best:
                best, bz = val, res.x
        c, v = net.split(bz)
        # junk report at the best point
        print(f"   best ratio over {args.starts} starts = {best:.12f}"
              f"   (>1 would refute (T))")
        out.append({'p': p, 'Gamma': G, 'support': S, 'block_ratio': rb,
                    'best_ratio': best, 'N_per_freq': net.N})

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
