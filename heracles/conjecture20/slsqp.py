# minimize P_b = sum_i ||omega_i||^b  s.t. all gaps >= 1 ; gamma = 1 / P_b^{nu/b}
import numpy as np, itertools, sys, math
from scipy.optimize import minimize

def dataset(p, n):
    X = np.array(list(itertools.product(range(p), repeat=n)))
    y = X.sum(1) % p
    return X, y

def setup(p, n, m):
    X, y = dataset(p, n)
    N = len(X)
    pairs = [(i, c) for i in range(N) for c in range(p) if c != y[i]]
    I = np.array([a for a, _ in pairs]); C = np.array([c for _, c in pairs])
    return X, y, I, C

def unpack(th, m, n, p):
    U = th[: m * n * p].reshape(m, n, p)
    W = th[m * n * p:].reshape(m, p)
    return U, W

def feats(U, X):
    m, n, p = U.shape
    Z = np.ones((m, len(X)))
    for t in range(n):
        Z = Z * U[:, t, X[:, t]]
    return Z  # (m, N)

def make(p, n, b, m):
    X, y, I, C = setup(p, n, m)
    N = len(X)
    def gaps(th):
        U, W = unpack(th, m, n, p)
        Z = feats(U, X)
        f = Z.T @ W
        return f[I, y[I]] - f[I, C]
    def gaps_jac(th):
        U, W = unpack(th, m, n, p)
        Z = feats(U, X)  # m,N
        K = len(I)
        J = np.zeros((K, th.size))
        # dW: d f[i,c]/dW[k,c'] = Z[k,i] [c==c']
        JW = np.zeros((K, m, p))
        JW[np.arange(K), :, y[I]] += Z[:, I].T
        JW[np.arange(K), :, C] -= Z[:, I].T
        dw = W[:, y[I]] - W[:, C]  # m,K
        JU = np.zeros((K, m, n, p))
        for t in range(n):
            # Z without factor t
            Zt = np.ones((m, N))
            for s in range(n):
                if s != t: Zt = Zt * U[:, s, X[:, s]]
            a = X[I, t]
            JU[np.arange(K), :, t, a] += (Zt[:, I] * dw).T
        J[:, : m * n * p] = JU.reshape(K, -1)
        J[:, m * n * p:] = JW.reshape(K, -1)
        return J
    def pen(th):
        U, W = unpack(th, m, n, p)
        sq = (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
        return ((sq + 1e-30) ** (b / 2)).sum()
    def pen_jac(th):
        U, W = unpack(th, m, n, p)
        sq = (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
        coef = b * (sq + 1e-30) ** (b / 2 - 1)
        gU = coef[:, None, None] * U
        gW = coef[:, None] * W
        return np.concatenate([gU.ravel(), gW.ravel()])
    return gaps, gaps_jac, pen, pen_jac

def solve(p, n, b, m, seed, maxiter=2000):
    rng = np.random.default_rng(seed)
    gaps, gj, pen, pj = make(p, n, b, m)
    th = rng.standard_normal(m * (n * p + p))
    # scale so that some gap structure; random start then the solver finds feasibility
    cons = [{"type": "ineq", "fun": lambda t: gaps(t) - 1.0, "jac": gj}]
    r = minimize(pen, th, jac=pj, constraints=cons, method="SLSQP",
                 options={"maxiter": maxiter, "ftol": 1e-14})
    g = gaps(r.x)
    hmin = g.min()
    P = pen(r.x)
    nu = n + 1
    gamma = hmin / P ** (nu / b) if hmin > 0 else -1
    return gamma, r

def iff(p, n, b, Gamma):
    nu = n + 1; s = nu / b
    Bp = 2 * (2 * p * nu) ** (-nu / 2)
    return nu ** (1 - s) * Bp * Gamma

if __name__ == "__main__":
    p, n, b, m = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    seeds = int(sys.argv[5]); G = float(sys.argv[6])
    T = iff(p, n, b, G)
    best = -1
    for sd in range(seeds):
        gm, r = solve(p, n, b, m, sd)
        best = max(best, gm)
        print(f"p={p} n={n} b={b} m={m} seed={sd} gamma={gm:.6e} ratio={gm/T:.5f} ok={r.success}", flush=True)
    print("BEST ratio", best / T)
