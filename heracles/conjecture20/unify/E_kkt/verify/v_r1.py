# Referee check of R1 on independently computed local maximisers (SLSQP on max t s.t. g >= t, cost = 1),
# including p = 3.  KKT measure by LP (L1 residual), critical values by own tensor contraction, tau by own ALS.
import numpy as np, sys, time
from scipy.optimize import minimize, linprog
from vlib import *


def jac_g(U, W, p, n, X, Y):
    m = U.shape[0]; N = len(X); A = N * (p - 1)
    JU = np.zeros((A, m, n, p)); JW = np.zeros((A, m, p))
    fac = np.stack([U[:, t, X[:, t]] for t in range(n)], 0)       # (n,m,N)
    Up = fac.prod(0)                                                # (m,N)
    rows = np.arange(N)
    for d in range(1, p):
        a = rows * (p - 1) + (d - 1)
        dw = W[:, Y] - W[:, (Y - d) % p]                            # (m,N)
        for t in range(n):
            oth = np.prod(np.delete(fac, t, 0), 0) * dw             # (m,N)
            for i in range(m):
                np.add.at(JU[:, i, t, :], (a, X[:, t]), oth[i])
        for i in range(m):
            np.add.at(JW[:, i, :], (a, Y), Up[i])
            np.add.at(JW[:, i, :], (a, (Y - d) % p), -Up[i])
    return np.concatenate([JU.reshape(A, -1), JW.reshape(A, -1)], 1)


def local_max(p, n, b, m, seed):
    X, Y = grid(p, n); rng = np.random.default_rng(seed)
    dim = m * n * p + m * p
    def split(v):
        return v[:m * n * p].reshape(m, n, p), v[m * n * p:dim].reshape(m, p)
    v0 = rng.standard_normal(dim); U, W = split(v0); c, _ = cost(U, W, b); v0 /= c ** (1 / b)
    U, W = split(v0); t0 = gaps(U, W, p, n, X, Y).min()
    def cons_g(v):
        U, W = split(v); return gaps(U, W, p, n, X, Y).ravel() - v[-1]
    def cons_gj(v):
        U, W = split(v); J = jac_g(U, W, p, n, X, Y)
        return np.hstack([J, -np.ones((J.shape[0], 1))])
    def cons_c(v):
        U, W = split(v); return np.array([cost(U, W, b)[0] - 1])
    def cons_cj(v):
        U, W = split(v); _, sq = cost(U, W, b); c = b * sq ** (b / 2 - 1)
        return np.concatenate([(c[:, None, None] * U).ravel(), (c[:, None] * W).ravel(), [0]])[None, :]
    x = np.concatenate([v0, [t0]])
    for _ in range(6):
        res = minimize(lambda z: -z[-1], x, jac=lambda z: np.concatenate([np.zeros(dim), [-1]]),
                       constraints=[{'type': 'ineq', 'fun': cons_g, 'jac': cons_gj},
                                    {'type': 'eq', 'fun': cons_c, 'jac': cons_cj}],
                       method='SLSQP', options={'maxiter': 3000, 'ftol': 1e-16})
        x = res.x
    U, W = split(x[:dim])
    return U, W


def kkt_lp(U, W, p, n, b, rtol=1e-7):
    X, Y = grid(p, n); nu = n + 1; s = nu / b
    G = gaps(U, W, p, n, X, Y).ravel(); h = G.min()
    act = np.where(G <= h + rtol * abs(h))[0]
    J = jac_g(U, W, p, n, X, Y)[act].T                              # (dim, |act|)
    P, sq = cost(U, W, b); c = b * sq ** (b / 2 - 1)
    gP = np.concatenate([(c[:, None, None] * U).ravel(), (c[:, None] * W).ravel()])
    rhs = s * h / P * gP
    D, k = J.shape
    cvec = np.concatenate([np.zeros(k), np.ones(2 * D)])
    Aeq = np.vstack([np.hstack([J, np.eye(D), -np.eye(D)]), np.concatenate([np.ones(k), np.zeros(2 * D)])[None, :]])
    beq = np.concatenate([rhs, [1]])
    r = linprog(cvec, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (k + 2 * D), method='highs')
    q = r.x[:k]
    resid = np.linalg.norm(J @ q - rhs) / np.linalg.norm(rhs)
    Q = np.zeros(len(X) * (p - 1)); Q[act] = q
    return Q.reshape(len(X), p - 1), resid, h, P


if __name__ == "__main__":
    cases = [(3, 2, 2.0, 4), (3, 2, 2.6, 4), (3, 3, 2.0, 5), (5, 2, 2.3, 6), (5, 2, 2.0, 6), (7, 2, 2.5, 7)]
    for (p, n, b, m) in cases:
        nu = n + 1; s = nu / b; X, Y = grid(p, n)
        best = None
        for seed in range(4):
            U, W = local_max(p, n, b, m, seed)
            G = gaps(U, W, p, n, X, Y); P, sq = cost(U, W, b); gam = G.min() / P ** s
            if best is None or gam > best[0]: best = (gam, U, W)
        gam, U, W = best
        # prune dead neurons
        _, sq = cost(U, W, b); keep = sq > 1e-6 * sq.max(); U, W = U[keep], W[keep]
        Q, resid, h, P = kkt_lp(U, W, p, n, b)
        _, sq = cost(U, W, b)
        wsh = sq ** (b / 2) / P
        # (b) balance and T_q(omega_i) = h w_i
        bal = max(np.abs(np.concatenate([(U[i] ** 2).sum(1), [(W[i] ** 2).sum()]]) / (sq[i] / nu) - 1).max() for i in range(len(sq)))
        M = dual_tensor(Q, p, n, X, Y)
        Tw = np.array([contract(M, [U[i, t] for t in range(n)] + [W[i]]) for i in range(len(sq))])
        errb = np.abs(Tw / (h * wsh) - 1).max()
        # (c) singular tuple and law
        t = []; sres = 0
        for i in range(len(sq)):
            v = [U[i, tt] / np.linalg.norm(U[i, tt]) for tt in range(n)] + [W[i] / np.linalg.norm(W[i])]
            ti = float(contract(M, v)); t.append(ti)
            for k in range(nu):
                sres = max(sres, np.linalg.norm(contract(M, v, skip=k) - ti * v[k]) / abs(ti))
        t = np.array(t)
        law = np.abs(t / (nu ** (nu / 2) * gam * wsh ** (1 - s)) - 1).max()
        H = -(wsh * np.log(wsh)).sum()
        ident = np.log(nu ** (nu / 2) * gam) - ((wsh * np.log(t)).sum() - (s - 1) * H)
        # (e),(f)
        tau = inj_norm(M, starts=40)
        mu = Q.sum(0); qX = Q.sum(1)
        F = (p - 1) // 2
        low = nu ** (nu / 2) * Bp(p, n) * max(abs(Kl(p, mu, l)) for l in range(1, F + 1))
        up = np.sqrt(2) * min(1, np.linalg.norm(qX))
        Mu = dual_tensor(np.tile(mu / p ** n, (len(X), 1)), p, n, X, Y)
        tau_u = inj_norm(Mu, starts=40)
        print(f'(p,n,b)=({p},{n},{b}) width {len(sq)} gamma={gam:.6e} KKT LP resid {resid:.1e}; balance err {bal:.1e}; '
              f'T_q(omega)=h w err {errb:.1e}; singular resid {sres:.1e}; law err {law:.1e}; identity {ident:.1e}')
        print(f'    tau={tau:.6e}  max t/tau={t.max()/tau:.4f}  tau(x-unif)={tau_u:.6e} vs formula {low:.6e} (rel {tau_u/low-1:.1e}); '
              f'low<=tau: {low <= tau*(1+1e-9)}; tau<=up: {tau <= up}; m <= (tau/nu^{{nu/2}}gamma)^a: '
              f'{len(sq)} <= {((tau/(nu**(nu/2)*gam))**(1/(s-1)) if s>1 else np.inf):.2f}')
        sys.stdout.flush()
