# Sanity checks for the PROVED lemmas K2-K6 of E_kkt (see kkt_proofs.tex).
import numpy as np, itertools
from klib import *

rng = np.random.default_rng(1)
np.set_printoptions(precision=5, suppress=True)


def Ktilde(p, mu):          # K~_l(mu) = sum_delta mu_delta (1 - rho^{-l delta}), l = 1..F ; mu over delta=1..p-1
    F = (p - 1) // 2; d = np.arange(1, p)
    return np.array([np.sum(mu * (1 - np.exp(-2j * np.pi * l * d / p))) for l in range(1, F + 1)])


def restricted_programme(p, s, S):
    """Gamma^{(S)}_s by vertex enumeration restricted to support S (list of 1-based freqs)."""
    F = (p - 1) // 2; r = 1 / s; K = kappa_mat(p)[:F]; best = -1; bestv = None
    idx = [j - 1 for j in S]
    for k in range(1, len(idx) + 1):
        for Ssub in itertools.combinations(idx, k):
            for R in itertools.combinations(range(F), k):
                M = K[np.ix_(R, Ssub)]
                if abs(np.linalg.det(M)) < 1e-12: continue
                v = np.linalg.solve(M, np.ones(k))
                if np.any(v <= 1e-12): continue
                full = np.zeros(F); full[list(Ssub)] = v
                if np.any(K @ full < 1 - 1e-9): continue
                val = np.sum(v ** r) ** (-s)
                if val > best: best = val; bestv = full
    y = bestv / np.sum(bestv ** r) ** s
    return best, y


def programme_dual(p, s, y):
    """symmetric mu* on binding offsets with K_j(mu*) = Gamma y_j^{r-1} on spt y (least squares, mu>=0)."""
    from scipy.optimize import nnls
    F = (p - 1) // 2; r = 1 / s; K = kappa_mat(p)                 # rows delta 1..p-1
    vals = K @ y; G = vals.min(); bind = np.where(vals <= G * (1 + 1e-9))[0]
    spt = np.where(y > 0)[0]
    A = K[bind][:, spt].T                                          # K_j(mu) = sum_{delta in bind} mu_delta kappa_j(delta)
    rhs = G * y[spt] ** (r - 1)
    A2 = np.vstack([A, 1e3 * np.ones((1, len(bind)))]); b2 = np.concatenate([rhs, [1e3]])
    m, _ = nnls(A2, b2)
    mu = np.zeros(p - 1); mu[bind] = m
    mu = 0.5 * (mu + mu[::-1])                                       # symmetrise delta <-> -delta
    return mu, np.linalg.norm(A @ m - rhs) / np.linalg.norm(rhs), G


if __name__ == '__main__':
    print('== K2(a): the x-uniform measure p^{-n} mu* is a KKT measure of every decoupled IF network (any support S)')
    for (p, n, b, S) in [(5, 2, 2.3, [1, 2]), (7, 2, 2.0, [1, 2, 3]), (7, 2, 2.0, [1, 2]), (7, 3, 2.5, [1, 3]), (11, 2, 2.2, [1, 2, 4]), (5, 3, 2.0, [1])]:
        nu = n + 1; s = nu / b
        G, y = restricted_programme(p, s, S)
        net = if_network(p, n, y, b); c = net.cost(); net = Net(net.U * c ** (-1 / b), net.W * c ** (-1 / b), b)
        mu, mres, G2 = programme_dual(p, s, y)
        Q = np.tile(mu / net.N, (net.N, 1))
        gU, gW = grad_lin(net, Q)
        h = net.margin(); lam = net.s * h / net.cost(); cU, cW = net.grad_cost()
        res = np.linalg.norm(pack(gU, gW) - lam * pack(cU, cW)) / np.linalg.norm(lam * pack(cU, cW))
        act_ok = np.all(net.gaps()[:, mu > 0] <= h * (1 + 1e-9))
        print(f'  (p,n,b,S)=({p},{n},{b},{S}): programme-dual residual {mres:.1e}; KKT stationarity residual {res:.2e}; q on active set: {act_ok}')

    print('== K2(b): Lemma 2 averaged (E_{unif x mu} g = sum_l Re(conj(beta_l) K~_l(mu)) >= h) on random networks')
    worst = -np.inf; idmax = 0
    for trial in range(300):
        p = int(rng.choice([3, 5, 7])); n = int(rng.choice([2, 3])); m = int(rng.integers(1, 7))
        net = Net(rng.standard_normal((m, n, p)), rng.standard_normal((m, p)), 2.0)
        mu = rng.random(p - 1); mu /= mu.sum()
        g = net.gaps(); lhs = np.mean(g, axis=0) @ mu
        rhs = np.sum(np.real(np.conj(net.clean()) * Ktilde(p, mu)))
        idmax = max(idmax, abs(lhs - rhs) / (1 + abs(lhs)))
        worst = max(worst, g.min() - lhs)
    print(f'  max |E_{{unif x mu}} g - formula| = {idmax:.1e};  max (h - E_{{unif x mu}} g) = {worst:.3e} (must be <= 0)')

    print('== K3: IF networks: gamma <= nu^{1-s} B_p Gamma^{(S)}; equality structure and entropy log nu + H(z)')
    viol = -np.inf
    for trial in range(200):
        p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])); nu = n + 1
        b = float(rng.uniform(2.05, nu)); s = nu / b; F = (p - 1) // 2
        k = int(rng.integers(1, F + 1)); S = sorted(rng.choice(np.arange(1, F + 1), size=k, replace=False).tolist())
        Us = []; Ws = []
        for j in S:
            lam = rng.uniform(0.3, 2.0); U, W = if_block(p, n, j, lam, phase0=rng.uniform(0, 2 * np.pi)); Us.append(U); Ws.append(W)
        net = Net(np.concatenate(Us), np.concatenate(Ws), b)
        GS, _ = restricted_programme(p, s, S)
        viol = max(viol, net.gamma() / (nu ** (1 - s) * Bp(p, n) * GS) - 1)
    print(f'  random IF unions (200): max gamma/(nu^(1-s) B_p Gamma^(S)) - 1 = {viol:.2e} (must be <= 0)')
    for (p, n, b, S) in [(7, 2, 2.4, [1, 2]), (7, 3, 2.5, [1, 2, 3]), (11, 2, 2.6, [1, 2, 3, 4, 5])]:
        nu = n + 1; s = nu / b
        GS, y = restricted_programme(p, s, S)
        net = if_network(p, n, y, b)
        z = y ** (1 / s) / np.sum(y ** (1 / s))
        print(f'  ({p},{n},{b},S={S}): equality gamma/(nu^(1-s)B_p Gamma^S) = {net.gamma()/(nu**(1-s)*Bp(p,n)*GS):.10f}, H = {net.H():.8f}, log nu + H(z) = {np.log(nu)+entropy(z):.8f}')

    print('== K5: cheapness of an aligned budget-tight feature = exp((s-1)(log nu - H_s(w)))  (Renyi-s entropy of cost shares)')
    mx = 0
    for trial in range(200):
        p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])); nu = n + 1
        b = float(rng.uniform(2.05, nu)); s = nu / b; r = 1 / s
        m = int(rng.integers(1, 9)); j = 1; a_ = np.arange(p)
        U = []; W = []
        for i in range(m):                          # single-frequency neurons, aligned atoms (phases sum to 0), random sizes
            lam = rng.uniform(0.2, 2.0); ph = rng.uniform(0, 2 * np.pi, n)
            U.append([lam * np.cos(2 * np.pi * j * a_ / p + ph[t]) for t in range(n)])
            W.append(lam * np.cos(2 * np.pi * j * a_ / p + ph.sum()))
        net = Net(np.array(U), np.array(W), b)
        at = net.atoms()[:, 0]; beta = at.sum(); Q = np.sum(np.abs(at) ** r)
        c = np.abs(beta) / (nu ** (1 - s) * Q ** s)
        w = net.shares(); Hs = np.log(np.sum(w ** s)) / (1 - s)
        mx = max(mx, abs(c / np.exp((s - 1) * (np.log(nu) - Hs)) - 1))
    print(f'  max relative error of the identity over 200 random aligned features: {mx:.1e}')

    print('== K5b: Renyi form of the universal bound: gamma <= B_p min_delta sum_l (sum_i v_il^s) kappa_l(delta),  v_il = |beta_il|^r/(C_b P)')
    worst = -np.inf
    for trial in range(300):
        p = int(rng.choice([3, 5, 7])); n = int(rng.choice([2, 3])); nu = n + 1; m = int(rng.integers(2, 9))
        b = float(rng.uniform(2.0, nu)); s = nu / b; r = 1 / s
        net = Net(rng.standard_normal((m, n, p)), rng.standard_normal((m, p)), b)
        if net.margin() <= 0:                       # push to positive margin: add an IF block
            U, W = if_block(p, n, 1, 1.5); net = Net(np.concatenate([net.U * 0.3, U]), np.concatenate([net.W * 0.3, W]), b)
        if net.margin() <= 0: continue
        at = np.abs(net.atoms()); P = net.cost(); Cb = Bp(p, n) ** r
        v = at ** r / (Cb * P)
        y = (v ** s).sum(0)
        bound = Bp(p, n) * np.min(kappa_mat(p) @ y)
        worst = max(worst, net.gamma() / bound - 1)
    print(f'  max gamma/bound - 1 over random positive-margin networks: {worst:.3e} (must be <= 0)')

    print('== K6: class C:  hat g_delta(2 l 1_T) = (1/2) conj(beta_{l,T}) c_l(delta)  (so x-uniform KKT measure => IF)')
    mx = 0
    for trial in range(60):
        p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])); F = (p - 1) // 2; m = int(rng.integers(1, 6))
        U = []; W = []
        for i in range(m):
            j = int(rng.integers(1, F + 1)); Pj = None
            a_ = np.arange(p)
            U.append([rng.standard_normal() * np.cos(2 * np.pi * j * a_ / p) + rng.standard_normal() * np.sin(2 * np.pi * j * a_ / p) for t in range(n)])
            W.append(rng.standard_normal() * np.cos(2 * np.pi * j * a_ / p) + rng.standard_normal() * np.sin(2 * np.pi * j * a_ / p))
        net = Net(np.array(U), np.array(W), 2.0)
        g = net.gaps(); sf = net.signflip()
        for d in range(1, p):
            gh = np.fft.fftn(g[:, d - 1].reshape((p,) * n)) / net.N       # hat g(chi) = E_x g rho^{-<chi,x>}
            for l in range(1, F + 1):
                for T in itertools.product([0, 1], repeat=n):
                    if not any(T): continue
                    chi = tuple((2 * l * T[t]) % p for t in range(n))
                    pred = 0.5 * np.conj(sf[(l, T)]) * (1 - np.exp(-2j * np.pi * l * d / p))
                    mx = max(mx, abs(gh[chi] - pred) / (1e-12 + np.abs(g).max()))
    print(f'  max |hat g - prediction| / max|g| over 60 random class-C networks: {mx:.1e}')
