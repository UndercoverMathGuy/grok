# Slack principle input at (p,n) = (5,2) and (7,2):  how large is W_S(s) (best network whose features lie in S)
# relative to the decoupled prediction pi B_p Gamma^{(S)}_s  (pi = 3^{1-s} at n = 2, Note Thm q1n2)?
# The count (C) at (5,2) for all b holds iff W_{{1}} < gamma*, which follows if W_{{1}} <= (1+eps_tol) pi B_p kappa_min.
# Features outside S are suppressed by a penalty mu * sum_{j not in S} |beta_j| / cost^s (beta_j = clean coefficient);
# arbitrary inputs and outputs otherwise (compensating fields at other frequencies are allowed).
import sys, math, numpy as np
from lib import *
from transitions_fast import table, best_support_value

def obj(x, m, n, p, b, tau, S_off, mu):
    U, W = unpack(x, m, n, p); net = Net(U, W, b); c = net.cost(); s = net.s
    g = net.gaps() * c ** (-s)
    gmin = g.min(); z = np.exp(-(g - gmin) / tau); Z = z.sum(); lam = z / Z
    F = gmin - tau * np.log(Z)
    gU, gW = net.grad_lin(lam); cU, cW = net.grad_cost(); Eg = (lam * g).sum()
    dU = c ** (-s) * gU - s * Eg / c * cU; dW = c ** (-s) * gW - s * Eg / c * cW
    # penalty on clean coefficients beta_j, j in S_off:  beta_j = 2 p^{-nu} sum_i conj(What_i(j)) prod_t Uhat_it(j)
    nu = n + 1; Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1); a = np.arange(p)
    pen = 0.0; dUp = np.zeros_like(U); dWp = np.zeros_like(W)
    for j in S_off:
        prodU = np.prod(Uh[:, :, j], axis=1)
        beta = 2 * p ** (-nu) * np.sum(np.conj(Wh[:, j]) * prodU)
        mag = abs(beta) + 1e-300; pen += mag
        # d|beta| = Re(conj(beta)/|beta| dbeta)
        ph = np.conj(beta) / mag
        # dbeta/dW_i(c) = 2p^{-nu} prodU_i rho^{+jc};   dbeta/dU_it(a) = 2p^{-nu} conj(What_i(j)) prod_{t'!=t} Uhat rho^{-ja}
        dWp += np.real(ph * 2 * p ** (-nu) * prodU[:, None] * np.exp(2j * np.pi * j * a / p)[None, :])
        for t in range(n):
            others = np.prod(np.delete(Uh[:, :, j], t, axis=1), axis=1)
            dUp[:, t, :] += np.real(ph * 2 * p ** (-nu) * (np.conj(Wh[:, j]) * others)[:, None] * np.exp(-2j * np.pi * j * a / p)[None, :])
    F2 = F - mu * pen * c ** (-s)
    dU = dU - mu * (dUp * c ** (-s) - s * pen * c ** (-s - 1) * cU)
    dW = dW - mu * (dWp * c ** (-s) - s * pen * c ** (-s - 1) * cW)
    return F2, pack(dU, dW)

def run(p, n, b, S, m, starts, seed=0):
    nu = n + 1; s = nu / b; F_ = (p - 1) // 2; rng = np.random.default_rng(seed)
    S_off = [j for j in range(1, F_ + 1) if j not in S]
    ref = 3 ** (1 - s) * Bp(p, n) * max(best_support_value(p, s, T) for T in
                                        [tuple(sorted(T)) for k in range(1, len(S) + 1) for T in __import__('itertools').combinations(S, k)])
    best = (-np.inf, None); vals = []
    for st in range(starts):
        if st == 0:   # IF network on S (programme-restricted shares) plus noise at all frequencies
            y = np.zeros(F_); y[[j - 1 for j in S]] = 1.0
            net0 = if_network(p, n, y, b); U, W = net0.U, net0.W
            U = np.concatenate([U, 0.3 * rng.normal(size=(m - len(U), n, p))]) if m > len(U) else U[:m]
            W = np.concatenate([W, 0.3 * rng.normal(size=(m - len(W), p))]) if m > len(W) else W[:m]
            U = U + 0.05 * rng.normal(size=U.shape); W = W + 0.05 * rng.normal(size=W.shape)
        else:
            U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        x = pack(U, W); sc = Bp(p, n) * nu ** (1 - s)
        for tau, mu in [(3e-2, 1), (1e-2, 3), (3e-3, 10), (1e-3, 30), (3e-4, 100), (1e-4, 300), (3e-5, 1000)]:
            r = minimize(lambda z: tuple(-v for v in obj(z, m, n, p, b, tau * sc, S_off, mu)), x, jac=True,
                         method='L-BFGS-B', options={'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-12})
            x = r.x
        U, W = unpack(x, m, n, p); net = Net(U, W, b)
        A = np.abs(net.clean()); off = max([A[j - 1] for j in S_off] + [0]) / A.max()
        val = net.gamma() / ref; vals.append((round(val, 4), round(off, 6)))
        if off < 1e-4 and val > best[0]: best = (val, net)
    return best, vals

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2]); bs = [float(v) for v in sys.argv[3].split(",")]
    S = tuple(int(v) for v in sys.argv[4].split(",")); m = int(sys.argv[5]); starts = int(sys.argv[6])
    for b in bs:
        (val, net), vals = run(p, n, b, S, m, starts)
        s = (n + 1) / b
        from vertex_exact import vertices, gamma_S
        V = vertices(p); G = gamma_S(V, tuple(range(1, (p - 1) // 2 + 1)), s); GS = gamma_S(V, S, s)
        print(f"p={p} n={n} b={b} S={S}: W_S/(pi B_p Gamma^(S)) >= {val:.5f}   needed < 1+eps_tol = Gamma/Gamma^(S) = {G/GS:.4f}   starts {sorted(vals)[-5:]}", flush=True)
