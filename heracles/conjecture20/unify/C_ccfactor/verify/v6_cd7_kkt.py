# Referee check v6: (a) CC7 construction at (5,2,2) and (7,3,2) with independent relabelling code;
# (b) CC3(iv) for NON-IF features: copies at random allocations have Pi_within >= c(theta_1) (own cone price);
# (c) CC5 KKT charge identity at an E_kkt local maximiser (own NNLS for the Fritz-John measure).
import sys
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import vlib as V

rng = np.random.default_rng(99)


def relabel(U, W, j, p, cost_target, b):
    idx = (j * np.arange(p)) % p
    U2 = U[:, :, idx].copy(); W2 = W[:, idx].copy()
    c = (cost_target / (V.sq(U2, W2) ** (b / 2)).sum()) ** (1 / b)
    return U2 * c, W2 * c


def perturbed_feature(p, n, eps, extra):
    U, W = V.if_block(p, n, 1, 1.0)
    a = np.arange(p)
    for i in range(U.shape[0]):
        for t in range(n):
            U[i, t] = (1 + eps * rng.normal()) * np.cos(2 * np.pi * a / p + np.pi * i / (n + 1) + eps * rng.normal())
        W[i] = (1 + eps * rng.normal()) * np.cos(2 * np.pi * a / p + n * np.pi * i / (n + 1) + eps * rng.normal())
    Ue, We, _ = V.rand_classC(p, n, {1: extra}, rng, amp=(0.05, 0.4))
    return np.concatenate([U, Ue]), np.concatenate([W, We])


def cone_price(U, W, b):
    m, n, p = U.shape; s = (n + 1) / b
    Ph = V.Phi(V.table(U, W), p, n)[0]
    return (Ph.real - np.abs(Ph.imag) / np.tan(np.pi / p)).min() / (V.Bp(p, n) * (V.sq(U, W) ** (b / 2)).sum() ** s)


print('=== (a) CC7 construction')
for (p, n, b) in [(5, 2, 2.0), (7, 3, 2.0), (11, 2, 2.4)]:
    nu = n + 1; s = nu / b; B = V.Bp(p, n); pi = nu ** (1 - s); F = (p - 1) // 2; Kp = V.kap(p)
    U1, W1 = perturbed_feature(p, n, 0.2, 1)
    P1 = (V.sq(U1, W1) ** (b / 2)).sum()
    m1 = V.gaps(U1, W1).min(0) / (B * P1 ** s)                      # normalised profile m(delta), delta=1..p-1
    for (j, l) in [(1, 2), (2, (p - 1) // 2)]:
        if l % p in (1, p - 1): continue
        k = (j * pow(l, -1, p)) % p; k = min(k, p - k); d0 = (pow(j, -1, p) * l) % p
        if k == j: continue
        Uk, Wk = V.if_block(p, n, k, 1.0); Uk, Wk = relabel(Uk, Wk, 1, p, 1.0, b)
        for eps in [0.3, 0.05, 0.01]:
            Uj, Wj = relabel(U1, W1, j, p, eps, b)
            U = np.concatenate([Uj, Uk]); W = np.concatenate([Wj, Wk])
            h = V.gaps(U, W).min()
            Pl = np.zeros(F); Pl[j - 1] = eps; Pl[k - 1] = 1.0
            cov = Kp @ Pl ** s; bind = sorted(set(np.where(cov <= cov.min() * (1 + 1e-12))[0] + 1))
            cd = h / (pi * B * cov.min())
            pred = (pi * Kp[0, 0] + eps ** s * min(m1[l - 1], m1[p - l - 1])) / (pi * Kp[0, 0] + eps ** s * pi * Kp[l - 1, 0])
            print(f'  ({p},{n},{b}) j={j} l={l} k={k} d0={d0} eps={eps}: binding {bind} (expect {sorted({d0, p - d0})});'
                  f' CD ratio {cd:.6f}; predicted pair term {pred:.6f}; DU ratio min(m(+-l))/(pi kappa_1(l)) ='
                  f' {min(m1[l - 1], m1[p - l - 1]) / (pi * Kp[l - 1, 0]):.4f}')

print('=== (b) CC3(iv) with NON-IF features: Pi_within >= own cone price c(theta_1) at random allocations')
worst = np.inf
for (p, n, b) in [(7, 2, 2.0), (7, 3, 2.5), (11, 2, 2.2), (5, 3, 2.0)]:
    nu = n + 1; s = nu / b; B = V.Bp(p, n); F = (p - 1) // 2; Kp = V.kap(p)
    for t in range(6):
        U1, W1 = perturbed_feature(p, n, 0.15, 1)
        c1 = cone_price(U1, W1, b)
        if c1 <= 0: continue
        S = [l for l in range(1, F + 1) if rng.random() < 0.7] or [1]
        Pl = np.zeros(F); Us, Ws = [], []
        for l in S:
            Pl[l - 1] = rng.uniform(0.05, 1.0)
            Uj, Wj = relabel(U1, W1, l, p, Pl[l - 1], b); Us.append(Uj); Ws.append(Wj)
        U = np.concatenate(Us); W = np.concatenate(Ws)
        h = V.gaps(U, W).min()
        piw = h / (B * V.Mcov(Pl ** s, p))
        worst = min(worst, piw / c1 - 1)
print(f'  min over cases of Pi_within/c(theta_1) - 1 = {worst:.3e} (>= 0 required)')

print('=== (c) CC5: E_q g(T_i) = h a_i^r / P at a KKT point (E_kkt lm_p5_n2_b2.0_k0)')
d = np.load('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p5_n2_b2.0_k0.npz')
U, W = d['U'], d['W']; b = float(d['b']); m, n, p = U.shape; nu = n + 1; s = nu / b
g = V.gaps(U, W); h = g.min(); P = (V.sq(U, W) ** (b / 2)).sum()
X = np.indices((p,) * n).reshape(n, -1).T; Y = X.sum(1) % p
act = np.argwhere(g <= h + 1e-7 * abs(h))
cols = []
for (xi, dk) in act:
    x = X[xi]; y = Y[xi]; dd = dk + 1
    gU = np.zeros_like(U); gW = np.zeros_like(W)
    for i in range(m):
        wdiff = W[i, y] - W[i, (y - dd) % p]
        for t in range(n):
            gU[i, t, x[t]] += wdiff * np.prod([U[i, tt, x[tt]] for tt in range(n) if tt != t])
        pr = np.prod([U[i, tt, x[tt]] for tt in range(n)])
        gW[i, y] += pr; gW[i, (y - dd) % p] -= pr
    cols.append(np.concatenate([gU.ravel(), gW.ravel()]))
G = np.array(cols).T
sq = V.sq(U, W)
gP = np.concatenate([(b * sq[:, None, None] ** (b / 2 - 1) * U).ravel(), (b * sq[:, None] ** (b / 2 - 1) * W).ravel()])
target = s * (h / P) * gP
Aug = np.vstack([G, 1e3 * np.ones((1, G.shape[1]))]); rhs = np.concatenate([target, [1e3]])
q, res = nnls(Aug, rhs)
print(f'  active constraints {len(act)}; KKT residual |G q - s(h/P) grad P| / |target| = {np.linalg.norm(G @ q - target) / np.linalg.norm(target):.2e}; sum q = {q.sum():.6f}')
ratios = []
for i in range(m):
    gi = V.gaps(U[i:i + 1], W[i:i + 1])
    Eq = sum(qq * gi[xi, dk] for qq, (xi, dk) in zip(q, act))
    ratios.append(Eq / (h * sq[i] ** (b / 2) / P))
print(f'  E_q g(T_i) / (h a_i^r / P): min {min(ratios):.6f} max {max(ratios):.6f}')
