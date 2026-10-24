# Referee check of R3(a), R9, R13: decoupled IF unions with programme allocation; x-uniform KKT measure p^{-n} mu;
# critical values nu^{nu/2} B_p K_j(mu); tau(q*) = nu^{nu/2} B_p max_l K_l(mu), attained off the support.
import numpy as np, sys
from vlib import *


def if_union(p, n, b, y):
    nu = n + 1; s = nu / b; r = 1 / s; xs = np.arange(p); U = []; W = []; lab = []
    for j in range(1, len(y) + 1):
        if y[j - 1] <= 1e-12: continue
        sqn = (y[j - 1] ** r / nu) ** (2 / b)        # ||omega||^2 so that ||omega||^b = y^r / nu  (P = 1)
        lam = np.sqrt(sqn / (nu * p / 2))
        for l in range(nu):
            al = np.pi * l / nu
            U.append([lam * np.cos(2 * np.pi * j * xs / p + al) for _ in range(n)])
            W.append(lam * np.cos(2 * np.pi * j * xs / p + n * al)); lab.append(j)
    return np.array(U), np.array(W), np.array(lab)


def check_union(p, n, b, S, label=''):
    nu = n + 1; s = nu / b; X, Y = grid(p, n); F = (p - 1) // 2
    Gam, y = programme(p, s, S, starts=150)
    GamB, B, mu, dres = binding_and_dual(p, s, y, tol=1e-7)
    U, W, lab = if_union(p, n, b, y)
    G = gaps(U, W, p, n, X, Y); P, sq = cost(U, W, b); h = G.min()
    hIF = nu ** (1 - s) * Bp(p, n) * Gam
    xdep = (G.max(0) - G.min(0)).max() / h
    Q = np.tile(mu / p ** n, (len(X), 1))
    onA = np.all(G[:, mu > 0] <= h * (1 + 1e-9))
    gU, gW = grad_lin(U, W, Q, p, n, X, Y)
    c = b * sq ** (b / 2 - 1); lamb = s * h / P
    res = np.sqrt(np.sum((gU - lamb * c[:, None, None] * U) ** 2) + np.sum((gW - lamb * c[:, None] * W) ** 2)) / \
        np.sqrt(np.sum((lamb * c[:, None, None] * U) ** 2) + np.sum((lamb * c[:, None] * W) ** 2))
    M = dual_tensor(Q, p, n, X, Y)
    Kv = np.array([Kl(p, mu, l).real for l in range(1, F + 1)])
    terr = 0
    for i in range(len(lab)):
        v = [U[i, t] / np.linalg.norm(U[i, t]) for t in range(n)] + [W[i] / np.linalg.norm(W[i])]
        ti = float(contract(M, v)); pred = nu ** (nu / 2) * Bp(p, n) * Kv[lab[i] - 1]
        terr = max(terr, abs(ti / pred - 1))
    tau = inj_norm(M, starts=30)
    taupred = nu ** (nu / 2) * Bp(p, n) * Kv.max()
    spt = sorted(set(lab.tolist()))
    print(f'{label}(p,n,b)=({p},{n},{b}) S={S} spt y={spt} y={np.round(y[np.array(spt)-1],5)} '
          f'h/hIF={h/hIF:.10f} x-dependence {xdep:.1e} dual resid {dres:.1e} q* on argmin {onA} KKT resid {res:.1e} '
          f'crit-value err {terr:.1e}')
    print(f'      K_l(mu) = {np.round(Kv,4)} ; tau(q*) = {tau:.6e} vs nu^(nu/2) B_p max K = {taupred:.6e} (rel {tau/taupred-1:.1e}); '
          f'argmax_l K_l = {int(np.argmax(Kv))+1} in spt? {int(np.argmax(Kv))+1 in spt}')
    sys.stdout.flush()
    return y, mu, Kv


if __name__ == "__main__":
    print('== R3(a)/R13 ==')
    check_union(5, 2, 2.3, [1, 2])
    check_union(7, 2, 2.0, [1, 2, 3])
    check_union(7, 3, 2.5, [1, 3])
    check_union(11, 2, 2.2, [1, 2, 4])
    check_union(5, 3, 2.0, [1], label='R13 ')
    check_union(3, 3, 2.5, [1], label='p=3 ')

    print('== R9: uniform optimal allocations with proper support ==')
    for p in [7, 11, 13, 17, 19, 23]:
        F = (p - 1) // 2
        for s in [1.2, 1.5, 1.8, 2.2, 3.0]:
            Gam, y = programme(p, s, None, starts=120, seed=p)
            spt = np.where(y > 1e-9)[0] + 1
            k = len(spt)
            if k == F: continue
            uni = y[spt - 1].max() / y[spt - 1].min()
            GamB, B, mu, dres = binding_and_dual(p, s, y, tol=1e-7)
            Kv = np.array([Kl(p, mu, l).real for l in range(1, F + 1)])
            off = [l for l in range(1, F + 1) if l not in spt]
            print(f'p={p:2d} s={s}: k={k} spt={spt.tolist()} max/min y={uni:.4f} dual resid {dres:.1e}; '
                  f'sum_F K = {Kv.sum():.6f} (p/2={p/2}); K on S = {np.round(Kv[spt-1],4)} < p/(p-1)={p/(p-1):.4f}; '
                  f'mean off-S K = {Kv[np.array(off)-1].mean():.4f}; max off = {Kv[np.array(off)-1].max():.4f}; '
                  f'claim(avg off > all on) {Kv[np.array(off)-1].mean() > Kv[spt-1].max()}')
            sys.stdout.flush()
