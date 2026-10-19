# Referee check v4: the CC12 counterexample (cex_charge_rules_p5_n2_b2.npz), recomputed with vlib (float) and exact
# Fractions; robustness of the argmin rule (near-ties) on the UNROUNDED class-C network.
import sys, itertools
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import vlib as V

D = '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/'
p, n, b = 5, 2, 2.0; nu = 3; s = 1.5; B = V.Bp(p, n); pi = 3 ** (1 - s); F = 2
Kp = V.kap(p)


def analyse(U, W, lab, tag):
    f = V.table(U, W); g = V.gaps(U, W, f); h = g.min(); beta = V.clean(f, p, n)
    Pl = np.array([(V.sq(U, W)[lab == l] ** (b / 2)).sum() for l in (1, 2)])
    cov = Kp @ Pl ** s; bind = np.where(cov <= cov.min() * (1 + 1e-12))[0] + 1
    cd = h / (pi * B * cov.min())
    G = {(): np.zeros(p - 1)}
    gl = {l: V.gaps(U[lab == l], W[lab == l]) for l in (1, 2)}
    G[(1,)] = gl[1].min(0); G[(2,)] = gl[2].min(0); G[(1, 2)] = g.min(0)
    Rb = {l: Kp[:, l - 1] * beta[l - 1].real for l in (1, 2)}
    v = {T: sum((Rb[l] for l in T), np.zeros(p - 1)) - 0.5 * (G[T] + G[T][::-1]) for T in G}
    phi1 = 0.5 * (v[(1,)] - v[()] + v[(1, 2)] - v[(2,)])
    print(f'[{tag}] h = {h:.10f}; class C labels {list(lab)}; |beta| = {np.abs(beta)}; P = {Pl}; binding offsets {list(bind)}')
    print(f'[{tag}] CD ratio = {cd:.10f}; cheapness Re beta_l/(pi B P_l^s) = {beta.real / (pi * B * Pl ** s)}')
    for d in bind:
        k = d - 1; price = pi * B * Pl[0] ** s * Kp[k, 0]
        sh = (Rb[1][k] - phi1[k]) / price
        tm = 0; gaps_sorted = {}
        for sgn in (1, -1):
            kk = (sgn * d) % p - 1
            gg = g[:, kk]; mn = gg.min(); order = np.sort(gg)
            gaps_sorted[sgn] = (order[1] - order[0]) / max(abs(mn), 1)
            arg = np.where(gg <= mn + 1e-9 * max(1, abs(mn)))[0]
            tm += gl[1][arg, kk].mean() / 2
        print(f'[{tag}]  delta={d}: Shapley charge/price = {sh:.8f}; argmin charge/price = {tm / price:.8f};'
              f' relative gap between 1st and 2nd smallest g(.,+-delta): {gaps_sorted[1]:.2e}, {gaps_sorted[-1]:.2e};'
              f' v(1)={v[(1,)][k]:.5f} v(2)={v[(2,)][k]:.5f} v(12)={v[(1, 2)][k]:.5f}')
    return h, Pl


d = np.load(D + 'cex_charge_rules_p5_n2_b2.npz')
U, W, lab = d['U'], d['W'], d['lab']
analyse(U, W, lab, 'rounded')
# class-C purity of the rounded network: off-plane energy
a = np.arange(p)
off = 0
for i in range(len(lab)):
    for vec in list(U[i]) + [W[i]]:
        vh = np.fft.fft(vec); keep = np.zeros(p, bool); keep[[lab[i], p - lab[i]]] = True
        off = max(off, np.sqrt((np.abs(vh[~keep]) ** 2).sum() / max((np.abs(vh) ** 2).sum(), 1e-300)))
print(f'rounded network: max relative off-plane (other frequency or DC) amplitude = {off:.2e}')
# exact projection (unrounded)
d0 = np.load(D + 'shapley_best_sh_bind_p5_S1-2.npz'); U0, W0, lab0 = d0['U'], d0['W'], d0['lab']


def proj(vv, l):
    vh = np.fft.fft(vv); out = np.zeros(p, complex); out[l] = vh[l]; out[p - l] = vh[p - l]
    return np.real(np.fft.ifft(out))


U1 = np.array([[proj(U0[i, t], lab0[i]) for t in range(n)] for i in range(len(lab0))])
W1 = np.array([proj(W0[i], lab0[i]) for i in range(len(lab0))])
analyse(U1, W1, lab0, 'unrounded-projected')
# exact margin of the rounded network
Uq = [[[Fr(float(x)) for x in U[i, t]] for t in range(n)] for i in range(len(lab))]
Wq = [[Fr(float(x)) for x in W[i]] for i in range(len(lab))]
hmin = None
for x in itertools.product(range(p), repeat=n):
    y = sum(x) % p
    z = [Uq[i][0][x[0]] * Uq[i][1][x[1]] for i in range(len(lab))]
    fv = [sum(Wq[i][c] * z[i] for i in range(len(lab))) for c in range(p)]
    for dd in range(1, p):
        gv = fv[y] - fv[(y - dd) % p]
        hmin = gv if hmin is None or gv < hmin else hmin
print(f'exact margin of the stored (rounded) network: {float(hmin):.12f} > 0: {hmin > 0}')
