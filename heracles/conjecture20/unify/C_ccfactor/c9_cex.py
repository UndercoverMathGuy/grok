# C_ccfactor c9: high-precision certificate of the charge-rule counterexample at (p,n,b) = (5,2,2), s = 3/2.
# Weights are rounded to 10 significant digits and treated as exact rationals; margins and all minima over x are
# exact (Fractions); costs, kappa, prices in 50-digit mpmath.  pi_{2,b}(5) = 3^{1-s} (Note Thm q1n2).
# Claims checked: (i) class C, h > 0; (ii) CD holds: h <= pi B_p M(P^s) (ratio < 1); (iii) at a P^s-binding offset the
# Shapley charge and the argmin (test-measure) charge of the cheap feature exceed its price pi B_p P_1^s kappa_1(delta).
import itertools, sys
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
mp.mp.dps = 50
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
import cclib as C
d = np.load('/home/user/grok/heracles/conjecture20/unify/C_ccfactor/shapley_best_sh_bind_p5_S1-2.npz')
U, W, lab = d['U'], d['W'], d['lab']
p, n = 5, 2; nu = 3; s = mp.mpf(3) / 2; r = 1 / s; F = 2
ok, lab2 = C.neuron_freq(C.E.Net(U, W), 1e-9)
# project exactly onto the frequency planes (removes rounding noise) and round to 10 digits
a = np.arange(p)
def proj(v, l):
    c = np.cos(2 * np.pi * l * a / p); sn = np.sin(2 * np.pi * l * a / p)
    return (v @ c) / (c @ c) * c + (v @ sn) / (sn @ sn) * sn
U = np.array([[proj(U[i, t], lab[i]) for t in range(n)] for i in range(len(lab))]); W = np.array([proj(W[i], lab[i]) for i in range(len(lab))])
q = lambda x: Fr(float('%.10g' % x))
Uq = [[[q(U[i, t, x]) for x in range(p)] for t in range(n)] for i in range(len(lab))]
Wq = [[q(W[i, c]) for c in range(p)] for i in range(len(lab))]
m = len(lab)
print('neurons:', m, 'frequencies:', list(lab), ' class-C test (float):', ok)


def gaps_exact(idx):
    """g[x][delta-1] for the sub-network made of neurons idx (exact)."""
    out = {}
    for x in itertools.product(range(p), repeat=n):
        y = sum(x) % p
        z = [Uq[i][0][x[0]] * Uq[i][1][x[1]] for i in idx]
        f = [sum(Wq[i][c] * zz for i, zz in zip(idx, z)) for c in range(p)]
        out[x] = [f[y] - f[(y - dd) % p] for dd in range(1, p)]
    return out


allidx = list(range(m)); feat = {l: [i for i in range(m) if lab[i] == l] for l in (1, 2)}
g = gaps_exact(allidx); h = min(min(v) for v in g.values())
print('exact margin h =', float(h), '(> 0:', h > 0, ')')
sq = [sum(mp.mpf(Uq[i][t][x].numerator) / Uq[i][t][x].denominator for t in range(n) for x in [] ) for i in range(m)]
sq = [mp.fsum([mp.mpf(Uq[i][t][x].numerator) / Uq[i][t][x].denominator * mp.mpf(Uq[i][t][x].numerator) / Uq[i][t][x].denominator
               for t in range(n) for x in range(p)]) + mp.fsum([(mp.mpf(Wq[i][c].numerator) / Wq[i][c].denominator) ** 2 for c in range(p)])
      for i in range(m)]
Pl = {l: mp.fsum([sq[i] for i in feat[l]]) for l in (1, 2)}          # b = 2: ||omega||^b = ||omega||^2
kap = lambda l, dd: 1 - mp.cos(2 * mp.pi * l * dd / p)
Bp = 2 * (2 * p * nu) ** (-mp.mpf(nu) / 2); pi = mp.mpf(3) ** (1 - s)
cov = {dd: mp.fsum([Pl[l] ** s * kap(l, dd) for l in (1, 2)]) for dd in range(1, p)}
Mps = min(cov.values()); bind = [dd for dd in cov if cov[dd] <= Mps * (1 + mp.mpf(10) ** -30)]
print('P_1 =', mp.nstr(Pl[1], 12), ' P_2 =', mp.nstr(Pl[2], 12), ' binding offsets of P^s:', bind)
cd = h.numerator / mp.mpf(h.denominator) / (pi * Bp * Mps)
print('CD ratio h/(pi B_p M(P^s)) =', mp.nstr(cd, 15), '  (<1: CD holds)')
# clean coefficients (exact DFT is not needed: use the gap means, exact): E_x g_l(x,delta) averaged over +-delta = Re beta_l kappa_l
gl = {l: gaps_exact(feat[l]) for l in (1, 2)}
N = p ** n
mean = {l: [sum(v[dd - 1] for v in gl[l].values()) / N for dd in range(1, p)] for l in (1, 2)}
Rbar = {l: [(mean[l][dd - 1] + mean[l][p - dd - 1]) / 2 for dd in range(1, p)] for l in (1, 2)}
f2m = lambda x: mp.mpf(x.numerator) / x.denominator
Rebeta = {l: f2m(Rbar[l][0]) / (1 - mp.cos(2 * mp.pi * l / p)) for l in (1, 2)}
price = {l: pi * Bp * Pl[l] ** s for l in (1, 2)}
print('cheapness Re beta_l/(pi B_p P_l^s):', {l: mp.nstr(Rebeta[l] / price[l], 10) for l in (1, 2)})
# depth game v(T)(delta) and Shapley charge of feature 1; argmin (test-measure) charge
def Gmin(gdict):
    return [min(v[dd - 1] for v in gdict.values()) for dd in range(1, p)]
G = {(): [Fr(0)] * (p - 1), (1,): Gmin(gl[1]), (2,): Gmin(gl[2]), (1, 2): Gmin(g)}
v = {T: [sum((Rbar[l][dd - 1] for l in T), Fr(0)) - (G[T][dd - 1] + G[T][p - dd - 1]) / 2 for dd in range(1, p)] for T in G}
phi1 = [(v[(1,)][k] - v[()][k] + v[(1, 2)][k] - v[(2,)][k]) / 2 for k in range(p - 1)]
for dd in bind:
    k = dd - 1
    sh_charge = Rbar[1][k] - phi1[k]
    args = {sgn: [x for x, val in g.items() if val[(sgn * dd) % p - 1] == G[(1, 2)][(sgn * dd) % p - 1]] for sgn in (1, -1)}
    tm = sum((sum(gl[1][x][(sgn * dd) % p - 1] for x in args[sgn]) / len(args[sgn]) for sgn in (1, -1)), Fr(0)) / 2
    lim = price[1] * kap(1, dd)
    print(f'binding delta={dd}: Shapley charge/price = {mp.nstr(mp.mpf(sh_charge.numerator) / sh_charge.denominator / lim, 12)};'
          f' argmin charge/price = {mp.nstr(mp.mpf(tm.numerator) / tm.denominator / lim, 12)}; |argmin sets| = {len(args[1])},{len(args[-1])}')
    print(f'   depth game: v(1)={float(v[(1,)][k]):.6e} v(2)={float(v[(2,)][k]):.6e} v(12)={float(v[(1, 2)][k]):.6e}'
          f'  (compensation: v(12) < v(1)+v(2): {v[(1, 2)][k] < v[(1,)][k] + v[(2,)][k]})')
print('weights (10 significant digits):')
for i in range(m):
    print(f'  neuron {i} (freq {lab[i]}): u1={[float(t) for t in Uq[i][0]]} u2={[float(t) for t in Uq[i][1]]} w={[float(t) for t in Wq[i]]}')
np.savez('cex_charge_rules_p5_n2_b2.npz', U=np.array([[[float(t) for t in Uq[i][tt]] for tt in range(n)] for i in range(m)]),
         W=np.array([[float(t) for t in Wq[i]] for i in range(m)]), lab=lab)
