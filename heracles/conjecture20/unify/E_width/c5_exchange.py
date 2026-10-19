# Exact Renyi decomposition of the gain over the IF value for single-feature networks (S = {1}):
#   log(gamma(theta;s)/gamma_IF^{(S)}(s)) = -Delta_conv(theta) + (s-1)[log nu - H_{1/s}(m_theta)],
#   Delta_conv = log(B_p kappa_min / gamma(theta;1)) >= 0  (convex-end inefficiency),  m = neuron masses ||omega||^nu.
# Examples: the exact p=3 counterexamples (winners) and the p=5, n=6 width-6 network (LARGEN R10b; a loser).
import sys, json, numpy as np
from fractions import Fraction as Fr
from elib import *
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import exact_certificate as ec


def slot_sym_margin(U1, W, p, n):
    """U1 (m,p): same input in every slot. margin over count vectors."""
    import itertools
    m = len(U1); best = np.inf
    for N in itertools.product(range(n + 1), repeat=p - 1):
        if sum(N) > n: continue
        cnt = list(N) + [n - sum(N)]
        y = sum(a * cnt[a] for a in range(p)) % p
        z = np.prod([U1[:, a] ** cnt[a] for a in range(p)], axis=0)
        f = W.T @ z
        for c in range(p):
            if c != y: best = min(best, f[y] - f[c])
    return best


def report(name, U1, W, p, n, h=None):
    nu = n + 1; s = nu / 2
    if h is None: h = slot_sym_margin(U1, W, p, n)
    sq = n * (U1 ** 2).sum(1) + (W ** 2).sum(1); a = sq ** (nu / 2)
    kmin = 1 - np.cos(2 * np.pi / p)
    g_s = h / np.sum(sq) ** s; g_1 = h / a.sum()
    gIF = nu ** (1 - s) * Bp(p, n) * kmin
    dconv = np.log(Bp(p, n) * kmin / g_1)
    E = (s - 1) * (np.log(nu) - renyi(a, 1 / s))
    print(f"{name}: gamma/gamma_IF = {g_s / gIF:.6f};  -Delta_conv + (s-1)(log nu - H_1/s) = {-dconv:.4f} + {E:.4f} = {-dconv + E:.4f}"
          f"  vs log ratio {np.log(g_s / gIF):.4f};  m = {len(a)}, nu = {nu}, H_1/s = {renyi(a, 1 / s):.4f}")


for n in [11, 13]:
    d = ec.NETWORKS[n]
    U1 = np.array([[float(x) for x in row] for row in d['u']]); W = np.array([[float(x) for x in row] for row in d['w']])
    report(f"p=3 certificate n={n} (b=2)", U1, W, 3, n)

# p=5, n=6 width-6 network (mechanism_scripts/kkt/oned_p5_n6.json), rebuilt as in kkt/load.py
d = json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/kkt/oned_p5_n6.json'))
p, n = d['p'], d['n']; lam, lv, av = np.array(d['lam']), np.array(d['logV']), np.array(d['argV']); lv = lv - lv.max()
eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], dtype=float)
rho = np.exp(2j * np.pi * np.arange(p) / p)
U1, W = [], []
for i in range(len(lam)):
    u = eps * np.exp(lam[i] * e); w = np.real(np.exp(lv[i] + 1j * av[i]) * rho)
    nu_, nw = np.linalg.norm(u), np.linalg.norm(w); t = (nw * nu_ ** n) ** (1.0 / (n + 1))
    U1.append(u / nu_ * t); W.append(w / nw * t)
U1 = np.array(U1); W = np.array(W)
report("p=5 n=6 width-6 network (b=2)", U1, W, p, n)
print(f"   stored numerical_ratio = {d['numerical_ratio']:.6f}")
# IF block for comparison: Delta_conv = 0 and entropy term 0
for (p, n) in [(5, 6), (3, 11)]:
    U, Wb = if_block(p, n, 1)
    report(f"IF block p={p} n={n}", U[:, 0, :], Wb, p, n)
