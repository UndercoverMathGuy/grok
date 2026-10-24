"""Referee check: (1) mean-gap identity E_{x,delta} g = p/(p-1) sum_{j in F} Re beta_j (used in D3b, D4');
(2) Paper Lemma 2 h <= sum_j Re beta_j kappa_j(delta) for every delta (arbitrary inputs, DC, any p);
(3) r=1 single-neuron budget sum_j |beta_ij| <= B_p ||omega_i||^nu;
(4) D7: gamma(theta;s) <= B_p M(yhat) K_{1/s}(p_theta)^{1-s} for s in {1, 1.3, nu/2} and random b-independent tables;
(5) D1(i)/(ii) Renyi and Gibbs forms on the same networks;
(6) D7 on the exact p=3 certificates (n = 11, 13)."""
import numpy as np
from fractions import Fraction as Fr
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *

rng = np.random.default_rng(2024)
w1 = w2 = w3 = w4 = w5 = -np.inf; npos = 0; cnt = 0
for trial in range(3000):
    p = int(rng.choice([3, 5, 7, 11])); n = int(rng.choice([2, 3])) if p <= 7 else 2
    m = int(rng.integers(1, 10)); nu = n + 1
    mode = rng.integers(0, 3)
    U = rng.standard_normal((m, n, p)); W = rng.standard_normal((m, p))
    if mode == 1:   # near-IF-ish: frequency-pure blocks with noise
        x = np.arange(p)
        for i in range(m):
            j = int(rng.integers(1, (p - 1) // 2 + 1))
            for t in range(n):
                U[i, t] = np.cos(2 * np.pi * j * x / p + rng.uniform(0, 2 * np.pi)) + 0.2 * rng.standard_normal(p)
            W[i] = np.cos(2 * np.pi * j * x / p + rng.uniform(0, 2 * np.pi)) + 0.2 * rng.standard_normal(p)
    if mode == 2:
        U *= np.exp(rng.normal(0, 1.0, (m, 1, 1)))
    if trial % 2 == 0:   # positive-margin family: perturbed IF blocks, random support, unequal sizes, DC
        F = (p - 1) // 2
        y = rng.uniform(0.1, 1.0, F) * (rng.random(F) < 0.8)
        if y.sum() == 0: y[0] = 1.0
        k = int((y > 0).sum()) * nu
        U, W = if_block_net(p, n, y, scales=np.exp(rng.normal(0, rng.uniform(0, 0.5), k)), rng=rng,
                            noise=rng.uniform(0, 0.25), dc=rng.uniform(0, 0.2) * (rng.random() < 0.5))
        m = U.shape[0]
    f = table(U, W); g = gaps(f, p, n); h = g.min(); beta = clean(f, p, n)
    # (1) mean gap
    w1 = max(w1, abs(g.mean() - p / (p - 1) * beta.real.sum()) / (1 + abs(g).max()))
    # (2) Lemma 2: for every delta, h <= sum_j Re beta_j kappa_j(delta)
    K = kap(p)
    w2 = max(w2, (h - (K @ beta.real).min()) / (1 + abs(h)))
    # (3) budget
    at = atoms(U, W); a = acost(U, W)
    w3 = max(w3, (np.abs(at).sum(1) / (Bp(p, n) * a)).max())
    # check atoms sum to clean
    w5 = max(w5, abs(at.sum(0) - beta).max() / (1 + abs(beta).max()))
    if h > 0:
        npos += 1
        yh = np.abs(beta) / np.abs(beta).sum(); pp = a / a.sum()
        for s in [1.0, 1.3, nu / 2]:
            r = 1 / s
            gam = h / (a ** r).sum() ** s
            bound = Bp(p, n) * Mfun(yh, p) * np.exp(renyi(pp, r)) ** (1 - s)
            w4 = max(w4, gam / bound); cnt += 1
print('(1) mean-gap identity max rel err   = %.2e' % w1)
print('(2) Lemma 2 max (h - min_d sum Re beta kappa) = %.2e  (must be <= 0)' % w2)
print('(3) r=1 budget max ratio            = %.6f (must be <= 1)' % w3)
print('    atoms sum to clean coefficients: max err %.1e' % w5)
print('(4) D7 max gamma/bound over %d (network,s) with h>0 (%d nets) = %.6f (must be <= 1)' % (cnt, npos, w4))

# (5) D1 identities at 50 digits
import mpmath as mp
mp.mp.dps = 50
worst = 0; worstG = mp.mpf(0); minex = mp.inf
for t in range(100):
    m = int(rng.integers(1, 25)); s = mp.mpf(1) + 3 * mp.mpf(rng.random())
    a = [mp.mpf(rng.exponential()) ** mp.mpf(rng.uniform(0.1, 5)) for _ in range(m)]
    Z1 = sum(a); pp = [x / Z1 for x in a]; r = 1 / s
    lhs = s * mp.log(sum(x ** r for x in a))
    Hr = mp.log(sum(q ** r for q in pp)) / (1 - r)
    worst = max(worst, abs(lhs - mp.log(Z1) - (s - 1) * Hr))
    Zs = sum(q ** r for q in pp); w = [q ** r / Zs for q in pp]
    H = -sum(x * mp.log(x) for x in w); KL = sum(x * mp.log(x / q) for x, q in zip(w, pp))
    # Gibbs: s log sum p^{1/s} = max_pi [(s-1) H(pi) - KL(pi||p)], max at w
    worstG = max(worstG, abs(s * mp.log(Zs) - ((s - 1) * H - KL)))
    for k in range(20):
        pi = rng.dirichlet(np.ones(m) * rng.uniform(0.1, 3)); pi = [mp.mpf(x) for x in pi]; sp = sum(pi); pi = [x / sp for x in pi]
        val = sum(x * mp.log(x) for x in pi if x > 0) * (s - 1) * (-1) - sum(x * mp.log(x / q) for x, q in zip(pi, pp) if x > 0)
        minex = min(minex, s * mp.log(Zs) - val)
print('(5) D1(i) max err %s ; Gibbs equality at w_s err %s ; min over random pi of [max - value] = %s (>=0)'
      % (mp.nstr(worst, 3), mp.nstr(worstG, 3), mp.nstr(minex, 5)))

# (6) D7 at the exact p=3 certificates, b=2
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import importlib.util
spec = importlib.util.spec_from_file_location('ec', '/home/user/grok/heracles/conjecture20/exact_certificate.py')
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
for n, net in ec.NETWORKS.items():
    p = 3; nu = n + 1; s = nu / 2; r = 1 / s
    U1 = np.array([[float(Fr(x)) for x in row] for row in net['u']]); W = np.array([[float(Fr(x)) for x in row] for row in net['w']])
    m = len(W)
    # exact margin via count vectors (same input factor in every slot)
    Uq = [[Fr(x) for x in row] for row in net['u']]; Wq = [[Fr(x) for x in row] for row in net['w']]
    h = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; y = (N1 + 2 * N2) % 3
            z = [Uq[i][0] ** N0 * Uq[i][1] ** N1 * Uq[i][2] ** N2 for i in range(m)]
            ff = [sum(Wq[i][c] * z[i] for i in range(m)) for c in range(3)]
            for c in range(3):
                if c != y and (h is None or ff[y] - ff[c] < h): h = ff[y] - ff[c]
    h = float(h)
    S = n * (U1 ** 2).sum(1) + (W ** 2).sum(1); a = S ** (nu / 2); pp = a / a.sum()
    gam = h / (a ** r).sum() ** s
    gIF = nu ** (1 - s) * Bp(3, n) * 1.5
    # p=3: one feature, M(yhat) = kappa_min = 3/2; clean beta_1 via atoms of the symmetric neuron
    Uh = np.fft.fft(U1, axis=1); Wh = np.fft.fft(W, axis=1)
    at = 2 * 3.0 ** (-nu) * np.conj(Wh[:, 1]) * Uh[:, 1] ** n
    beta1 = at.sum()
    bound = Bp(3, n) * 1.5 * np.exp(renyi(pp, r)) ** (1 - s)
    # Lemma 2 at p=3: h <= Re beta_1 * 3/2
    print('(6) p=3 n=%d: gamma/gIF=%.6f  gamma/D7bound=%.6f  K_r=%.4f  h/(1.5 Re beta_1)=%.6f  budget max=%.4f'
          % (n, gam / gIF, gam / bound, np.exp(renyi(pp, r)), h / (1.5 * beta1.real),
             (np.abs(at) / (Bp(3, n) * a)).max()))
