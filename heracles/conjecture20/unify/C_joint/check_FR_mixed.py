# Does (FR) with the CLEAN budget hold beyond class C?  gamma <= gamma_IF * (sum_l Q_l Xi_l / sum Q_l)^s * (sum Q_l/(B_p^r P_b))^s
# (the last factor <= 1 by the single-neuron budget).  Test on positive-margin mixed networks: IF networks + random
# mixed/DC perturbations, the (5,2,2) violator, local maxima, p=3 winners.
import sys, glob, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from frlib import *
rng = np.random.default_rng(23)
def FRratio(net, s, h=None):
    """returns gamma / [B_p Gamma_s (sum_l M_l / sum_l Q_l)^s]  (the (FR) bound with the clean single-neuron budget)."""
    p, n = net.p, net.n; nu = n+1; r = 1/s; G = Gam(p, s)
    h = net.margin() if h is None else h
    mn, M = FR_terms(net, s); Q = clean_Q(net, s); P = net.cost(nu/s); Bp = EW.Bp(p, n)
    g = h/P**s
    return g/(Bp*G*(M.sum()/Q.sum())**s), Q.sum()/(Bp**r*P)
worst = -1e9; cnt = 0
for trial in range(80):
    p, n = [(5, 2), (7, 2), (5, 3), (11, 2)][trial % 4]; s = rng.uniform(1, (n+1)/2)
    G0, ys = gamma_exact(p, s); U, W = EW.if_network(p, n, ys[0])
    eps = rng.choice([0.01, 0.05, 0.2])
    U = U + eps*np.abs(U).max()*rng.normal(size=U.shape); W = W + eps*np.abs(W).max()*rng.normal(size=W.shape)
    net = EW.Net(U, W); h = net.margin()
    if h <= 0: continue
    a, b = FRratio(net, s, h); worst = max(worst, a - 1); cnt += 1
print(f"mixed perturbations of IF networks ({cnt} with h>0): max gamma/(FR bound with clean budget) - 1 = {worst:.3e}")
d = np.load('/home/user/grok/heracles/conjecture20/unify/split/viol_p5_n2_b2.npz')
net = EW.Net(d['U0'], d['W0']); a, b = FRratio(net, 1.5)
print(f"(5,2,2) violator: gamma/(FR bound, clean budget) = {a:.4f}  (budget ratio {b:.4f})")
for f in sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p*_k*.npz')):
    dd = np.load(f); net = EW.Net(dd['U'], dd['W']); nu = net.n+1; s = nu/float(dd['b'])
    if net.margin() <= 0: continue
    a, b = FRratio(net, s); print(f"   {f.split('/')[-1]}: gamma/(FR bound, clean budget) = {a:.4f} (budget ratio {b:.4f})")

import exact_certificate as ec
for n in [11, 13]:
    dd = ec.NETWORKS[n]
    U1 = np.array([[float(x) for x in row] for row in dd['u']]); W = np.array([[float(x) for x in row] for row in dd['w']])
    s = (n+1)/2; p = 3
    h, mn, M, Q, P = slotsym_FR(U1, W, p, n, s); G = Gam(3, s); g = h/P**s
    print(f"p=3 n={n} winner: gamma/(FR bound) = {g/(EW.Bp(p,n)*G*(M.sum()/Q.sum())**s):.4f}")
