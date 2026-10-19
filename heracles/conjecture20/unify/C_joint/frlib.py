# helpers for the field-row theorem checks
import sys, numpy as np, itertools
from math import factorial
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_width')
import elib as EW
from cjlib import gamma_exact
GAM = {}
def Gam(p, s):
    key = (p, round(s, 12))
    if key not in GAM: GAM[key] = gamma_exact(p, s)[0] if p > 3 else 1 - np.cos(2*np.pi/3)
    return GAM[key]
def fields_x(net):
    p, n = net.p, net.n; F = (p-1)//2; f = net.table(); fc = np.fft.fft(f, axis=n)
    y = (np.indices((p,)*n).sum(0)) % p
    return np.array([(2.0/p)*np.exp(2j*np.pi*l*y/p)*fc[..., l] for l in range(1, F+1)])
def FR_terms(net, s):
    r = 1/s; Phi = fields_x(net); F = Phi.shape[0]
    psi = np.maximum(Phi.real, 0)**r
    tot = psi.reshape(F, -1).sum(0)
    return tot.min(), psi.reshape(F, -1).mean(1)
def clean_Q(net, s):
    r = 1/s; beta, xi = net.atoms()
    return (np.abs(beta)**r).sum(0)          # beta-units: sum_l Q_l <= B_p^r P_b (single-neuron budget)
def slotsym_FR(U1, W, p, n, s):
    """slot-symmetric network (same input U1[i] in every slot): margin, min_x/mean of sum_l (Re Phi_l)_+^r over x in Z_p^n
    (via count vectors with multinomial weights), clean Q_l, cost."""
    r = 1/s; F = (p-1)//2; m = len(U1); rho = np.exp(2j*np.pi/p)
    h = np.inf; Ms = np.zeros(F); tot_min = np.inf; Z = 0
    for N in itertools.product(range(n+1), repeat=p-1):
        if sum(N) > n: continue
        cnt = list(N) + [n-sum(N)]
        wgt = factorial(n)/np.prod([factorial(c) for c in cnt])
        y = sum(a*cnt[a] for a in range(p)) % p
        z = np.prod([U1[:, a]**cnt[a] for a in range(p)], axis=0)
        f = W.T@z
        h = min(h, min(f[y]-f[c] for c in range(p) if c != y))
        Phi = np.array([(2.0/p)*rho**(l*y)*np.sum(f*rho**(-l*np.arange(p))) for l in range(1, F+1)])
        psi = np.maximum(Phi.real, 0)**r
        Ms += wgt*psi; Z += wgt; tot_min = min(tot_min, psi.sum())
    Ms /= Z
    Uh = np.fft.fft(U1, axis=1); Wh = np.fft.fft(W, axis=1); nu = n+1
    beta = np.array([2*p**(-nu)*np.conj(Wh[:, j])*Uh[:, j]**n for j in range(1, F+1)]).T
    Q = (np.abs(beta)**r).sum(0)
    sq = n*(U1**2).sum(1) + (W**2).sum(1)
    return h, tot_min, Ms, Q, np.sum(sq**(nu/(2*s)))
