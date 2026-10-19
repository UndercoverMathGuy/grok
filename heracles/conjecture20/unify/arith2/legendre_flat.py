# Why efficiency must be used: a perfectly flat first difference with flat spectrum (Legendre symbol).
# p = 1 mod 4: chi(w) = (w/p) is real, even, +-1 off 0.  Choose phi with D_1 = phi - phi(.-1) = chi (possible since sum chi = 0):
#   phi_hat(l) = chi_hat(l)/(1 - e^{-2 pi i l/p}).  Then v_l = 2|phi_hat(l)| on all of F.
# Report: bonus c (K v)_1 versus detection (1/2) osc D_1 = 1, and the efficiency of the cheapest allocation a = v/tau'
# that can carry this interference (two-atom cap), e(a) = m(a)/(Gamma ||a||).
import numpy as np
from lib import *
s = 1.5; r, c, tp = consts(s)
for p in [61, 101, 211, 401, 1009]:
    if p % 4 != 1: continue
    F = (p-1)//2; w = np.arange(p)
    chi = np.array([0] + [1 if pow(int(x), (p-1)//2, p) == 1 else -1 for x in range(1, p)], float)
    ch = np.fft.fft(chi)/p                         # chi(w) = sum_k ch[k] e^{2 pi i k w/p}
    l = np.arange(1, F+1)
    ph = ch[l]/(1-np.exp(-2j*np.pi*l/p))           # phi_hat(l), phi real => phi_hat(-l) = conj
    v = 2*np.abs(ph); th = np.angle(ph)
    phi = (v[:, None]*np.cos(2*np.pi*np.outer(l, w)/p + th[:, None])).sum(0)
    D = phi - np.roll(phi, 1)
    K = kmat(p)
    bonus = c*(v*K[0]).sum(); det = 0.5*(D.max()-D.min()); pars = np.sqrt((v**2*K[0]).sum())
    a = v/tp; G = load_support(p)[0]
    e = (K @ a).min()/(G*norm_r(a, s))
    print(f"p={p}: max|D_1 - chi| = {np.abs(D-chi).max():.1e}; detection (1/2)osc D_1 = {det:.3f}, Parseval sqrt(E D^2) = {pars:.3f}, "
          f"bonus c(Kv)_1 = {bonus:.3f}  (ratio {bonus/det:.2f});  efficiency of a = v/tau': e(a) = {e:.4f}  (closed if e <= 0.8165)")
