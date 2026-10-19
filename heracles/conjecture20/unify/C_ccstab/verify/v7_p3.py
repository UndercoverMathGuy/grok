# Referee consistency check against the p = 3 counterexample (exact_certificate.py, n = 11, b = 2):
# S1 identities and S2(a),(b) must hold (they are general); Theorem L must not apply (not class C);
# its frequency split (class C, same beta, beta_T) obeys Theorem L's inequality.
import numpy as np, math, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from exact_certificate import NETWORKS
from vlib import *
from v2_theoremL import ledger
n = 11; p = 3; nu = 12; b = 2.0
u = np.array([[float(a) for a in row] for row in NETWORKS[n]['u']]); w = np.array([[float(a) for a in row] for row in NETWORKS[n]['w']])
U = np.repeat(u[:, None, :], n, axis=1)
net = Net(U, w, b); B = Bp(p, n); G1 = 1.5; K = kappa(p)
g = net.gaps(); h = g.min(); a = net.masses(); msh = a / a.sum(); at = net.atoms(); beta = at.sum(0)
s = net.s; gIF = nu ** (1 - s) * B * 1.5
print('gamma/gamma_IF = %.9f (expected 1.0something)' % (net.gamma() / gIF))
N = B * G1 * a.sum(); sl = g - h; sbar = sl.mean(0); ssym = 0.5 * (sbar + sbar[::-1])
cleanmin = (K @ np.real(beta)).min()
dprog = (G1 * np.real(beta).sum() - cleanmin) / N; dint = (cleanmin - h) / N
sig_n = 1 - np.real(at).sum(1) / (B * a)
print('S1: |dint - min ssym/N| = %.2e, |dprog-(mean-min)/N| = %.2e, identity err = %.2e' % (
    abs(dint - ssym.min() / N), abs(dprog - (sbar.mean() - ssym.min()) / N),
    abs(1 - h / a.sum() / (B * G1) - (msh * sig_n).sum() - sl.mean() / N)))
print('    Em_sig = %.4f, dprog = %.4f, dint = %.4f, sigma_neur range [%.3f, %.3f]' % ((msh * sig_n).sum(), dprog, dint, sig_n.min(), sig_n.max()))
Phi = net.fields(); f = Phi - Phi.mean(0); E = (np.abs(f) ** 2).mean(0)
tau = sl - sbar
print('S1 energy: rel err %.2e' % (abs(E.sum() - (2 / p ** 2) * (p * (tau ** 2).sum(1) - tau.sum(1) ** 2).mean()) / E.sum()))
print('S2(a) slack (>=0): %.3e' % ((2 * (1 - 1 / p) * sl.max() * N * (dprog + dint) - E.sum()) / N ** 2))
Fm = g - g.mean(0)
print('S2(b) rel err: %.2e' % (abs((Fm ** 2).mean() - G1 / 2 * (E.sum() + 2 * (np.real(f.sum(1)) ** 2).mean())) / (Fm ** 2).mean()))
print('class-C single-offset variance identity off class C: E F_d^2 = %.4e vs sum kappa E = %.4e' % ((Fm[:, 0] ** 2).mean(), (K[0] * E).sum()))
print('circularity E f^2 / E|f|^2 = %.3f (0 for class C)' % (abs((f[:, 0] ** 2).mean()) / E[0]))
# frequency split: project each factor to the frequency-1 plane (drop DC)
x = np.arange(p); c1 = np.cos(2 * np.pi * x / p); s1 = np.sin(2 * np.pi * x / p)
Bm = np.stack([c1 / np.linalg.norm(c1), s1 / np.linalg.norm(s1)], 1); P = Bm @ Bm.T
spl = Net(U @ P, w @ P, b)
print('split: margin h = %.4e, gamma/gamma_IF = %s' % (spl.gaps().min(), ('%.4f' % (spl.gamma() / gIF)) if spl.gaps().min() > 0 else 'n/a (h<=0)'))
d = ledger(spl, [1])
print('Theorem L on the split: X = %.4f ; h <= (1-X) M holds: %s (slack %.3e)' % (d['X'], d['bound'] >= -1e-12 * d['M'], d['bound'] / d['M']))
