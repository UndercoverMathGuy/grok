# BRIEF_CORE section 3 and stored adversarial networks: identity decomposition, class-C / alignment status,
# the Theorem-L oscillation ratio (only meaningful for aligned class C), and the slack-transport inequality (n = 2).
import numpy as np, math, glob, itertools, os
from sclib import *

U = '/home/user/grok/heracles/conjecture20/unify/'

def classC_defect(net):
    """fraction of neuron energy outside the dominant frequency plane (incl. DC) - 0 for class C."""
    p = net.p; F = (p - 1) // 2
    tot = 0; off = 0
    for i in range(net.m):
        fac = list(net.U[i]) + [net.W[i]]
        e = np.zeros(F + 1)
        for v in fac:
            vh = np.fft.fft(v); e[0] += abs(vh[0]) ** 2 / p
            for k in range(1, F + 1): e[k] += 2 * abs(vh[k]) ** 2 / p
        tot += e.sum(); off += e.sum() - e[1:].max()
    return off / tot

def phase_spread(net):
    at = net.atoms(); out = 0
    for l in range(at.shape[1]):
        a = at[:, l]; a = a[np.abs(a) > 1e-6 * max(np.abs(at).max(), 1e-300)]
        if len(a) > 1: out = max(out, 1 - abs(a.sum()) / np.abs(a).sum())
    return out

def osc_ratio(net, S):
    p, n, nu, s = net.p, net.n, net.nu, net.s; r = 1 / s
    g = net.gaps(); Phi = fields(net); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0)
    at = net.atoms(); A = np.abs(at.sum(0)); Q = (np.abs(at) ** r).sum(0); a = nu ** (1 - s) * Q ** s
    K = kappa_mat(p); act = [l - 1 for l in S]
    d0 = int(np.argmin(K[:, act] @ a[act])); dm = p - 2 - d0
    O = max(np.ptp(g[:, d0]), np.ptp(g[:, dm])); kE = K[d0, act] * E[act]
    if kE.sum() <= 1e-14 * A.sum() ** 2: return 0.0
    Abar = kE.sum() / (kE / np.maximum(A[act], 1e-300)).sum()
    return O / (nu / (2 * (s - 1)) * Abar)

rows = []
for f in sorted(glob.glob(U + 'split/viol_p*_n2_b2.npz')):
    d = np.load(f)
    for kk in range(4): rows.append((os.path.basename(f)[:-4] + ':%d' % kk, Net(d['U%d' % kk], d['W%d' % kk], 2.0)))
for f in sorted(glob.glob(U + 'E_kkt/lm_*.npz')):
    d = np.load(f); rows.append((os.path.basename(f)[:-4], Net(d['U'], d['W'], float(d['b']))))

print('%-26s %8s %8s %8s %8s %8s %9s %9s %8s' % ('network', 'g/gIF', 'Em_sig', 'dprog', 'dint', 'H1/s', 'Cdefect', 'phasedef', 'oscRat'))
for name, net in rows:
    p, n, b = net.p, net.n, net.b; s = net.s
    if net.margin() <= 0: continue
    Gs, _ = programme_exact(p, s)
    gIF = (n + 1) ** (1 - s) * Bp(p, n) * Gs
    dd = decomp(net)
    beta = net.clean(); S = [l + 1 for l in range(len(beta)) if abs(beta[l]) > 1e-6 * np.abs(beta).max()]
    split = G.split(net)
    rows_out = (name, net.gamma() / gIF, dd['Em_sig'], dd['dprog'], dd['dint'], renyi(dd['m'], 1 / s), classC_defect(net), phase_spread(net), osc_ratio(split, S))
    print('%-26s %8.5f %8.5f %8.5f %8.5f %8.5f %9.2e %9.2e %8.3f' % rows_out)

# slack-transport inequality (n = 2):  S(s) - S(s1) >= int_s^{s1} [H(Theta*,sigma) - H(theta,sigma)] dsigma
print('\nslack transport at n=2, s1 = 1.5 (Theta* = decoupled b=2 maximiser), on the stored n=2 networks:')
worst = np.inf
for name, net in rows:
    if net.n != 2 or net.margin() <= 0: continue
    p = net.p; B = Bp(p, 2); a = masses(net); m = a / a.sum(); g1 = net.margin() / a.sum()
    G15, ys = programme_exact(p, 1.5); yh = ys[0] / ys[0].sum()
    def logD(s): return math.log(programme_exact(p, s)[0]) - (s - 1) * math.log(3) + math.log(B)
    def Sth(s): return logD(s) - math.log(g1) + (s - 1) * renyi(m, 1 / s)
    def Hesc(w, s): z = w[w > 0] ** (1 / s); z /= z.sum(); return float(-(z * np.log(z)).sum())
    for s in [1.05, 1.2, 1.35]:
        sig = np.linspace(s, 1.5, 401)
        integ = np.trapezoid([math.log(3) + Hesc(yh, q) - Hesc(m, q) for q in sig], sig)
        lhs = Sth(s) - Sth(1.5)
        worst = min(worst, lhs - integ)
print('  min over networks and s of [S(s)-S(1.5) - int(H*-H)] =', worst, '(>= 0 up to quadrature error)')

# joint detection tightness on the stored networks: ratio  2(1-1/p) smax N D' / sum E  (>= 1), and spike size smax/N
print('\njoint detection on stored networks: J = 2(1-1/p) smax N Dprime / sum E (>=1);  smax/N;  Dprime')
for name, net in rows:
    if net.margin() <= 0: continue
    d = decomp(net); Phi = fields(net); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0).sum()
    if E < 1e-12 * d['Nn'] ** 2: continue
    smax = (net.gaps() - net.margin()).max(); Dp = d['dprog'] + d['dint']; p = net.p
    print('  %-26s J=%7.3f  smax/N=%6.3f  Dprime=%.4f' % (name, 2 * (1 - 1 / p) * smax * d['Nn'] * Dp / E, smax / d['Nn'], Dp))

# where do the stored coupled networks lose?  (frequency split, class C)  at delta_0:
#   cheap = sum_S kappa(|A|-a) (true IF-price cheapness), det = sum kappa E / O, eff_a = min_d sum a kappa / (Gamma_s (sum a^r)^s)
print('\nTheorem-L ledger on the frequency splits: cheap vs det at delta_0, and programme efficiency of the IF-price allocation a')
for name, net0 in rows:
    if net0.margin() <= 0: continue
    net = G.split(net0); p, n, nu, s = net.p, net.n, net.nu, net.s; r = 1 / s; F = (p - 1) // 2
    if net.margin() <= 0: continue
    at = net.atoms(); A = np.abs(at.sum(0)); Q = (np.abs(at) ** r).sum(0); a = nu ** (1 - s) * Q ** s
    S = [l + 1 for l in range(F) if A[l] > 1e-6 * A.max()]; act = [l - 1 for l in S]
    if len(S) == 0: continue
    g = net.gaps(); Phi = fields(net); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0)
    K = kappa_mat(p); d0 = int(np.argmin(K[:, act] @ a[act])); dm = p - 2 - d0
    O = max(np.ptp(g[:, d0]), np.ptp(g[:, dm]))
    if (K[d0] * E).sum() < 1e-12 * A.sum() ** 2: continue
    cheap = (K[d0, act] * (A[act] - a[act])).sum(); det = (K[d0] * E).sum() / O
    Gs, _ = programme_exact(p, s, S=S)
    eff = (K[d0, act] @ a[act]) / (Gs * (a[act] ** r).sum() ** s)
    gIF = nu ** (1 - s) * Bp(p, n) * programme_exact(p, s)[0]
    print('  %-26s split g/gIF=%.4f  cheap=%+.4f det=%.4f (units sum|A|)  eff_a(S)=%.4f  Gamma^S/Gamma=%.4f'
          % (name, net.gamma() / gIF, cheap / A.sum(), det / A.sum(), eff, Gs / programme_exact(p, s)[0]))
