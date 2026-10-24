# Theorem L (local regime, class C): check every step and the conclusion.
#   delta_0 in argmin_delta sum_{l in S} a_l kappa_l(delta), O = max(osc_x g(.,delta_0), osc_x g(.,-delta_0)),
#   a_l^- = L_l max{(1+V'_l)^{-(s-1)}, nu^{1-s}},  V'_l = (2/(nu L_l^2)) sum_I (|beta_{l,I}| + sqrt(2 L_l (L_l-|A_l|)))^2.
#   (L)  sum_{l in S} kappa_l(delta_0)(|A_l| - a_l^-)  <=  sum_l kappa_l(delta_0) E_l / O    ==>   gamma <= gamma_IF^{(S)}.
#   (L1, aligned) O <= nu/(2(s-1)) * Abar,  Abar = sum kappa E / sum kappa E/|A|      ==> (L).
#   (L2, any phases) sum kappa [(4(s-1)/(nu L)) E^I + (4n(s-1)-1)_+ (L-|A|)] <= sum kappa E / O   ==> (L).
import numpy as np, math, itertools, sys
from sclib import *

def intervals(n):
    out = []
    for t2 in range(1, n + 1):
        for t1 in range(0, t2):
            out.append(tuple(1 if t1 < t + 1 <= t2 else 0 for t in range(n)))
    return out

def analyse(net, S, GamS):
    p, n, nu, s = net.p, net.n, net.nu, net.s; r = 1 / s; F = (p - 1) // 2
    B = Bp(p, n)
    g = net.gaps(); h = g.min()
    Phi = fields(net); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0)
    sf = net.signflip()
    at = net.atoms(); A = np.abs(at.sum(0)); L = np.abs(at).sum(0); Q = (np.abs(at) ** r).sum(0)
    a = nu ** (1 - s) * Q ** s
    Iv = intervals(n)
    EI = np.array([sum(abs(sf[(l, T)]) ** 2 for T in Iv) for l in range(1, F + 1)])
    tau = np.sqrt(np.maximum(2 * L * (L - A), 0))
    Vp = np.array([2 / (nu * L[l] ** 2) * sum((abs(sf[(l + 1, T)]) + tau[l]) ** 2 for T in Iv) if L[l] > 0 else 0 for l in range(F)])
    aminus = L * np.maximum((1 + Vp) ** (-(s - 1)), nu ** (1 - s))
    K = kappa_mat(p); act = [l - 1 for l in S]
    d0 = int(np.argmin(K[:, act] @ a[act])); dm = p - 2 - d0
    Fd = g[:, d0] - g[:, d0].mean(); Fm = g[:, dm] - g[:, dm].mean()
    O = max(np.ptp(Fd), np.ptp(Fm))
    det = (K[d0] * E).sum() / O if O > 0 else 0.0
    lhs = (K[d0, act] * (A[act] - aminus[act])).sum()
    condL = lhs <= det + 1e-15 * A.sum()
    aligned = np.max([1 - A[l] / L[l] for l in range(F) if L[l] > 1e-9 * L.max()]) < 1e-12
    kE = K[d0, act] * E[act]
    Abar = kE.sum() / (kE / A[act]).sum() if kE.sum() > 0 else np.inf
    condL1 = aligned and (O <= nu / (2 * (s - 1)) * Abar if s > 1 else True)
    lhs2 = (K[d0, act] * (4 * (s - 1) / (nu * L[act]) * EI[act] + max(4 * n * (s - 1) - 1, 0) * (L[act] - A[act]))).sum()
    condL2 = lhs2 <= det
    gIF = nu ** (1 - s) * B * GamS
    X = (det - lhs) / (K[d0, act] * a[act]).sum()
    # d_int lower bound (Prop O(ii)): dint*N >= min_delta sum_l kappa_l E_l / O_delta
    Odel = np.array([max(np.ptp(g[:, i]), np.ptp(g[:, p - 2 - i])) for i in range(p - 1)])
    dd = decomp(net)
    dlow = np.min([(K[i] * E).sum() / Odel[i] if Odel[i] > 0 else 0.0 for i in range(p - 1)])
    chk = dict(
        quant=(1 - X) - net.gamma() / gIF if X >= 0 else 1.0,                         # gamma/gamma_IF <= 1 - X
        dint=(dd['dint'] * dd['Nn'] - dlow) / A.sum(),                                 # >= 0
        pay=np.min(a[act] - aminus[act]) / A.sum(),                                   # W6(c) + trivial bound: >= 0
        var=abs((Fd ** 2).mean() - (K[d0] * E).sum()) / A.sum() ** 2,                  # variance identity
        mz=((-Fd.min()) * (np.ptp(Fd)) - (Fd ** 2).mean()) / A.sum() ** 2,             # mean-zero lemma >= 0
        mar=((K[d0, act] * A[act]).sum() - det - h) / A.sum(),                         # margin step >= 0
        l1=(not condL1) or condL, l2=(not condL2) or condL)                            # corollaries imply (L)
    return dict(ratio=net.gamma() / gIF, condL=condL, condL1=condL1, condL2=condL2, aligned=aligned,
                Orel=O / (nu / (2 * (s - 1)) * Abar) if s > 1 else 0.0, chk=chk)

def km2(l, lam, n, nu, phase_noise=0.0, amp_noise=0.0, psi_noise=0.0, rng=None):
    out = []
    for k in range(nu):
        al = np.pi * k / nu
        amps = lam * np.exp(amp_noise * rng.normal(size=nu)) if rng is not None else lam * np.ones(nu)
        phis = al + (phase_noise * rng.normal(size=n) if rng is not None else np.zeros(n))
        out.append((l, amps, phis, psi_noise * rng.normal() if rng is not None else 0.0))
    return out

def build(p, n, specs, b):
    """specs entries (l, amps, phis, dpsi): output phase sum(phis)+dpsi (dpsi != 0 breaks alignment)."""
    x = np.arange(p); U = []; W = []
    for (l, amps, phis, dpsi) in specs:
        U.append([amps[t] * np.cos(2 * np.pi * l * x / p + phis[t]) for t in range(n)])
        W.append(amps[n] * np.cos(2 * np.pi * l * x / p + np.sum(phis) + dpsi))
    return Net(np.array(U), np.array(W), b)

rng = np.random.default_rng(11)
cases = [(3, 2, 2.0), (3, 3, 2.0), (5, 2, 2.0), (7, 2, 2.0), (11, 2, 2.0), (5, 3, 2.0), (7, 3, 2.0), (5, 4, 2.0), (7, 2, 2.5), (5, 3, 3.0), (11, 2, 2.4)]
stats = []; worst = dict(pay=np.inf, var=0, mz=np.inf, mar=np.inf, l1=True, l2=True, quant=np.inf, dint=np.inf)
for (p, n, b) in cases:
    nu = n + 1; s = nu / b; F = (p - 1) // 2
    Gam, maxs = programme_exact(p, s); y = maxs[0]; S = [j + 1 for j in range(F) if y[j] > 1e-12]
    for trial in range(72):
        eps = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0][trial % 6]
        mode = ['aligned', 'misaligned', 'singleset'][(trial // 6) % 3]
        specs = []
        for l in S:
            lam = (2 ** n * y[l - 1] / nu) ** (1.0 / nu)
            if mode == 'singleset':
                # J4: KM2 background (weight 1-eps) + n-atom single-set system (weight eps) with phase phi in slot 1
                lamA = (2 ** n * (1 - eps) * y[l - 1] / nu) ** (1.0 / nu); lamB = (2 ** n * eps * y[l - 1] / n) ** (1.0 / nu)
                if eps < 1: specs += km2(l, lamA, n, nu)
                phi = rng.uniform(0, 2 * np.pi)
                for k in range(n):
                    if eps == 0: break
                    phis = np.full(n, -np.pi * k / n); phis[0] -= phi / 2
                    specs.append((l, lamB * np.ones(nu), phis, 0.0))
            else:
                specs += km2(l, lam, n, nu, phase_noise=0.5 * eps, amp_noise=0.5 * eps,
                             psi_noise=(0.5 * eps if mode == 'misaligned' else 0.0), rng=rng)
                for k in range(rng.integers(0, 3) if eps > 0 else 0):
                    specs.append((l, lam * 0.3 * eps * np.exp(rng.normal(size=nu)), rng.uniform(0, 2 * np.pi, n),
                                  (0.5 * eps * rng.normal() if mode == 'misaligned' else 0.0)))
        net = build(p, n, specs, b)
        if net.margin() <= 0: continue
        d = analyse(net, S, Gam)
        stats.append(((p, n, b), mode, eps, d['ratio'], d['condL'], d['condL1'], d['condL2'], d['Orel']))
        for k in ['pay', 'mz', 'mar', 'quant', 'dint']: worst[k] = min(worst[k], d['chk'][k])
        worst['var'] = max(worst['var'], d['chk']['var'])
        worst['l1'] &= d['chk']['l1']; worst['l2'] &= d['chk']['l2']
print('step checks (all positive-margin class-C networks, aligned / misaligned / single-set):')
print('  min [a_l - a_l^-]/sum|A|                 (>=0) =', worst['pay'])
print('  max |E F^2 - sum kappa E|/(sum|A|)^2          =', worst['var'])
print('  min [m*osc - E F^2]/(sum|A|)^2           (>=0) =', worst['mz'])
print('  min [sum kappa|A| - det - h]/sum|A|      (>=0) =', worst['mar'])
print('  L1 => L and L2 => L on every network            :', worst['l1'], worst['l2'])
print('  min [(1-X) - gamma/gamma_IF] over X>=0     (>=0) =', worst['quant'])
print('  min [dint*N - min_d sum kappa E/O_d]/sum|A| (>=0) =', worst['dint'])
R = np.array([x[3] for x in stats]); cL = np.array([x[4] for x in stats])
print('networks: %d; (L) holds for %d; max gamma/gamma_IF where (L) holds: %.6f; where it fails: %.6f'
      % (len(R), cL.sum(), R[cL].max(), R[~cL].max() if (~cL).any() else float('nan')))
for case in cases:
    for mode in ['aligned', 'misaligned', 'singleset']:
        rr = [x for x in stats if x[0] == case and x[1] == mode and x[2] > 0]
        if rr:
            print('  %-13s %-10s n=%2d  (L)=%2d (L1)=%2d (L2)=%2d  max ratio=%.5f  max ratio|L=%.5f'
                  % (case, mode, len(rr), sum(x[4] for x in rr), sum(x[5] for x in rr), sum(x[6] for x in rr),
                     max(x[3] for x in rr), max([x[3] for x in rr if x[4]] + [0])))
