# Referee test of L2's consequence "class-C networks near IF never beat gamma_IF, whatever the atom phases"
# and of O's "conical stability": in EVERY neighbourhood of the aligned IF equality case there are class-C networks
# WITH interference (E > 0) on which Theorem L has X < 0 and L2's condition fails.  Take the IF curve point at
# phase-deformation eps (v3) and add interference of size eta << eps^2 (rotate one input phase of one neuron by eta while
# keeping its atom phase).  Payment side (L - |A| ~ eps^2) then dominates detection (~eta).
import numpy as np, math, itertools
from vlib import *
from v2_theoremL import ledger, intervals
from v3_ifmanifold import block_specs

def l2_terms(net, S):
    p, n, nu, s = net.p, net.n, net.nu, net.s; F = (p - 1) // 2
    g = net.gaps(); Phi = net.fields(); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0)
    at = net.atoms(); A = np.abs(at.sum(0)); L = np.abs(at).sum(0); r = 1 / s
    a = nu ** (1 - s) * ((np.abs(at) ** r).sum(0)) ** s
    K = kappa(p); act = [l - 1 for l in S]; d0 = int(np.argmin(K[:, act] @ a[act])); dm = p - 2 - d0
    O = max(np.ptp(g[:, d0]), np.ptp(g[:, dm])); det = (K[d0] * E).sum() / O if O > 0 else 0
    Iv = intervals(n)
    lhs = 0
    for l in S:
        EI = sum(abs(net.signflip_atoms(l, T).sum()) ** 2 for T in Iv)
        lhs += K[d0, l - 1] * (4 * (s - 1) / (nu * L[l - 1]) * EI + max(4 * n * (s - 1) - 1, 0) * (L[l - 1] - A[l - 1]))
    return lhs, det, E.sum() / A[act].sum() ** 2

for (p, n, b) in [(5, 2, 2.0), (7, 3, 2.0)]:
    nu = n + 1; s = nu / b; S = [1]; B = Bp(p, n); gIF = nu ** (1 - s) * B * programme_S(p, s, S)
    alphas0 = 2 * np.pi * np.arange(nu) / nu
    base = classC(p, n, block_specs(1, n, alphas0, 1.0)[0], b)
    base_par = np.concatenate([base.U.ravel(), base.W.ravel()])
    print('(p,n,b)=(%d,%d,%g)' % (p, n, b))
    print('   eps     eta      |theta-theta0|   E/|A|^2    X(TheoremL)   L2: lhs     det       gamma/gamma_IF')
    for eps in [1e-1, 3e-2, 1e-2, 3e-3]:
        eta = 0.05 * eps ** 2
        al = alphas0.copy(); al[0] += eps
        specs, z = block_specs(1, n, al, 1.0)
        l, amps, phis, psi = specs[1]; phis = phis.copy(); phis[0] += eta; psi = psi + eta    # keep atom phase
        specs[1] = (l, amps, phis, psi)
        net = classC(p, n, specs, b)
        d = ledger(net, S); lhs, det, Er = l2_terms(net, S)
        dist = np.linalg.norm(np.concatenate([net.U.ravel(), net.W.ravel()]) - base_par)
        print('   %.0e  %.1e   %.3e        %.1e    %+.3e    %.3e   %.3e   %.8f  h>0:%s' % (eps, eta, dist, Er, d['X'], lhs, det, d['gam'] / gIF, d['h'] > 0))
