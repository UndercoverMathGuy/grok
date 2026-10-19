# p=3 counterexamples (exact_certificate.py weights, n=11 and n=13, b=2): they must be cheap (no-cheap chain gives
# gamma <= nu^{1-s} B_p Gamma = gamma_IF for EVERY network), must satisfy R7(b), and are not class C (input DC).
import sys, importlib.util, numpy as np, mpmath as mp
mp.mp.dps = 40
spec = importlib.util.spec_from_file_location('ec', '/home/user/grok/heracles/conjecture20/exact_certificate.py')
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
from fractions import Fraction as Fr
p = 3; rho = mp.exp(2j * mp.pi / 3)
for n, net in ec.NETWORKS.items():
    nu = n + 1; b = 2; s = mp.mpf(nu) / b; r = 1 / s
    U = [[mp.mpf(a) for a in row] for row in net['u']]; W = [[mp.mpf(a) for a in row] for row in net['w']]
    m = len(U)
    hat = lambda v, k: sum(v[x] * rho ** (-k * x) for x in range(3))
    beta = [2 * mp.mpf(3) ** (-nu) * mp.conj(hat(W[i], 1)) * hat(U[i], 1) ** n for i in range(m)]
    bT = {k: abs(sum(2 * mp.mpf(3) ** (-nu) * mp.conj(hat(W[i], 1)) * hat(U[i], 1) ** (n - k) * hat(U[i], -1) ** k for i in range(m))) for k in range(1, n + 1)}
    sq = [n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(m)]
    P = sum(q ** (mp.mpf(b) / 2) for q in sq)
    # margin by counts (as in exact_certificate)
    h = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; y = (N1 + 2 * N2) % 3
            z = [U[i][0] ** N0 * U[i][1] ** N1 * U[i][2] ** N2 for i in range(m)]
            f = [sum(W[i][c] * z[i] for i in range(m)) for c in range(3)]
            for c in range(3):
                if c != y: h = f[y] - f[c] if h is None else min(h, f[y] - f[c])
    gam = h / P ** s
    B = 2 * (6 * mp.mpf(nu)) ** (-mp.mpf(nu) / 2); Gam = mp.mpf(1.5)
    gIF = mp.mpf(nu) ** (1 - s) * B * Gam
    Q = sum(abs(bb) ** r for bb in beta); bl = abs(sum(beta))
    Cb = B ** r
    v = [abs(bb) ** r / (Cb * P) for bb in beta]
    R7b = B * Gam * sum(vi ** s for vi in v)
    dc = max(abs(sum(U[i])) / mp.sqrt(sum(a * a for a in U[i])) for i in range(m))
    print(f'n={n}: gamma/gIF = {mp.nstr(gam/gIF, 10)}; cheapness |beta|/(nu^(1-s)Q^s) = {mp.nstr(bl/(mp.mpf(nu)**(1-s)*Q**s), 8)} '
          f'(must be > 1); R7(b): gamma/bound = {mp.nstr(gam/R7b, 8)} (<= 1); universal B_p Gamma bound ratio {mp.nstr(gam/(B*Gam),6)}; '
          f'max input DC ratio {mp.nstr(dc,4)} (not class C); max_T|beta_T|/|beta| = {mp.nstr(max(bT.values())/bl,4)}')
