"""Checks for D5 (entropic Lemma 7 / chain-Frobenius collision bound), D6 (static entropic gain bound),
D4 (s = 1 maximisers have entropy >= log nu + log F) and D5' (aligned single features).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy --with mpmath python check_entropic.py"""
import itertools, numpy as np
from elib import G, nu_costs, shares, shannon, renyi

rng = np.random.default_rng(7)


def intervals(n):
    return [(k0, k1) for k0 in range(n + 1) for k1 in range(k0 + 1, n + 1)]   # I = {k0+1..k1} (slots 1-based)


def atom_data(beta, xi):
    """beta (N,), xi (N,n) unit; beta_T = sum_l beta_l prod_{t in T} conj(xi_lt)."""
    n = xi.shape[1]
    bI = []
    for (k0, k1) in intervals(n):
        bI.append(np.sum(beta * np.prod(np.conj(xi[:, k0:k1]), axis=1)))
    return np.array(bI)


print('(1) entropic Lemma 7:  sum (|b_l|/L)^2 <= 1/nu + (2/nu^2) sum_I (|b_I|/L + sqrt(2(1-alpha)))^2')
worst = -np.inf; worst_al = -np.inf; eqIF = 0
for trial in range(20000):
    n = int(rng.integers(1, 7)); nu = n + 1; N = int(rng.integers(1, 3 * nu))
    mode = rng.integers(0, 4)
    mod = rng.exponential(1, N) ** rng.uniform(0.3, 3)
    if mode == 0:     # aligned, random xi
        beta = mod * np.exp(1j * 0.3); xi = np.exp(2j * np.pi * rng.random((N, n)))
    elif mode == 1:   # random phases
        beta = mod * np.exp(2j * np.pi * rng.random(N)); xi = np.exp(2j * np.pi * rng.random((N, n)))
    elif mode == 2:   # near-IF: KM2 block(s) + noise
        k = int(rng.integers(1, 3)); bl = []; xl = []
        for q in range(k):
            wq = rng.exponential(1); c = np.exp(2j * np.pi * rng.random(n)); om = np.exp(2j * np.pi / nu)
            for l in range(nu):
                bl.append(wq / nu); xl.append(c * om ** l)
        beta = np.array(bl) * np.exp(1j * rng.normal(0, 0.05, len(bl))) * (1 + 0.05 * rng.standard_normal(len(bl)))
        xi = np.array(xl) * np.exp(1j * rng.normal(0, 0.05, (len(bl), n))); N = len(bl)
    else:             # exact IF (equality test): one KM2 block, aligned
        c = np.exp(2j * np.pi * rng.random(n)); om = np.exp(2j * np.pi / nu)
        beta = np.full(nu, 1.0 / nu) * np.exp(1j * 1.1); xi = np.array([c * om ** l for l in range(nu)]); N = nu
    L = np.abs(beta).sum(); A = beta.sum(); al = abs(A) / L
    bI = atom_data(beta, xi)
    lhs = np.sum((np.abs(beta) / L) ** 2)
    rhs = 1 / nu + 2 / nu ** 2 * np.sum((np.abs(bI) / L + np.sqrt(max(0, 2 * (1 - al)))) ** 2)
    worst = max(worst, lhs - rhs)
    if mode in (0, 3):
        rhs_al = 1 / nu + 2 / nu ** 2 * np.sum((np.abs(bI) / L) ** 2)
        worst_al = max(worst_al, lhs - rhs_al)
    if mode == 3:
        eqIF = max(eqIF, abs(lhs - 1 / nu))
print('   max(lhs - rhs) general = %.2e ; aligned (no phase term) = %.2e ; KM2 block |lhs - 1/nu| = %.1e'
      % (worst, worst_al, eqIF))

print('(2) static gain bound: gamma/gamma_IF <= [nu K_r(yhat)/K_r(p)]^{s-1} * M(yhat)/(||yhat||_r Gamma_s)')
worst = -np.inf; cnt = 0
for (p, n) in [(5, 2), (7, 2), (5, 3), (7, 3), (11, 2)]:
    nu = n + 1; F = (p - 1) // 2; K = G.kappa_mat(p)[:F]
    for b in [2.0, 2.5, nu - 0.3]:
        s = nu / b; r = 1 / s
        Gam, y, _ = G.programme(p, s); gIF = G.gamma_if(p, n, b, Gam)
        for t in range(25):
            yy = y * rng.uniform(0.2, 1.0, F) * (rng.random(F) < 0.8)
            if yy.sum() == 0: yy = y
            net = G.if_network(p, n, yy, b)
            sc = rng.uniform(0, 0.3)
            U = net.U + sc * rng.standard_normal(net.U.shape) * np.abs(net.U).mean()
            W = net.W + sc * rng.standard_normal(net.W.shape) * np.abs(net.W).mean()
            U = U * np.exp(rng.normal(0, 0.4, (U.shape[0], 1, 1)))      # unequal neuron sizes
            net = G.Net(U, W, b); h = net.margin()
            if h <= 0: continue
            a = nu_costs(net); pp = a / a.sum(); A = np.abs(net.clean()); yh = A / A.sum()
            lhs = net.gamma() / gIF
            Kp = np.exp(renyi(pp, r)); Ky = np.exp(renyi(yh, r))
            rhs = (nu * Ky / Kp) ** (s - 1) * (K @ yh).min() / ((yh ** r).sum() ** s * Gam)
            worst = max(worst, lhs / rhs); cnt += 1
print('   %d positive-margin networks: max lhs/rhs = %.6f (must be <= 1)' % (cnt, worst))
# p = 3 certified networks (exact certificate weights, n = 11 and 13)
try:
    import json, os
    path = '/home/user/grok/heracles/conjecture20/oned_certified'
    fs = sorted(f for f in os.listdir(path) if f.endswith('.json'))[:2]
    print('   p=3 files:', fs)
except Exception as e:
    print('   p=3 files not read:', e)

print('(3) s = 1 (b = nu) maximisers: entropy of nu-cost shares >= log nu + log F')
for (p, n, m) in [(5, 2, 8), (7, 2, 10), (5, 3, 10)]:
    nu = n + 1; F = (p - 1) // 2; b = float(nu)
    best = []
    for t in range(4):
        U0 = rng.standard_normal((m, n, p)); W0 = rng.standard_normal((m, p))
        net = G.optimise(U0, W0, b, iters=3000)
        a = nu_costs(net); keep = a > 1e-9 * a.max(); a = a[keep]
        best.append((net.gamma() / (G.Bp(p, n) * p / (p - 1)), shannon(a / a.sum()), keep.sum()))
    for g, H, k in best:
        print('   p=%d n=%d  gamma/(B_p Gamma_1)=%.6f  H(p)=%.4f  log(nu F)=%.4f  width=%d' % (p, n, g, H, np.log(nu * F), k))
