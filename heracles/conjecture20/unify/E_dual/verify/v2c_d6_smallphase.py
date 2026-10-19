"""D6 random check with small non-zero common phase (many positive-margin cases)."""
"""Referee checks of D5 (entropic Lemma 7: collision bound, majorisation) and D6 (aligned class-C single features).
D5: random + adversarial (Nelder-Mead / L-BFGS on the violation) atom systems, general phases.
D6: built from ACTUAL single-frequency networks (field Phi_1 computed from the table, not from the polynomial model),
    with a NON-ZERO common atom phase, random + adversarial maximisation of c/(nu^{1-s}Q^s)."""
import numpy as np, itertools, sys, os
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *

rng = np.random.default_rng(11)


def intervals(n):
    return [(k0, k1) for k0 in range(n + 1) for k1 in range(k0 + 1, n + 1)]


# ---------------- D6 on actual networks
def single_freq_net(p, n, m, uparams, rmag, phi):
    """frequency-1 inputs u_it = a cos + b sin; output w_i = r_i cos(2 pi c/p + psi_i), psi_i set so beta_i = |.| e^{i phi}."""
    x = np.arange(p); C = np.cos(2 * np.pi * x / p); S = np.sin(2 * np.pi * x / p)
    U = uparams[:, :, :1] * C + uparams[:, :, 1:] * S                 # (m, n, p)
    Uh1 = np.fft.fft(U, axis=2)[:, :, 1]
    pr = np.prod(Uh1, axis=1)
    psi = np.angle(pr) - phi                  # conj(what) = (p/2) r e^{-i psi}; beta ~ e^{-i psi} pr
    W = rmag[:, None] * np.cos(2 * np.pi * x[None, :] / p + psi[:, None])
    return U, W


def d6_quantities(p, n, U, W, s):
    nu = n + 1; r = 1 / s
    f = table(U, W)
    idx = np.indices((p,) * n).reshape(n, -1).T; y = idx.sum(1) % p
    rho = np.exp(2j * np.pi / p)
    ff = f.reshape(-1, p)
    Phi = (2 / p) * rho ** (1 * y) * (ff * rho ** (-np.arange(p))[None, :]).sum(1)
    cmv = Phi.real - np.abs(Phi.imag) / np.tan(np.pi / p)
    c = cmv.min()
    fh = np.fft.fftn(f)
    bT = {}
    for T in itertools.product([0, 1], repeat=n):
        k = tuple((p - 1) if T[t] else 1 for t in range(n)) + (p - 1,)
        bT[T] = 2 * p ** (-nu) * fh[k]
    A = bT[(0,) * n]
    at = atoms(U, W)[:, 0]
    Q = np.sum(np.abs(at) ** r)
    S2 = sum(abs(v) ** 2 for T, v in bT.items() if any(T))
    I2 = sum(abs(bT[tuple(1 if k0 <= t < k1 else 0 for t in range(n))]) ** 2 for (k0, k1) in intervals(n))
    return c, A, Q, S2, I2, at, Phi


worstA = worstB = worstC = -np.inf; ncase = 0; ncpos = 0; checkPhi = 0
for trial in range(3000):
    p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])) if p < 11 else 2; nu = n + 1
    m = int(rng.integers(nu, 2 * nu + 1)); phi = rng.uniform(-0.4, 0.4)
    taup = (1 - np.cos(2 * np.pi / p)) / np.cos(2 * np.pi / p)
    # near-KM2 start sometimes, random otherwise
    if m >= nu:
        up = np.zeros((m, n, 2))
        for i in range(m):
            l = i % nu; ph = -np.pi * l / nu + rng.normal(0, 0.3)
            up[i, :, 0] = np.cos(ph); up[i, :, 1] = -np.sin(ph)
            up[i] += rng.normal(0, rng.uniform(0, 0.25), (n, 2))
        rm = np.exp(rng.normal(0, 0.3, m))
    else:
        up = rng.standard_normal((m, n, 2)); rm = np.exp(rng.normal(0, 0.5, m))
    U, W = single_freq_net(p, n, m, up, rm, phi)
    for s in [1.0 + 0.5 * rng.random() * (nu / 2 - 1) * 2, nu / 2, 1.25]:
        s = min(max(s, 1.0), nu / 2)
        c, A, Q, S2, I2, at, Phi = d6_quantities(p, n, U, W, s)
        ncase += 1
        # aligned check
        assert np.allclose(np.angle(at * np.exp(-1j * phi)), 0, atol=1e-8) or np.allclose(abs(np.angle(at * np.exp(-1j * phi))), 0, atol=1e-8)
        checkPhi = max(checkPhi, abs(Phi.mean() - np.conj(A)))
        worstB = max(worstB, (abs(A) - nu ** (1 - s) * Q ** s - 2 * (s - 1) / (nu * abs(A)) * I2) / abs(A))
        if c > 0:
            ncpos += 1
            worstA = max(worstA, (S2 - taup * (A.real - c) ** 2) / abs(A) ** 2)
            if 2 * (s - 1) * taup <= nu:
                worstC = max(worstC, c / (nu ** (1 - s) * Q ** s) - 1)
print('D6 random: %d cases, %d with c>0; E Phi = conj(A) err %.1e' % (ncase, ncpos, checkPhi))
print('   (a) max (sum|b_T|^2 - tau (ReA-c)^2)/|A|^2 = %.3e ; (b) max rel = %.3e ; (c) max c/(nu^{1-s}Q^s)-1 = %.3e'
      % (worstA, worstB, worstC))

