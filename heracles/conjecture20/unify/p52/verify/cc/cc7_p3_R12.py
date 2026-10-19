"""Referee: R12 (delta-Fourier isolation, used by R14) tested on the p = 3 counterexamples (n = 11, 13; exact weights from
exact_certificate.py), which beat the IF value.  R12 claims, for every network and every non-empty T:
     h <= K_1(mu) Re beta_1 - (p/2) min mu |beta_{1,T}|,   at p = 3: mu = (1/2, 1/2), K_1 = 3/2.
The counterexamples are shared-slot networks: beta_{1,T} = 2 p^{-nu} sum_i conj(what_i(1)) conj(uhat_i(1))^{|T|} uhat_i(1)^{n-|T|}.
We also verify the exact identity sum_x q_w g = K Re beta_1 - p Re(w conj beta_{1,T}) on the full table for n = 11
(3^11 inputs) with random w, T.
"""
import sys, os, itertools
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
import importlib.util
spec = importlib.util.spec_from_file_location('ec', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'exact_certificate.py')))
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)

p = 3; rho = np.exp(2j * np.pi / 3)
for n, net in ec.NETWORKS.items():
    U = np.array([[float(a) for a in row] for row in net['u']]); W = np.array([[float(a) for a in row] for row in net['w']])
    nu = n + 1
    uh = U @ rho ** (-np.arange(3)); wh = W @ rho ** (-np.arange(3))   # hat(1)
    # margin by counts
    h = np.inf
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; y = (N1 + 2 * N2) % 3
            z = U[:, 0] ** N0 * U[:, 1] ** N1 * U[:, 2] ** N2
            f = W.T @ z
            for c in range(3):
                if c != y:
                    h = min(h, f[y] - f[c])
    beta1 = 2 * p ** (-nu) * np.sum(np.conj(wh) * uh ** n)
    worst = -np.inf
    for k in range(1, n + 1):
        bT = 2 * p ** (-nu) * np.sum(np.conj(wh) * np.conj(uh) ** k * uh ** (n - k))
        rhs = 1.5 * beta1.real - 1.5 * 0.5 * abs(bT)
        worst = max(worst, h - rhs)
    P2 = np.sum(n * np.sum(U ** 2, 1) + np.sum(W ** 2, 1))
    s = nu / 2
    gam = h / P2 ** (nu / 2)
    gIF = nu ** (1 - s) * 2 * (6 * nu) ** (-nu / 2) * 1.5
    print(f"n={n}: gamma/gamma_IF = {gam/gIF:.4f} (>1: counterexample); h={h:.6g}; max_T [h - R12 rhs] = {worst:.3e} (must be <= 0)")

# full-table identity at n = 11 with random w, T (vectorised over all 3^11 inputs)
n = 11; net = ec.NETWORKS[11]
U = np.array([[float(a) for a in row] for row in net['u']]); W = np.array([[float(a) for a in row] for row in net['w']])
rng = np.random.default_rng(3)
X = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int64)
cnt = np.stack([(X == a).sum(1) for a in range(3)], 1)
Z = np.prod(U[None, :, :] ** cnt[:, None, :], axis=2)          # (3^n, m)
f = Z @ W                                                         # (3^n, 3)
y = X.sum(1) % 3
g = {d: f[np.arange(len(X)), y] - f[np.arange(len(X)), (y - d) % 3] for d in (1, 2)}
h_full = min(g[1].min(), g[2].min())
nu = n + 1
uh = U @ rho ** (-np.arange(3)); wh = W @ rho ** (-np.arange(3))
beta1 = 2 * p ** (-nu) * np.sum(np.conj(wh) * uh ** n)
maxerr = 0
for trial in range(5):
    T = rng.choice(n, rng.integers(1, n + 1), replace=False)
    chi = np.zeros(n, int); chi[T] = 2
    k = len(T)
    bT = 2 * p ** (-nu) * np.sum(np.conj(wh) * np.conj(uh) ** k * uh ** (n - k))
    w = (rng.normal() + 1j * rng.normal()) * 0.2
    mu = {1: 0.5, 2: 0.5}
    ph = rho ** (-(X @ chi))
    tot = 0
    for d in (1, 2):
        q = p ** (-n) * (mu[d] + 2 * np.real(w * rho ** d * ph))
        tot += np.sum(q * g[d])
    K = sum(mu[d] * (1 - np.cos(2 * np.pi * d / 3)) for d in (1, 2))
    pred = K * beta1.real - p * np.real(w * np.conj(bT))
    maxerr = max(maxerr, abs(tot - pred) / max(1, abs(pred)))
print(f"n=11 full-table identity: max relative error {maxerr:.2e}; h(full table) = {h_full:.6g}")
