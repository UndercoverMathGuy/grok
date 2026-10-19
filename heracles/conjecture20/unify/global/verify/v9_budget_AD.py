# R6 Corollary (a) ingredients on random ARBITRARY networks (any inputs, DC, all frequencies):
#  (1) single-neuron budget  sum_j |beta_ij|^r <= C_b ||omega_i||^b,  C_b = B_p^{1/s}   (cited; tested)
#  (2) Lemma 2:  h <= min_delta sum_j Re(beta_j) kappa_j(delta)
#  (3) AD with P_j = Q_j / C_b:  h <= B_p min_delta sum_j P_j^s kappa_j(delta),  sum_j P_j <= P
import math, numpy as np
from vnet import *

rng = np.random.default_rng(9)
w1 = w2 = w3 = w4 = -np.inf
for trial in range(300):
    p = int(rng.choice([3, 5, 7])); n = int(rng.choice([2, 3])); m = int(rng.integers(1, 6)); b = rng.uniform(2, n + 1)
    nu = n + 1; s = nu / b; r = 1 / s; F = (p - 1) // 2
    U = rng.normal(size=(m, n, p)) + rng.normal(size=(m, n, 1)) * rng.uniform(0, 2)
    W = rng.normal(size=(m, p))
    if trial % 3 == 0:          # make positive-margin by adding IF content
        U0, W0 = if_block(p, n, 1, 1.5); U = np.concatenate([U0, 0.3 * U]); W = np.concatenate([W0, 0.3 * W])
    f = table(U, W); h = margin(f); beta = clean(f)
    Cb = B_p(p, n) ** (1 / s)
    A = np.stack([atoms(U, W, j) for j in range(1, F + 1)], 1)       # (m, F)
    S = neuron_sq(U, W)
    w1 = max(w1, ((np.abs(A) ** r).sum(1) / (Cb * S ** (b / 2))).max())
    kap = np.array([[kappa(p, j, d) for j in range(1, F + 1)] for d in range(1, p)])
    w2 = max(w2, h - (kap @ beta.real).min())
    Q = (np.abs(A) ** r).sum(0); Pj = Q / Cb
    w3 = max(w3, (h - B_p(p, n) * (kap @ Pj ** s).min()) / (B_p(p, n) * (kap @ Pj ** s).min() + 1e-300))
    w4 = max(w4, Pj.sum() / cost(U, W, b))
print(f"(1) max budget ratio {w1:.6f} (<=1);  (2) max h - Lemma2 bound {w2:.3e} (<=0);  (3) max rel AD violation {w3:.3e} (<=0);  sum P_j / P max {w4:.6f} (<=1)")
