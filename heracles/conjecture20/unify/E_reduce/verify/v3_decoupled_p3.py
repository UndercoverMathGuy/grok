# R2(b) decoupled networks; cone lemma per delta; and p = 3 certificates against every value statement.
import numpy as np, sys, json
from fractions import Fraction as Fr
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_reduce/verify')
from vlib import *
rng = np.random.default_rng(11)

def cm(z, p): return np.real(z) - np.abs(np.imag(z)) / np.tan(np.pi / p)

def if_block(p, n, j, amp=1.0):
    nu = n + 1; a = np.arange(p); U = []; W = []
    for l in range(nu):
        al = np.pi * l / nu
        U.append([amp * np.cos(2 * np.pi * j * a / p + al) for t in range(n)]); W.append(amp * np.cos(2 * np.pi * j * a / p + n * al))
    return np.array(U), np.array(W)

worst_cone = -np.inf; worst_psi = np.inf; worst_H = 0; worst_off = 0
for trial in range(30):
    p = [5, 7, 11][trial % 3]; n = 2 if p == 11 else [2, 3][trial % 2]; nu = n + 1; F = (p - 1) // 2
    U1, W1 = if_block(p, n, 1)
    # perturb inside the single-feature class: inputs arbitrary, outputs in span{1,cos1,sin1}
    Pout = proj(p, 0) + proj(p, 1)
    U1 = U1 + 0.08 * rng.normal(size=U1.shape); W1 = (W1 + 0.08 * rng.normal(size=W1.shape)) @ Pout
    Ue, We = rng.normal(size=(1, n, p)) * 0.2, (rng.normal(size=(1, p)) * 0.2) @ Pout
    U1 = np.concatenate([U1, Ue]); W1 = np.concatenate([W1, We])
    f1 = table(U1, W1); ph = Phi(f1, p, n, 1); c = cm(ph, p).min()
    if c <= 0: continue
    # cone lemma per delta: g_1(x,delta) >= kappa_1(delta) cm(Phi(x))
    g1 = gaps(f1, p, n); kap = 1 - np.cos(2 * np.pi * np.arange(1, p) / p)
    worst_cone = max(worst_cone, (kap[None, :] * cm(ph, p)[:, None] - g1).max())
    # decoupled union
    S = sorted(rng.choice(np.arange(1, F + 1), size=rng.integers(1, F + 1), replace=False).tolist())
    A = np.zeros(F); A[[j - 1 for j in S]] = rng.uniform(0.2, 2, len(S))
    lam = rng.uniform(0.6, 1.6, F)
    Us = []; Ws = []
    idx = np.arange(p)
    for l in S:
        sc = A[l - 1] ** (1 / nu)
        Us.append(sc * U1[:, :, (l * idx) % p]); Ws.append(sc * W1[:, (l * idx) % p])
    U = np.concatenate(Us); W = np.concatenate(Ws); f = table(U, W)
    be = beta(f, p, n); worst_off = max(worst_off, max([abs(be[j - 1]) for j in range(1, F + 1) if j not in S] + [0]))
    hl = sum(lam[l - 1] * Gl(f, p, n, l) for l in range(1, F + 1)).min()
    mA = m_lam(A, p, lam)
    a1 = neuron_sq(U1, W1) ** (nu / 2)
    for sig in [1.0, 1.2, nu / 2 * 0.9, nu / 2]:
        b = nu / sig
        psi = c * mA / (np.sum(np.outer(A[[j - 1 for j in S]], a1) ** (1 / sig))) ** sig
        worst_psi = min(worst_psi, hl / cost(U, W, b) ** sig / psi - 1)
        z = A[[j - 1 for j in S]] ** (1 / sig); z /= z.sum()
        worst_H = max(worst_H, abs(H(shares(U, W, b)) - H(z) - H(shares(U1, W1, b))))
print('R2(b): cone lemma per delta, max [kappa cm - g] = %.2e (<= 0); min gamma_lam(Theta)/psi - 1 = %.2e (>= 0); '
      'max |H(Theta)-H(z)-H(theta1)| = %.2e; max |beta_j| off spt A = %.2e' % (worst_cone, worst_psi, worst_H, worst_off))

# p = 3 exact certificates (n = 11, 13): every network is single-feature; W = D = kappa_min B_3 pi.
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import importlib.util
spec = importlib.util.spec_from_file_location('ec', '/home/user/grok/heracles/conjecture20/exact_certificate.py'); ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
for n, net in ec.NETWORKS.items():
    p = 3; nu = n + 1; s = nu / 2
    U = np.array([[float(a) for a in r] for r in net['u']]); W = np.array([[float(a) for a in r] for r in net['w']])
    m = len(U)
    # slot-symmetric: work with count vectors
    best_h = np.inf; best_c = np.inf
    rho = np.exp(2j * np.pi / 3)
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; y = (N1 + 2 * N2) % 3
            z = U[:, 0] ** N0 * U[:, 1] ** N1 * U[:, 2] ** N2
            fx = W.T @ z
            best_h = min(best_h, min(fx[y] - fx[c] for c in range(3) if c != y))
            ph = (2 / 3) * rho ** y * sum(fx[c] * rho ** (-c) for c in range(3))
            best_c = min(best_c, cm(ph, 3))
    P = np.sum((n * (U ** 2).sum(1) + (W ** 2).sum(1)))  # b = 2
    gam = best_h / P ** s
    B3 = Bp(3, n); gIF = nu ** (1 - s) * B3 * 1.5
    # balance / output DC / nu E_1
    un = np.linalg.norm(U, axis=1); wn = np.linalg.norm(W, axis=1)
    sq = n * un ** 2 + wn ** 2; w = sq / sq.sum()
    nuE = nu * np.sum(w * np.linalg.norm(W @ proj(3, 1), axis=1) ** 2 / sq)
    print('p=3 n=%d: gamma/gamma_IF=%.6f; h - kappa_min*min cm(Phi) = %.2e; gamma/(B_3 Gamma_s)=%.4f (<=1); '
          'balance max| ||u||-||w|| |=%.2e; nu E_1=%.6f; H=%.5f, log m=%.5f, log nu=%.5f' % (
          n, gam / gIF, best_h - 1.5 * best_c, gam / (B3 * 1.5), np.abs(un - wn).max(), nuE, H(w), np.log(m), np.log(nu)))
