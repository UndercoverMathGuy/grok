# Network-level spectral bound (Theorem W2), every network:
#   log gamma(theta;s) <= log gamma_IF^{(S)}(s) + (s-1)[log nu - Htil(theta,s)] - Delta'(theta,s)
#   Htil = (1/(1-r)) log sum_j q_j exp((1-r) H_j),  q_j = |A_j|^r / sum |A|^r,  H_j = max(H_r(mu_j), H_r(pi_j))
#   Delta' = log( Gamma^{(S)}_s (sum_j |A_j|^r)^s / h ) >= 0  (margin deficit vs the programme bound at own amplitudes)
# Checked on random networks, locally optimised networks, and the exact p=3 counterexample certificates.
import sys, numpy as np, itertools
from fractions import Fraction as Fr
from elib import *
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/global')
import lib as glib

rng = np.random.default_rng(2)


def feature_entropy(beta, xi, s):
    r = 1 / s
    lam = spec_chain(beta, xi); lam = np.where(np.abs(lam) < 1e-10 * np.abs(lam).max(), 0, lam)
    pis = spec_twisted(beta, xi); pis = np.where(pis < 1e-10 * pis.max(), 0, pis)
    return max(renyi(np.abs(lam), r), renyi(pis, r))


def decompose(h, mass_b_cost, A, beta, xi, p, n, s, S):
    """returns (log gamma, log gamma_IF^S, (s-1)(log nu - Htil), Delta', Xi)"""
    nu = n + 1; r = 1 / s
    feats = [j for j in S]
    Aabs = np.array([abs(A[j - 1]) for j in feats])
    Hs = np.array([feature_entropy(beta[:, j - 1], xi[:, :, j - 1], s) for j in feats])
    Sh = np.sum(Aabs ** r); q = Aabs ** r / Sh
    Htil = np.log(np.sum(q * np.exp((1 - r) * Hs))) / (1 - r) if r < 1 else np.sum(q * Hs)
    G, _ = programme_vertices(p, s, feats)
    lg = np.log(h) - s * np.log(mass_b_cost)
    lgif = np.log(nu ** (1 - s) * Bp(p, n) * G)
    ent = (s - 1) * (np.log(nu) - Htil)
    Dp = np.log(G * Sh ** s / h)
    Xi = lgif + ent - Dp - lg
    return lg, lgif, ent, Dp, Xi


worst = np.inf; rows = 0
# (1) random networks with positive margin are rare; use IF networks plus perturbations (and mixing) to get h > 0
for (p, n) in [(5, 2), (7, 2), (5, 3), (11, 2), (7, 3)]:
    F = (p - 1) // 2
    for trial in range(25):
        s = rng.uniform(1.0, (n + 1) / 2)
        G, ys = programme_vertices(p, s)
        U, W = if_network(p, n, ys[0])
        eps = [0.0, 0.02, 0.1, 0.3][trial % 4]
        m = U.shape[0]
        U2 = np.concatenate([U + eps * rng.normal(size=U.shape), eps * rng.normal(size=(3, n, p))])
        W2 = np.concatenate([W + eps * rng.normal(size=W.shape), eps * rng.normal(size=(3, p))])
        net = Net(U2, W2); h = net.margin()
        if h <= 0: continue
        A = net.clean(); beta, xi = net.atoms()
        S = [j for j in range(1, F + 1) if abs(A[j - 1]) > 1e-12]
        out = decompose(h, net.cost((n + 1) / s), A, beta, xi, p, n, s, S)
        worst = min(worst, out[4]); rows += 1
print(f"perturbed IF networks: {rows} checked, min slack Xi = {worst:.3e} (must be >= -1e-10)")

# (2) locally optimised networks (global/lib.py optimiser), b = 2 and b = 2.5
for (p, n, b, m, seed) in [(5, 2, 2.0, 6, 0), (5, 2, 2.0, 4, 1), (7, 2, 2.5, 9, 2), (5, 3, 2.0, 8, 3), (7, 2, 2.0, 5, 4)]:
    r0 = np.random.default_rng(seed)
    U0 = r0.normal(size=(m, n, p)) * 0.5; W0 = r0.normal(size=(m, p)) * 0.5
    gnet = glib.optimise(U0, W0, b, taus=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4), iters=1500)
    net = Net(gnet.U, gnet.W); s = (n + 1) / b; h = net.margin()
    if h <= 0:
        print(f"opt ({p},{n},b={b},m={m}): margin {h:.2e} <= 0, skipped"); continue
    F = (p - 1) // 2; A = net.clean(); beta, xi = net.atoms()
    S = [j for j in range(1, F + 1) if abs(A[j - 1]) > 1e-9 * np.abs(A).max()]
    lg, lgif, ent, Dp, Xi = decompose(h, net.cost(b), A, beta, xi, p, n, s, S)
    GIF = (n + 1) ** (1 - s) * Bp(p, n) * programme_vertices(p, s)[0]
    print(f"opt ({p},{n},b={b},m={m}): gamma/gamma_IF={np.exp(lg) / GIF:.5f} features={S} | entropy term={ent:.4f} Delta'={Dp:.4f} slack Xi={Xi:.4f}")

# (3) exact p = 3 certificates (slot-symmetric), n = 11 and 13, b = 2
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import exact_certificate as ec
for n in [11, 13]:
    net = ec.NETWORKS[n]; p = 3; nu = n + 1; s = nu / 2; r = 1 / s
    U = np.array([[float(a) for a in row] for row in net['u']]); W = np.array([[float(a) for a in row] for row in net['w']])
    # exact margin via counts (Fractions)
    Uf = [[Fr(a) for a in row] for row in net['u']]; Wf = [[Fr(a) for a in row] for row in net['w']]
    h = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; y = (N1 + 2 * N2) % 3
            z = [Uf[i][0] ** N0 * Uf[i][1] ** N1 * Uf[i][2] ** N2 for i in range(len(Uf))]
            f = [sum(Wf[i][c] * z[i] for i in range(len(Uf))) for c in range(3)]
            for c in range(3):
                if c != y and (h is None or f[y] - f[c] < h): h = f[y] - f[c]
    h = float(h)
    cost = np.sum((n * (U ** 2).sum(1) + (W ** 2).sum(1)) ** (2 / 2))
    Uh = np.fft.fft(U, axis=1); Wh = np.fft.fft(W, axis=1)
    beta = 2 * p ** (-nu) * np.conj(Wh[:, 1]) * Uh[:, 1] ** n
    xi = np.repeat((np.conj(Uh[:, 1]) / Uh[:, 1])[:, None], n, axis=1)
    A = np.array([beta.sum()])
    lg, lgif, ent, Dp, Xi = decompose(h, cost, A, beta[:, None], xi[:, :, None], p, n, s, [1])
    G = np.exp(lg - lgif)
    print(f"p=3 certificate n={n}: gamma/gamma_IF = {G:.6f}; Htil = {np.log(nu) - ent / (s - 1):.4f} (log nu = {np.log(nu):.4f}, log m = {np.log(len(U)):.4f});"
          f" winner width e^Htil = {np.exp(np.log(nu) - ent / (s - 1)):.3f} <= nu G^(-1/(s-1)) e^(-Delta'/(s-1)) = {nu * G ** (-1 / (s - 1)) * np.exp(-Dp / (s - 1)):.3f}; Delta'={Dp:.4f}; Xi={Xi:.4f}")
