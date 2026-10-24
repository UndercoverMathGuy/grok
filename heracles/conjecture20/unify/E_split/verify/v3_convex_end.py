"""Referee check of E3 (convex-end rigidity at s=1, b=nu).
(A) LP for Gamma_1^{(S)}; non-block aligned IF class-C networks satisfying (i)-(iv) attain B_p Gamma_1^{(S)};
(B) Fourier coefficient of g(.,delta) at 2l 1_T in class C equals (1/2) conj(beta_{l,T}) (1 - rho^{l delta}) [checked numerically];
(C) violating any of (i)-(iii) strictly lowers gamma(.;1);
(D) local maximisation at s=1 from random starts: near-maximisers have H >= H^dec(1);
(E) p=3 certified counterexample networks evaluated at s=1 stay below B_3 Gamma_1(3)."""
import itertools, math, json
import numpy as np
from scipy.optimize import linprog, minimize
from fractions import Fraction as Fr
from vlib import *

rng = np.random.default_rng(3)
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def Gamma1(p, S):
    """max min_d sum_{j in S} y_j kappa_j(d), y>=0, sum y <= 1 (LP). Returns value, y."""
    S = list(S); k = len(S)
    # variables y_1..y_k, t ; maximise t
    c = np.zeros(k + 1); c[-1] = -1
    A = []; bb = []
    for d in range(1, p):
        A.append(np.concatenate([-np.array([kappa(j, d, p) for j in S]), [1]])); bb.append(0)
    A.append(np.concatenate([np.ones(k), [0]])); bb.append(1)
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(bb), bounds=[(0, None)] * k + [(None, None)], method='highs')
    return -r.fun, dict(zip(S, r.x[:k]))


def gamma1(U, W, h=None):
    m, n, p = U.shape
    nu = n + 1
    if h is None:
        h = margin(U, W)
    return h / (sqnorm(U, W) ** (nu / 2)).sum()


def aligned_feature(p, n, j, amp, kind, rng):
    """class-C aligned IF feature at frequency j with clean coefficient proportional to amp; several shapes."""
    nu = n + 1
    if kind == 'block':
        U, W = block(p, n, j, 1.0, rng.uniform(0, 1))
    elif kind == 'wide':
        U, W = block(p, n, j, 1.0, rng.uniform(0, 1), m=nu + 2)
    else:  # mixture of two rotated blocks with unequal scales (non-uniform weights)
        U1, W1 = block(p, n, j, 1.0, 0.1); U2, W2 = block(p, n, j, 0.7, 0.37)
        U = np.concatenate([U1, U2]); W = np.concatenate([W1, W2])
    beta, betaT, _ = atoms(U, W)
    lam = (amp / abs(beta[j])) ** (1 / nu)
    return U * lam, W * lam


say("== (A) attainment by non-block aligned IF networks, and H vs H^dec(1) ==")
for (p, n) in [(5, 2), (7, 2), (5, 3), (11, 2), (3, 3)]:
    nu = n + 1
    Fp = F_of(p)
    for S in [tuple(Fp), tuple(Fp[:1]), tuple(Fp[1:]) if len(Fp) > 1 else tuple(Fp)]:
        G, y = Gamma1(p, S)
        for kind in ['block', 'wide', 'mix']:
            Us, Ws = [], []
            for j in S:
                if y[j] > 1e-12:
                    U, W = aligned_feature(p, n, j, y[j], kind, rng); Us.append(U); Ws.append(W)
            U = np.concatenate(Us); W = np.concatenate(Ws)
            g1 = gamma1(U, W)
            _, w = shares(U, W, 1.0)
            yy = np.array([y[j] for j in S if y[j] > 1e-12]); yy = yy / yy.sum()
            Hdec = math.log(nu) + ent(yy)
            say(f"  (p,n)=({p},{n}) S={S} {kind:5s}: gamma(.;1)/(B_p Gamma_1^S) = {g1/(Bp(p,n)*G):.12f};  H - [log nu + H(y)] = {ent(w)-Hdec:+.4f}")

say("== (B) Fourier coefficient of g(.,delta) at 2l 1_T for class-C networks ==")
worst = 0
for (p, n) in [(5, 2), (7, 2), (5, 3)]:
    nu = n + 1
    # random class-C network: neurons each in one plane, no DC, random phases/amplitudes
    m = 7
    a = np.arange(p)
    U = np.zeros((m, n, p)); W = np.zeros((m, p))
    for i in range(m):
        j = rng.choice(F_of(p))
        for t in range(n):
            U[i, t] = rng.uniform(0.5, 1.5) * np.cos(2 * np.pi * j * a / p + rng.uniform(0, 6.3))
        W[i] = rng.uniform(0.5, 1.5) * np.cos(2 * np.pi * j * a / p + rng.uniform(0, 6.3))
    f = table(U, W); G = gaps(f, p, n)
    beta, betaT, _ = atoms(U, W)
    for d in range(1, p):
        g = G[:, d - 1].reshape((p,) * n)
        gh = np.fft.fftn(g) / p ** n     # coefficient of rho^{k.x}: g = sum_k ghat(k) rho^{k.x}  -> fftn uses e^{-}, so index -k
        for l in F_of(p):
            for r in range(1, n + 1):
                for T in itertools.combinations(range(n), r):
                    k = tuple(((2 * l) % p if t in T else 0) for t in range(n))   # fftn index k = coefficient of rho^{k.x}
                    coef = gh[k]
                    pred = 0.5 * np.conj(betaT[(l, T)]) * (1 - np.exp(-2j * np.pi * l * d / p))
                    worst = max(worst, abs(coef - pred))
say(f"  max |ghat(2l 1_T) - (1/2) conj(beta_(l,T)) (1 - rho^(-l d))| = {worst:.2e}  (nonzero factor for d != 0)")

say("== (C) violating (i)/(ii)/(iii) lowers gamma(.;1) strictly ==")
p, n = 7, 2; nu = 3
G, y = Gamma1(p, F_of(p))
Us, Ws = [], []
for j in F_of(p):
    U, W = aligned_feature(p, n, j, y[j], 'block', rng); Us.append(U); Ws.append(W)
U0 = np.concatenate(Us); W0 = np.concatenate(Ws)
g0 = gamma1(U0, W0)
for what in ['DC on one factor', 'leak plane', 'unequal factor norms', 'rotate one atom phase', 'drop one neuron (interference)']:
    U = U0.copy(); W = W0.copy()
    if what == 'DC on one factor':
        U[0, 0] += 0.05
    elif what == 'leak plane':
        U[0, 1] += 0.05 * np.cos(2 * np.pi * 2 * np.arange(p) / p)
    elif what == 'unequal factor norms':
        U[0, 0] *= 1.1; W[0] /= 1.1
    elif what == 'rotate one atom phase':
        W[0] = np.roll(W[0], 0); W[0] = np.real(np.fft.ifft(np.fft.fft(W[0]) * np.exp(-1j * 0.2 * np.sign(np.fft.fftfreq(p)))))
    else:
        U = U[1:]; W = W[1:]
    say(f"  {what:30s}: gamma/gamma_opt = {gamma1(U, W)/g0:.6f}")

say("== (D) local maximisation at s=1 from random starts ((5,2), width 6 and 8) ==")
def softmin(G, tau):
    m0 = G.min(); return m0 - tau * np.log(np.exp(-(G - m0) / tau).sum())
for (p, n, m) in [(5, 2, 6), (5, 2, 8)]:
    nu = n + 1
    G1, ystar = Gamma1(p, F_of(p))
    yy = np.array(list(ystar.values())); Hdec = math.log(nu) + ent(yy / yy.sum())
    for trial in range(4):
        x0 = rng.standard_normal(m * nu * p)
        def unpack(x):
            X = x.reshape(m, nu, p); return X[:, :n, :], X[:, n, :]
        for tau in [1e-1, 1e-2, 1e-3, 1e-4]:
            def obj(x):
                U, W = unpack(x)
                Gm = gaps(table(U, W), p, n)
                return -softmin(Gm, tau * max(1e-9, abs(Gm).max())) / (sqnorm(U, W) ** (nu / 2)).sum()
            x0 = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 3000}).x
            x0 = x0 / np.linalg.norm(x0)
        U, W = unpack(x0)
        g1 = gamma1(U, W)
        _, w = shares(U, W, 1.0)
        say(f"  (p,n,m)=({p},{n},{m}) run {trial}: gamma/(B Gamma_1) = {g1/(Bp(p,n)*G1):.6f}, H - H^dec(1) = {ent(w)-Hdec:+.4f}")

say("== (E) p=3 certified networks at s=1 (b = nu) ==")
import importlib.util, sys
spec = importlib.util.spec_from_file_location("excert", "/home/user/grok/heracles/conjecture20/exact_certificate.py")
excert = importlib.util.module_from_spec(spec); spec.loader.exec_module(excert)
for n, net in excert.NETWORKS.items():
    U = np.array([[float(a) for a in r] for r in net["u"]]); W = np.array([[float(a) for a in r] for r in net["w"]])
    nu = n + 1; p = 3
    # margin via counts (same input factor in every slot)
    h = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; yv = (N1 + 2 * N2) % 3
            z = U[:, 0] ** N0 * U[:, 1] ** N1 * U[:, 2] ** N2
            fv = W.T @ z
            for c in range(3):
                if c != yv:
                    h = fv[yv] - fv[c] if h is None else min(h, fv[yv] - fv[c])
    sq = n * (U ** 2).sum(1) + (W ** 2).sum(1)
    g_at_1 = h / (sq ** (nu / 2)).sum()
    BG = Bp(3, n) * 1.5
    s2 = nu / 2
    gIF2 = nu ** (1 - s2) * BG
    g2 = h / (sq.sum()) ** s2
    say(f"  n={n}: gamma(.;1)/(B_3 Gamma_1) = {g_at_1/BG:.6f} (<=1 required);  at b=2: gamma/gamma_IF = {g2/gIF2:.6f}")
open(__file__.replace('.py', '.out'), 'w').write("\n".join(out) + "\n")
