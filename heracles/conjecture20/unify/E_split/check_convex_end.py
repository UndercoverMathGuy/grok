"""Sanity checks for R3 (convex-end rigidity: at s = 1 every maximiser is class C, interference-free and has
H >= H^dec(1) = log nu + min H(z*(1))) and for R5 (KKT: cost shares = margin shares under a Fritz-John measure).
Optimiser: torch smoothed-min L-BFGS from unify/split/netopt.py (copied logic, imported read-only)."""
import sys, math, itertools
import numpy as np
from scipy.optimize import linprog
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/grok/heracles/conjecture20/unify/split")
import netopt as no
import elib as E

rng = np.random.default_rng(3)
out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def prog1(p):
    """Gamma_1 and its maximisers: LP max t s.t. K y >= t, sum y = 1 (unique for p = 5, 7)."""
    K = no.kappa_mat(p); F = K.shape[1]
    c = np.zeros(F + 1); c[-1] = -1
    A = np.hstack([-K, np.ones((K.shape[0], 1))]); bvec = np.zeros(K.shape[0])
    res = linprog(c, A_ub=A, b_ub=bvec, A_eq=np.concatenate([np.ones(F), [0]])[None], b_eq=[1], bounds=[(0, None)] * F + [(None, None)], method='highs')
    return -res.fun, res.x[:F]


def analyse(U, W, p, n, s):
    nu = n + 1; b = nu / s
    st = E.stats(U, W, s)
    beta, betaT, atoms = E.own_coeffs(U, W)
    A = max(abs(v) for v in beta.values())
    ifness = max(abs(v) for v in betaT.values()) / A
    e = E.piece_energies(U, W)
    sec = (e.sum(1) - e[:, 1:].max(1)) / e.sum(1)      # fraction of each neuron's energy off its dominant frequency (incl. DC)
    return st, ifness, float((sec * st['w']).sum())


say("== R3: maximisers at s = 1 (b = nu) ==")
for (p, n, widths) in [(5, 2, [6, 8]), (7, 2, [9, 12]), (5, 3, [8, 10])]:
    nu = n + 1; b = float(nu); s = 1.0
    G1, y1 = prog1(p)
    Hdec = math.log(nu) + E.H(y1)
    model = no.Full(p, n)
    best = []
    for m in widths:
        for rep in range(4):
            if rep < 2:
                U0 = rng.standard_normal((m, n, p)); W0 = rng.standard_normal((m, p))
            else:   # IF network on F plus random extra neurons, perturbed
                Ui, Wi = E.if_network(p, n, np.ones(len(E.Fset(p))))
                k = max(0, m - len(Wi))
                U0 = np.concatenate([Ui, 0.3 * rng.standard_normal((k, n, p))])[:m] + 0.2 * rng.standard_normal((m, n, p))
                W0 = np.concatenate([Wi, 0.3 * rng.standard_normal((k, p))])[:m] + 0.2 * rng.standard_normal((m, p))
            U, W, h, P = no.optimise(model, U0, W0, b)
            if h <= 0:
                continue
            st, ifn, sec = analyse(U, W, p, n, s)
            ratio = st['gamma'] / (E.Bp(p, n) * G1)
            best.append((ratio, st['H'], ifn, sec, m))
    best.sort(reverse=True)
    say(f"(p,n)=({p},{n}): {len(best)} runs with h>0; showing near-maximisers (ratio >= 0.9999)")
    for (ratio, Hh, ifn, sec, m) in [x for x in best if x[0] >= 0.9999][:6]:
        say(f"(p,n)=({p},{n}) m={m}: gamma/(B Gamma_1) = {ratio:.6f}; H(theta,1) = {Hh:.4f} vs H^dec(1) = log nu + H(z*) = {Hdec:.4f} "
            f"(diff {Hh-Hdec:+.4f}); max|beta_T|/max|beta| = {ifn:.2e}; off-dominant energy share = {sec:.2e}")

say("== R5: KKT margin-share identity at numerically found maximisers (b in (2, nu)) ==")
for (p, n, b, m) in [(5, 2, 2.5, 6), (7, 2, 2.7, 9), (5, 3, 3.2, 8)]:
    nu = n + 1; s = nu / b
    model = no.Full(p, n)
    bestnet = None
    for rep in range(6):
        if rep < 3:
            U0 = rng.standard_normal((m, n, p)); W0 = rng.standard_normal((m, p))
        else:
            Ui, Wi = E.if_network(p, n, np.ones(len(E.Fset(p))))
            k = max(0, m - len(Wi))
            U0 = np.concatenate([Ui, 0.3 * rng.standard_normal((k, n, p))])[:m] + 0.2 * rng.standard_normal((m, n, p))
            W0 = np.concatenate([Wi, 0.3 * rng.standard_normal((k, p))])[:m] + 0.2 * rng.standard_normal((m, p))
        U, W, h, P = no.optimise(model, U0, W0, b)
        g = h / P ** s
        if bestnet is None or g > bestnet[0]:
            bestnet = (g, U, W)
    g, U, W = bestnet
    st = E.stats(U, W, s)
    # per-neuron gaps on the near-argmin set
    Gtot = E.gaps(U, W)[:, 1:]
    h = Gtot.min()
    mask = Gtot <= h + 1e-4 * abs(h)
    idx = np.argwhere(mask)
    Gi = np.array([E.gaps(U[i:i + 1], W[i:i + 1])[:, 1:][mask] for i in range(len(W))])   # m x |A|
    # LP: min t s.t. -t <= Gi q / h - w <= t, q >= 0, sum q = 1
    na = Gi.shape[1]; mm = len(W)
    c = np.zeros(na + 1); c[-1] = 1
    A1 = np.hstack([Gi / h, -np.ones((mm, 1))]); A2 = np.hstack([-Gi / h, -np.ones((mm, 1))])
    res = linprog(c, A_ub=np.vstack([A1, A2]), b_ub=np.concatenate([st['w'], -st['w']]),
                  A_eq=np.concatenate([np.ones(na), [0]])[None], b_eq=[1], bounds=[(0, None)] * na + [(0, None)], method='highs')
    gIF = no.gamma_IF(p, n, b, no.programme(p, s)[0])
    say(f"(p,n,b)=({p},{n},{b}) width {m}: gamma/gamma_IF = {g/gIF:.6f}; |argmin set| = {na}; "
        f"min_q max_i |E_q g_i / h - w_i| = {res.fun:.2e} (w_i range {st['w'].min():.3f}-{st['w'].max():.3f}); H = {st['H']:.4f}")
open("check_convex_end.out", "w").write("\n".join(out) + "\n")
