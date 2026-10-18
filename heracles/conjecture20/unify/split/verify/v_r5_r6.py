"""Referee checks of SPLIT R5 (cost status of mixed modes) and R6 (joint circularity, 2l-pairs, extension of
companion Thm 14), with adversarial optimisation for R6(d).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy python v_r5_r6.py
"""
import itertools, numpy as np
from scipy.optimize import minimize
from vlib import *

rng = np.random.default_rng(5150)
lines = []
def rep(name, val, ok):
    lines.append(('PASS ' if ok else 'FAIL ') + f'{name}   [{val:+.3e}]')

# ---------------- R5
w_q = -np.inf; w_dc = 0; w_slack0 = np.inf
for (p, n, m) in [(5, 2, 4), (7, 2, 3), (5, 3, 3), (3, 2, 3), (11, 2, 2)]:
    nu = n + 1
    for kind in ['gauss', 'sparse', 'heavydc']:
        for trial in range(5):
            U, W = rand_net(rng, m, n, p, kind)
            A = atoms(U, W)
            e = piece_energies(U, W); L = live_mask(U, W)
            for b in [2.0, 2.5, 3.0, float(nu)]:
                s = nu / b; r = 1 / s
                Cb = Bp(p, n) ** (1 / s)
                # atoms of numerically-dead pieces (|uhat| ~ 1e-16 after exact projection) are set to 0
                Lm = live_mask(U, W)
                Q = sum((np.where(Lm[:, j - 1], np.abs(A[j]), 0) ** r).sum() for j in F_(p))
                Plive = sum(P_live_by_freq(U, W, b).values())
                w_q = max(w_q, (Q - Cb * Plive) / (Cb * cost(U, W, b)))   # Q_total <= C_b P_live (normalised)
                if b == 2.0:
                    P = cost(U, W, 2.0)
                    dead = np.where(L, 0, e[:, 1:]).sum()
                    w_dc = max(w_dc, abs(P - Plive - e[:, 0].sum() - dead))
                if b > 2:
                    # a shared live neuron (>=2 nonzero pieces) has strictly positive slack at b>2
                    for i in range(m):
                        if (e[i] > 1e-9).sum() >= 2:
                            q = b / 2
                            w_slack0 = min(w_slack0, (e[i].sum()) ** q - (e[i, 1:] ** q).sum())
rep('R5 sum_l Q_l <= C_b P_live (max of ratio-1, must be <=0)', w_q, w_q <= 1e-12)
rep('R5 b=2: P - P_live = DC energy + dead energy (max abs err)', w_dc, w_dc < 1e-9)
rep('R5 b>2: multi-piece neuron has positive slack (min slack)', w_slack0, w_slack0 > 0)

# ---------------- R6(a,b): class C joint fields proper (output DC allowed, as in the Note's class C)
def pseudo_cov(f, l, k):
    pl = fcoef(field(f, l)); pk = fcoef(field(f, k))
    n = f.ndim - 1; p = f.shape[-1]
    neg = tuple((-np.indices((p,) * n)) % p)
    return (pl * pk[neg]).sum() - pl[(0,) * n] * pk[(0,) * n]

worst = 0
for (p, n) in [(3, 2), (3, 4), (5, 2), (5, 3), (7, 2), (11, 2)]:
    F = F_(p); m = 2 * len(F) + 1
    U = np.zeros((m, n, p)); W = np.zeros((m, p))
    for i in range(m):
        k = F[i % len(F)]
        U[i] = fproj(rng.normal(size=(n, p)), [k], p); W[i] = fproj(rng.normal(size=p), [0, k], p)
    f = table(U, W)
    for l in F:
        for k in F:
            worst = max(worst, abs(pseudo_cov(f, l, k)))
rep('R6b class C (incl. p=3, output DC): all joint pseudo-covariances vanish', worst, worst < 1e-12)

# ---------------- R6(c): 2l-pair criterion for single-output networks
def lab(x, p):
    return min(x % p, (-x) % p)

def pairs(p, l):
    G = [0] + F_(p)
    return {frozenset((a, c)) for a in G for c in G if a != c and
            any((sa * a + sc * c - 2 * l) % p == 0 for sa in (1, -1) for sc in (1, -1))}

def pair_free_sets(p, l):
    G = [0] + F_(p); P2 = pairs(p, l); out = []
    for r_ in range(1, len(G) + 1):
        for S in itertools.combinations(G, r_):
            if l in S and not any(frozenset((a, c)) in P2 for a in S for c in S if a != c):
                out.append(S)
    return out

mism_forward = 0; mism_converse = 0; tot = 0
for (p, n) in [(5, 2), (7, 2), (7, 3), (11, 2), (13, 2)]:
    F = F_(p); G = [0] + F
    for trial in range(60):
        l = int(rng.choice(F))
        # random per-slot frequency sets (union over neurons), each containing l (on-frequency content)
        slotsets = [sorted(set([l] + [int(k) for k in rng.choice(G, size=rng.integers(0, 3), replace=False)])) for _ in range(n)]
        m = 3
        U = np.zeros((m, n, p)); W = np.zeros((m, p))
        for i in range(m):
            for t in range(n):
                U[i, t] = fproj(rng.normal(size=p), slotsets[t], p)
            W[i] = fproj(rng.normal(size=p), [0, l], p)
        pv = abs(pseudo_cov(table(U, W), l, l))
        P2 = pairs(p, l)
        has = any(frozenset((a, c)) in P2 for S in slotsets for a in S for c in S if a != c)
        tot += 1
        if not has and pv > 1e-10:
            mism_forward += 1
        if has and pv < 1e-10:
            mism_converse += 1
rep(f'R6c forward: no 2l-pair => circular  (violations out of {tot})', mism_forward, mism_forward == 0)
rep(f'R6c converse (generic): 2l-pair (+ on-frequency elsewhere) => non-circular (exceptions out of {tot})', mism_converse, True)

# ---------------- R6(d): adversarial check of  min_x cm(Phi_l) <= Re beta_l - k_p sigma  on 2l-pair-free single-output nets,
# and gamma <= nu^{1-s} kappa_min B_p (single-feature IF value) when k_p >= k*(n,s).
def cm(z, p):
    return z.real - np.abs(z.imag) / np.tan(np.pi / p)

def build(z, m, n, p, l, S):
    U = z[:m * n * p].reshape(m, n, p); W = z[m * n * p:].reshape(m, p)
    return fproj(U, S, p), fproj(W, [l], p)

def stats(U, W, p, l):
    f = table(U, W)
    Phi = field(f, l)
    b0, _ = own_from_table(f)
    sig = np.sqrt(0.5 * np.mean(np.abs(Phi - Phi.mean()) ** 2))
    return cm(Phi, p).min(), b0[l].real, sig, f

for (p, n, l, S) in [(5, 2, 1, (0, 1)), (7, 2, 1, (1, 2)), (7, 2, 1, (0, 1)), (11, 2, 1, (0, 1, 4)), (11, 2, 1, (1, 2, 5)), (7, 3, 1, (1, 2))]:
    assert S in pair_free_sets(p, l), (p, l, S)
    kp = np.sqrt(1 / np.tan(np.pi / p) ** 2 - 1)
    m = 4; best = -np.inf; bestg = -np.inf
    nu = n + 1; b = 2.0; s = nu / b
    kmin = kappa(1, 1, p)
    gIF1 = nu ** (1 - s) * kmin * Bp(p, n)
    for rs in range(6):
        z0 = rng.normal(size=m * n * p + m * p)
        def obj(z, tau=0.02):
            U, W = build(z, m, n, p, l, S)
            Phi = field(table(U, W), l); c = cm(Phi, p).ravel()
            sc = np.abs(c).mean() + 1e-12
            smin = -tau * sc * np.log(np.exp(-(c - c.min()) / (tau * sc)).sum()) + c.min()
            mu = Phi.mean(); sig = np.sqrt(0.5 * np.mean(np.abs(Phi - mu) ** 2))
            return -(smin - mu.real + kp * sig) / (np.abs(Phi).mean() + 1e-12)
        res = minimize(obj, z0, method='L-BFGS-B', options=dict(maxiter=300))
        U, W = build(res.x, m, n, p, l, S)
        c, reb, sig, f = stats(U, W, p, l)
        best = max(best, (c - (reb - kp * sig)) / (abs(reb) + 1e-12))
        # gamma maximisation within the same class (margin of single-output net)
        def objg(z, tau=0.01):
            U, W = build(z, m, n, p, l, S)
            G = gaps(table(U, W))[..., 1:].ravel()
            sc = np.abs(G).mean() + 1e-12
            smin = G.min() - tau * sc * np.log(np.exp(-(G - G.min()) / (tau * sc)).sum())
            return -smin / cost(U, W, b) ** s
        a_ = np.arange(p); Ui = np.zeros((m, n, p)); Wi = np.zeros((m, p))
        for q_ in range(m):
            al = np.pi * q_ / nu
            Ui[q_] = np.cos(2 * np.pi * l * a_ / p + al)[None, :].repeat(n, 0); Wi[q_] = np.cos(2 * np.pi * l * a_ / p + n * al)
        if rs % 2 == 0:
            z0 = np.concatenate([Ui.ravel(), Wi.ravel()]) + 0.3 * rng.normal(size=z0.size)
        res = minimize(objg, z0, method='L-BFGS-B', options=dict(maxiter=600))
        U, W = build(res.x, m, n, p, l, S)
        bestg = max(bestg, margin(U, W) / cost(U, W, b) ** s / gIF1)
    rep(f'R6d p={p} n={n} l={l} inputs {S}: max (min cm - (Re b - k_p sig))/|Re b| (<=0)', best, best <= 1e-9)
    rep(f'R6d p={p} n={n} l={l} inputs {S}: max gamma / single-feature IF value (<=1)', bestg, bestg <= 1 + 1e-9)

for x in lines:
    print(x)
