"""hr2: independent checks of the twisted Rotfel'd payment R7 (chains, disk bound, Rotfel'd, (P,q)-monotonicity,
closed form for S_min, disk-to-circle reduction) and of the imported payments (Note Thm c1sharp, Part I Thm A)."""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(777)
out = []


def rep(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


def atoms(K, mode):
    if mode == 'rand':
        rr = rng.random(K) ** 2; p = np.exp(1j * rng.uniform(-np.pi, np.pi, K))
        xi = np.exp(1j * rng.uniform(-np.pi, np.pi, K)); eta = np.exp(1j * rng.uniform(-np.pi, np.pi, K))
    elif mode == 'cluster':     # nearly aligned atoms with nearly equal sign phases (near single atom)
        rr = rng.random(K); p = np.exp(1j * rng.normal(0, 0.3, K))
        xi = np.exp(1j * rng.normal(0, 0.4, K)); eta = np.exp(1j * rng.normal(0, 0.4, K))
    else:                       # IF-like: 3-ish atoms with phases cancelling sign flips
        rr = rng.random(K) + 0.5; p = np.exp(1j * rng.normal(0, 0.2, K))
        xi = np.exp(2j * np.pi * rng.integers(0, 3, K) / 3 + 1j * rng.normal(0, 0.2, K))
        eta = np.exp(2j * np.pi * rng.integers(0, 3, K) / 3 + 1j * rng.normal(0, 0.2, K))
    beta = rr * p
    return beta, xi, eta


def data(beta, xi, eta):
    A = beta.sum(); B = (beta * xi).sum(); C = (beta * eta).sum(); D = (beta * xi * eta).sum()
    return A, B, C, D


def P1(beta, s):
    return 3 ** (1 - s) * np.sum(np.abs(beta) ** (1 / s)) ** s


def chains(beta, xi, eta):
    rr = np.abs(beta); p = beta / np.where(rr > 0, rr, 1)
    us = [np.stack([np.ones_like(p), p * xi, p ** 2 * xi * eta], 1),
          np.stack([np.ones_like(p), p * xi, p * xi * eta], 1),
          np.stack([np.ones_like(p), p * eta, p * xi * eta], 1)]
    return [np.einsum('i,ia,ib->ab', rr, u, u.conj()) for u in us]


def S_of(P, q, r):
    roots = np.roots([1, 0, -P, -q]).real
    return np.sum(np.maximum(1 + roots, 0) ** r)


def lowK(P):
    return max(P - 1, -2 * (P / 3) ** 1.5)


def upK(P):
    return 2 * (P / 3) ** 1.5


def Smin_closed(Pbar, qlow, r):
    Pbar = min(Pbar, 3.0)
    if qlow > upK(Pbar) + 1e-15:
        return np.inf
    return S_of(Pbar, max(qlow, lowK(Pbar)), r)


def Smin_brute(Pbar, qlow, r, n=400):
    best = np.inf
    for P in np.linspace(0, min(Pbar, 3.0), n):
        lo = max(qlow, lowK(P)); hi = upK(P)
        if lo > hi:
            continue
        for q in np.linspace(lo, hi, 25):
            best = min(best, S_of(P, q, r))
    return best


# ---------------------------------------------------------------- (a) chains: entries, disk, Rotfel'd
worst_ent = 0; worst_disk = -np.inf; worst_rot = -np.inf
for trial in range(3000):
    K = rng.integers(1, 7); mode = ['rand', 'cluster', 'if'][trial % 3]
    beta, xi, eta = atoms(K, mode)
    A, B, C, D = data(beta, xi, eta)
    L = np.abs(beta).sum(); al = A / abs(A); rhoL = np.sqrt(max(2 * L * (L - abs(A)), 0))
    Hs = chains(beta, xi, eta)
    exact = [(Hs[0][1, 0], B), (Hs[0][2, 1], C), (Hs[1][1, 0], B), (Hs[1][2, 0], D), (Hs[2][1, 0], C), (Hs[2][2, 0], D)]
    worst_ent = max(worst_ent, max(abs(x - y) for x, y in exact), np.abs(np.diag(Hs[0]) - L).max())
    unc = [(Hs[0][2, 0], al * D), (Hs[1][2, 1], np.conj(al) * C), (Hs[2][2, 1], np.conj(al) * B)]
    worst_disk = max(worst_disk, max(abs(x - y) for x, y in unc) - rhoL)
    for s in (1.0, 1.2, 1.5):
        r = 1 / s
        for H in Hs:
            nu = np.clip(np.linalg.eigvalsh(H / L), 0, None)
            lb = L * (np.mean(nu ** r)) ** s
            worst_rot = max(worst_rot, (lb - P1(beta, s)) / P1(beta, s))
rep(f"[R7 chains] entry err {worst_ent:.1e}; max[|X'-alpha^{{+-1}}X| - rho_L] {worst_disk:.2e} (<=0); max rel[L M_r(nu) - P1] {worst_rot:.2e} (<=0)")

# ---------------------------------------------------------------- (b) monotonicity of S_r in (P,q) on K
viol = 0; tot = 0
for trial in range(4000):
    r = rng.choice([2 / 3, 0.75, 0.85, 0.95])
    P = rng.uniform(0, 3); lo, hi = lowK(P), upK(P)
    if lo >= hi:
        continue
    q = rng.uniform(lo, hi)
    dq = 1e-4
    if q + dq <= hi:
        tot += 1; viol += S_of(P, q + dq, r) < S_of(P, q, r) - 1e-12
    dP = 1e-4
    if P + dP <= 3 and lowK(P + dP) <= q <= upK(P + dP):
        tot += 1; viol += S_of(P + dP, q, r) > S_of(P, q, r) + 1e-12
rep(f"[R7 monotone] violations {viol}/{tot}")

# ---------------------------------------------------------------- (c) closed form vs brute force
worst_c = 0; worst_c_neg = 0
for trial in range(250):
    r = rng.choice([2 / 3, 0.8, 1.0])
    Pbar = rng.uniform(0, 3.5); qlow = rng.uniform(-1.2, 2.2)
    sc = Smin_closed(Pbar, qlow, r); sb = Smin_brute(Pbar, qlow, r)
    if np.isinf(sc) or np.isinf(sb):
        if np.isinf(sc) != np.isinf(sb):
            worst_c = max(worst_c, 99)
        continue
    worst_c = max(worst_c, sc - sb)            # closed form must be <= every feasible value (it is the min)
    worst_c_neg = min(worst_c_neg, sc - sb)    # brute force grid can only be >= true min
rep(f"[R7 (c)] max[closed - brute] {worst_c:.2e} (<= ~0: closed is a valid lower bound); min {worst_c_neg:.2e} (grid coarseness)")


# ---------------------------------------------------------------- (d) disk-to-circle, and full circle payment <= P1
def circle_payment(Am, mB, mC, mD, phipay, s, nL=160, npsi=181):
    """min over L >= Lmin, psi of L (Smin/3)^s, for each chain, then max over chains (grid; >= true min)"""
    r = 1 / s
    Lmin = max(Am, mB, mC, mD)
    best = -np.inf
    for (mX, mY, mZ) in [(mD, mB, mC), (mC, mB, mD), (mB, mC, mD)]:
        cur = 3 ** (1 - s) * 2.5 * Lmin            # tail
        for t in np.concatenate([np.linspace(0, 0.05, 40), np.linspace(0.05, 1.5, nL)]):
            L = Lmin * (1 + t); rh = np.sqrt(max(2 * L * (L - Am), 0))
            for psi in np.linspace(-np.pi, np.pi, npsi):
                Pb = (mY ** 2 + mZ ** 2 + mX ** 2 + rh ** 2 + 2 * mX * rh * np.cos(psi - phipay)) / L ** 2
                q = 2 * mY * mZ * (mX * np.cos(phipay) + rh * np.cos(psi)) / L ** 3
                S = Smin_closed(Pb, q, r)
                if np.isfinite(S):
                    cur = min(cur, L * (S / 3) ** s)
        best = max(best, cur)
    return best


worst_d = -np.inf; rows = []
for trial in range(120):
    K = rng.integers(1, 6); mode = ['rand', 'cluster', 'if'][trial % 3]
    beta, xi, eta = atoms(K, mode)
    A, B, C, D = data(beta, xi, eta)
    phipay = np.angle(A * D * np.conj(B * C)) if min(abs(B), abs(C), abs(D)) > 0 else 0.0
    for s in (1.0, 1.25, 1.5):
        lb = circle_payment(abs(A), abs(B), abs(C), abs(D), phipay, s, nL=60, npsi=91)
        worst_d = max(worst_d, (lb - P1(beta, s)) / P1(beta, s))
rep(f"[R7 (d) full circle payment] max rel [payment - P1] = {worst_d:.3e} (<=0 up to grid)")

# direct disk-to-circle step: interior X' moved along d to circle never increases S
worst_dc = -np.inf
for trial in range(3000):
    r = rng.choice([2 / 3, 0.8, 1.0])
    L = 1.0; c0 = rng.normal() + 1j * rng.normal(); rad = rng.uniform(0, 0.8)
    Y = (rng.normal() + 1j * rng.normal()) * 0.4; Z = (rng.normal() + 1j * rng.normal()) * 0.4
    Xp = c0 + rad * np.sqrt(rng.random()) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    P = (abs(Xp) ** 2 + abs(Y) ** 2 + abs(Z) ** 2); q = 2 * np.real(Y * Z * np.conj(Xp))
    if not (P <= 3 and lowK(P) <= q <= upK(P)):
        continue
    S_true = S_of(P, q, r)
    # min over the circle of Smin(Pbar, q) must be <= S_true
    thetas = np.linspace(0, 2 * np.pi, 721)
    Xc = c0 + rad * np.exp(1j * thetas)
    Pc = np.abs(Xc) ** 2 + abs(Y) ** 2 + abs(Z) ** 2; qc = 2 * np.real(Y * Z * np.conj(Xc))
    mn = min(Smin_closed(a, b, r) for a, b in zip(Pc, qc))
    worst_dc = max(worst_dc, mn - S_true)
rep(f"[R7 (d) disk->circle] max [min_circle Smin - S(interior point)] = {worst_dc:.2e} (<=0)")


# ---------------------------------------------------------------- imported payments: c1sharp and Thm A (adversarial)
def neg_gap(v, s, which, K):
    rr = np.abs(v[:K]); ph = v[K:2 * K]; a = v[2 * K:3 * K]; b = v[3 * K:4 * K]
    beta = rr * np.exp(1j * ph); xi = np.exp(1j * a); eta = np.exp(1j * b)
    A, B, C, D = data(beta, xi, eta)
    tot = np.abs(beta).sum() + 1e-300
    if which == 'c1':
        g = abs(A) - P1(beta, s) - (1 - 3 ** (1 - s)) * max(abs(B), abs(C), abs(D))
    else:
        g = abs(A) - P1(beta, s) - np.sqrt(abs(B) ** 2 + abs(C) ** 2 + abs(D) ** 2) / (2 * np.sqrt(3))
    return -g / tot


for which in ('c1', 'A'):
    worst = -np.inf
    for s in (1.0, 1.1, 1.25, 1.4, 1.5):
        for trial in range(60):
            K = rng.integers(1, 6)
            v0 = np.concatenate([rng.random(K), rng.normal(0, 0.5 if trial % 2 else 3, 3 * K)])
            res = minimize(neg_gap, v0, args=(s, which, K), method='Nelder-Mead', options=dict(maxiter=6000, fatol=1e-13, xatol=1e-10))
            worst = max(worst, -res.fun)
    rep(f"[imported {'c1sharp' if which=='c1' else 'Thm A'}] max normalised violation over adversarial search (s in [1,1.5]) = {worst:.3e} (<=0)")

open('hr2_payment.out', 'w').write("\n".join(out) + "\n")
