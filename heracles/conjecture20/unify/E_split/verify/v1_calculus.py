"""Referee check of E1 (free-energy calculus) and E9 (split can lower entropy). Independent code."""
import math, itertools
import numpy as np
from scipy.integrate import quad
from vlib import *

rng = np.random.default_rng(12345)
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def generic_net(p, n, extra=3):
    """IF blocks on all of F (positive margin) plus noise and random multi-frequency/DC neurons."""
    Us, Ws = [], []
    for j in F_of(p):
        U, W = block(p, n, j, 1.0, rng.uniform(0, 1))
        Us.append(U); Ws.append(W)
    U = np.concatenate(Us); W = np.concatenate(Ws)
    U = U + 0.03 * rng.standard_normal(U.shape); W = W + 0.03 * rng.standard_normal(W.shape)
    U2 = 0.25 * rng.standard_normal((extra, n, p)) + 0.1; W2 = 0.25 * rng.standard_normal((extra, p))
    return np.concatenate([U, U2]), np.concatenate([W, W2])


def Var_logw(U, W, s):
    c, w = shares(U, W, s)
    lw = np.log(w)
    return float((w * lw ** 2).sum() - ((w * lw).sum()) ** 2)


cases = [(5, 2), (7, 2), (5, 3), (3, 2), (3, 4)]
say("== E1(a),(b): F formulas, tangent intercept, dF/ds, integral identity ==")
worst = 0
for (p, n) in cases:
    nu = n + 1
    for trial in range(3):
        U, W = generic_net(p, n)
        h = margin(U, W)
        if h <= 0:
            say("  skip h<=0", p, n); continue
        for s in [1.0, 1.17, 1.5, nu / 2]:
            b = nu / s
            lg, H, F = logF(U, W, s, h)
            c, w = shares(U, W, s)
            Fa = math.log(h) - math.log(c.sum()) - (s - 1) * float((w * np.log(c)).sum())
            Fb = math.log(h) - math.log(c.sum()) - float((w * np.log(np.sqrt(sqnorm(U, W)) ** (nu - b))).sum())
            # analytic tangent intercept: phi(s) + (1-s) phi'(s), phi' = -H (independent FD check)
            e = 1e-6
            dphi = (logF(U, W, s + e, h)[0] - logF(U, W, s - e, h)[0]) / (2 * e)
            Fc = lg + (1 - s) * dphi
            # dF/ds vs (s-1)Var/s
            dF = (logF(U, W, s + e, h)[2] - logF(U, W, s - e, h)[2]) / (2 * e)
            dFp = (s - 1) * Var_logw(U, W, s) / s
            I, _ = quad(lambda x: (x - 1) * Var_logw(U, W, x) / x, 1, s, epsabs=1e-12, epsrel=1e-12)
            Fd = logF(U, W, 1.0, h)[0] + I
            worst = max(worst, abs(Fa - F), abs(Fb - F), abs(Fc - F), abs(dF - dFp), abs(Fd - F))
            assert F >= logF(U, W, 1.0, h)[0] - 1e-12
say(f"  max discrepancy (a),(b) over {len(cases)} (p,n) cases incl. p=3: {worst:.2e}")

say("== E1(c): replication with arbitrary cost fractions t (incl. s=1 equality case) ==")
worst = 0; s1note = []
for (p, n) in [(5, 2), (5, 3), (3, 2)]:
    nu = n + 1
    U, W = generic_net(p, n)
    h = margin(U, W)
    for s in [1.0, 1.3, nu / 2]:
        b = nu / s
        lg, H, F = logF(U, W, s, h)
        for k in [2, 3, 5]:
            t = rng.dirichlet(np.ones(k) * 0.5)
            Ur = np.concatenate([U * tm ** (1 / b) for tm in t]); Wr = np.concatenate([W * tm ** (1 / b) for tm in t])
            hr = margin(Ur, Wr)
            lg2, H2, F2 = logF(Ur, Wr, s, hr)
            c0 = shares(U, W, s)[0].sum(); c1 = shares(Ur, Wr, s)[0].sum()
            pred_dlg = -(s - 1) * renyi(t, s)
            worst = max(worst, abs(c1 - c0) / c0, abs(lg2 - lg - pred_dlg), abs(H2 - H - ent(t)),
                        abs(F2 - F - (s - 1) * (ent(t) - renyi(t, s))))
            if s == 1.0:
                s1note.append(F2 - F)
say(f"  max error: {worst:.2e};  at s=1 dF for NON-uniform t = {max(abs(x) for x in s1note):.1e} (equality without uniformity: 'iff' fails at s=1)")
# small-rate limit
for s in [1.2, 2.0]:
    for tau in [1e-2, 1e-4, 1e-8, 1e-16]:
        t = np.array([1 - tau, tau])
        say(f"  s={s} tau={tau:.0e}: rate (s-1)H_s/H = {(s-1)*renyi(t,s)/ent(t):.4f}  (s/log(1/tau) = {s/math.log(1/tau):.4f})")

say("== E1(d): merging; isolated pair formula; proportional tables via even sign flips ==")
worst = 0; strict_ok = True
for (p, n) in [(5, 2), (5, 3)]:
    nu = n + 1
    for trial in range(20):
        s = rng.uniform(1.01, nu / 2); b = nu / s
        om_U = rng.standard_normal((1, n, p)); om_W = rng.standard_normal((1, p))
        lam = rng.uniform(0.05, 2)
        # neuron 2 = lam * omega with an even number of factor sign flips -> table proportional (ratio lam^nu > 0)
        sg = np.ones(nu); idx = rng.choice(nu, 2, replace=False); sg[idx] = -1
        U2 = lam * om_U * sg[:n, None]; W2 = lam * om_W * sg[n]
        pairU = np.concatenate([om_U, U2]); pairW = np.concatenate([om_W, W2])
        mu = lam ** nu
        mergeU = (1 + mu) ** (1 / nu) * om_U; mergeW = (1 + mu) ** (1 / nu) * om_W
        assert np.allclose(table(pairU, pairW), table(mergeU, mergeW))
        cp = shares(pairU, pairW, s)[0].sum(); cm = shares(mergeU, mergeW, s)[0].sum()
        strict_ok &= cm < cp
        # isolated pair formula with t = cost fractions (h formal: use any positive constant, F diff independent of h)
        t = shares(pairU, pairW, s)[1]
        Fp = math.log(1.0) - s * math.log(cp) + (s - 1) * ent(t)
        Fm = math.log(1.0) - s * math.log(cm)
        worst = max(worst, abs((Fp - Fm) - (s - 1) * (ent(t) - renyi(t, s))))
say(f"  merge strictly cheaper (s>1): {strict_ok}; isolated-pair F formula max err {worst:.2e}")

say("== E1(e): Sigma elements (global and per-neuron, incl. DC flips) preserve every c_i ==")
worst = 0
for (p, n) in [(7, 2), (5, 3)]:
    U, W = generic_net(p, n)
    for trial in range(10):
        U2 = U.copy(); W2 = W.copy()
        for i in range(len(W)):
            S = rng.choice([-1, 1], size=(len(F_of(p)), n))
            for k in F_of(p):
                for t in range(n):
                    if S[k - 1, t] < 0:
                        U2[i, t] -= 2 * proj(U[i, t], k, p)
                if np.prod(S[k - 1]) < 0:
                    W2[i] -= 2 * proj(W[i], k, p)
        worst = max(worst, np.abs(sqnorm(U2, W2) - sqnorm(U, W)).max())
        b1, bT1, _ = atoms(U, W); b2, bT2, _ = atoms(U2, W2)
        worst = max(worst, max(abs(b1[j] - b2[j]) for j in b1), max(abs(bT1[k] - bT2[k]) for k in bT1))
say(f"  max |c change| and |beta, beta_T change| under random per-neuron flips: {worst:.2e}")

say("== E1(f): split cost / entropy chain rule; b=2 no-DC monotonicity; block split = block-group average ==")
worst = 0
for (p, n) in [(7, 2), (5, 3), (11, 2)]:
    nu = n + 1
    for trial in range(4):
        m = 4
        U = rng.standard_normal((m, n, p)); W = rng.standard_normal((m, p))
        if trial % 2 == 0:  # remove DC
            U = U - U.mean(-1, keepdims=True); W = W - W.mean(-1, keepdims=True)
        e = plane_energy(U, W); E = e.sum(1); rho = e / E[:, None]
        for b in [2.0, 2.5, nu]:
            s = nu / b
            c, w = shares(U, W, s)
            lam = (rho[:, 1:] ** (b / 2)).sum(1)
            assert np.all(lam <= (1 - rho[:, 0]) ** (b / 2) + 1e-12)
            pi = (rho[:, 1:] ** (b / 2)) / lam[:, None]
            what = w * lam / (w * lam).sum()
            # explicit split network
            SU = np.array([[proj(U[i, t], k, p) for t in range(n)] for i in range(m) for k in F_of(p)])
            SW = np.array([proj(W[i], k, p) for i in range(m) for k in F_of(p)])
            cs, ws = shares(SU, SW, s)
            worst = max(worst, abs(cs.sum() - c.sum() * (w * lam).sum()) / c.sum(),
                        abs(ent(ws) - (ent(what) + sum(what[i] * ent(pi[i]) for i in range(m)))))
            if b == 2.0 and trial % 2 == 0:
                assert ent(ws) >= ent(w) - 1e-12
                worst = max(worst, abs(ent(ws) - ent(w) - sum(w[i] * ent(rho[i, 1:]) for i in range(m))))
say(f"  split formula max err: {worst:.2e}")

# block split vs group average (explicit enumeration of the block sign group)
worst = 0; costok = True
for (p, n, blocks) in [(7, 2, [[0, 2], [1], [3]]), (5, 3, [[0], [1, 2]]), (7, 2, [[0, 1, 2, 3]])]:
    U = rng.standard_normal((2, n, p)); W = rng.standard_normal((2, p))
    nb = len(blocks)
    def PB(v, B):
        return sum(proj(v, k, p) for k in B)
    acc = 0; cnt = 0
    for bits in itertools.product([-1, 1], repeat=nb * n):
        S = np.array(bits).reshape(nb, n)
        U2 = U.copy(); W2 = W.copy()
        for bi, B in enumerate(blocks):
            for t in range(n):
                if S[bi, t] < 0:
                    U2[:, t] -= 2 * PB(U[:, t], B)
            if np.prod(S[bi]) < 0:
                W2 -= 2 * PB(W, B)
        acc = acc + table(U2, W2); cnt += 1
    avg = acc / cnt
    BU = np.array([[PB(U[i, t], B) for t in range(n)] for i in range(2) for B in blocks])
    BW = np.array([PB(W[i], B) for i in range(2) for B in blocks])
    worst = max(worst, np.abs(avg - table(BU, BW)).max())
    b1, bT1, _ = atoms(U, W); b2, bT2, _ = atoms(BU, BW)
    worst = max(worst, max(abs(b1[j] - b2[j]) for j in b1), max(abs(bT1[k] - bT2[k]) for k in bT1))
    for b in [2.0, 2.7, n + 1]:
        costok &= (sqnorm(BU, BW) ** (b / 2)).sum() <= (sqnorm(U, W) ** (b / 2)).sum() + 1e-12
say(f"  block split = block-group average & keeps beta, beta_T: max err {worst:.2e}; cost not raised: {costok}")

say("== E1(g): dilation union ==")
worst = 0; minslack = 1e9
for (p, n) in [(7, 2), (5, 3), (3, 3)]:
    nu = n + 1
    U, W = generic_net(p, n)
    h = margin(U, W)
    f = table(U, W)
    G = gaps(f, p, n); mprof = np.concatenate([[np.nan], G.min(0)])  # m[d], d=1..p-1
    for s in [1.3, nu / 2]:
        b = nu / s
        J = list(range(1, p)); z = rng.dirichlet(np.ones(len(J)))
        Us, Ws = [], []
        for j, zj in zip(J, z):
            idx = (j * np.arange(p)) % p
            sc = zj ** (1 / b) / 1.0
            Us.append(U[:, :, idx] * sc); Ws.append(W[:, idx] * sc)
        UU = np.concatenate(Us); WW = np.concatenate(Ws)
        # scale: copy j has cost z_j * P where P = cost(theta)
        hT = margin(UU, WW)
        lgT, HT, FT = logF(UU, WW, s, hT); lg, H, F = logF(U, W, s, h)
        lb = min(sum(zj ** s * mprof[(j * d) % p] for j, zj in zip(J, z)) for d in range(1, p))
        esc = z ** s / (z ** s).sum()
        lbF = (s - 1) * (ent(z) - renyi(z, s)) + math.log(min(sum(esc[q] * mprof[(J[q] * d) % p] for q in range(len(J))) for d in range(1, p)) / h)
        worst = max(worst, abs(HT - H - ent(z)))
        assert hT >= lb - 1e-12 and FT - F >= lbF - 1e-12 and lbF >= -1e-12
        minslack = min(minslack, FT - F - lbF)
say(f"  H additivity err {worst:.2e}; h and F lower bounds hold (min slack {minslack:.2e}), bound >= 0")

say("== E1(h): cancelling pairs ==")
p, n = 5, 2; nu = 3; s = 1.5; b = 2.0
U, W = generic_net(p, n); h = margin(U, W)
lg0, H0, F0 = logF(U, W, s, h); P = shares(U, W, s)[0].sum()
for N in [1, 100, 10000]:
    eps = 0.05
    v = rng.standard_normal((N, n, p)); ww = rng.standard_normal((N, p))
    c1 = (sqnorm(v, ww)) ** (b / 2)
    sc = (eps * P / (2 * c1.sum())) ** (1 / b)
    v *= sc; ww *= sc
    UU = np.concatenate([U, v, v]); WW = np.concatenate([W, ww, -ww])
    if N <= 100:
        assert abs(margin(UU, WW) - h) < 1e-10
    lg, H, F = logF(UU, WW, s, h)
    say(f"  N={N}: gamma ratio {math.exp(lg-lg0):.4f} (pred {(1+eps)**(-s):.4f}), F-F0 = {F-F0:.4f}")

say("== E9: explicit split-entropy decreases ==")
p, n = 11, 2; nu = 3
a = np.arange(p)
def cosv(j, ph=0.0):
    return np.cos(2 * np.pi * j * a / p + ph)
# neuron 1: spread equally over 5 frequencies, norm^2 ratio 0.1 ; neuron 2: pure frequency 1
v1 = sum(cosv(j, 0.3 * j) for j in range(1, 6)); v1 /= np.linalg.norm(v1)
v2 = cosv(1); v2 /= np.linalg.norm(v2)
U = np.array([[v1, v1], [v2, v2]]) ; W = np.array([v1, v2])
U[0] *= math.sqrt(0.1); W[0] *= math.sqrt(0.1)
for b in [2.0, 3.0]:
    s = nu / b
    _, w = shares(U, W, s)
    SU = np.array([[proj(U[i, t], k, p) for t in range(n)] for i in range(2) for k in F_of(p)])
    SW = np.array([proj(W[i], k, p) for i in range(2) for k in F_of(p)])
    _, ws = shares(SU, SW, s)
    say(f"  b={b}: H(theta) = {ent(w):.4f}, H(split) = {ent(ws):.4f}")
# b=2 with DC
v0 = np.ones(p) / math.sqrt(p)
u1 = math.sqrt(0.9) * v0 + math.sqrt(0.1) * v2
U = np.array([[u1, u1], [v2, v2]]); W = np.array([u1, v2])
_, w = shares(U, W, 1.5)
SU = np.array([[proj(U[i, t], k, p) for t in range(n)] for i in range(2) for k in F_of(p)])
SW = np.array([proj(W[i], k, p) for i in range(2) for k in F_of(p)])
_, ws = shares(SU, SW, 1.5)
say(f"  b=2 with DC: H(theta) = {ent(w):.4f}, H(split) = {ent(ws):.4f}")

open(__file__.replace('.py', '.out'), 'w').write("\n".join(out) + "\n")
