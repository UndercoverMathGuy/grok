"""Sanity checks for R1 (free-energy calculus of the operations) and R2(a) (F = tangent intercept).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy --with mpmath python check_calculus.py"""
import math, itertools
import numpy as np
from elib import *

rng = np.random.default_rng(1)
out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def pos_net(p, n, extra=2, noise=0.05, dc=0.1):
    """IF network on all of F (uniform y) + small random neurons (multi-frequency, DC) -> positive margin, generic."""
    U, W = if_network(p, n, np.ones(len(Fset(p))))
    U = U + noise * rng.standard_normal(U.shape); W = W + noise * rng.standard_normal(W.shape)
    U2 = 0.3 * rng.standard_normal((extra, n, p)) + dc; W2 = 0.3 * rng.standard_normal((extra, p))
    return np.concatenate([U, U2]), np.concatenate([W, W2])


say("== (1) free energy: three formulas, tangent intercept, integral identity ==")
worst = 0.0
for (p, n) in [(5, 2), (7, 2), (5, 3)]:
    for trial in range(4):
        U, W = pos_net(p, n)
        h = margin(U, W)
        if h <= 0:
            continue
        nu = n + 1
        for s in [1.2, 1.5, nu / 2]:
            st = stats(U, W, s, h)
            b = nu / s
            c = st['c']; w = st['w']
            F2 = math.log(h) - math.log(c.sum()) - (s - 1) * float((w * np.log(c)).sum())
            nrm = np.sqrt(norms2(U, W))
            F3 = math.log(h) - math.log(c.sum()) - float((w * np.log(nrm ** (nu - b))).sum())
            # tangent intercept: log gamma(s) + (s-1) * (-d/ds log gamma)
            e = 1e-6
            d = (math.log(stats(U, W, s + e, h)['gamma']) - math.log(stats(U, W, s - e, h)['gamma'])) / (2 * e)
            F4 = math.log(st['gamma']) - (s - 1) * d
            # integral identity F(s) = log gamma(1) + int_1^s (sig-1) V(sig)/sig
            sig = np.linspace(1, s, 401)
            vals = [(x - 1) * stats(U, W, x, h)['V'] / x for x in sig]
            I = np.trapezoid(vals, sig) if hasattr(np, 'trapezoid') else np.trapz(vals, sig)
            F5 = math.log(stats(U, W, 1.0, h)['gamma']) + I
            worst = max(worst, abs(F2 - st['F']), abs(F3 - st['F']), abs(F4 - st['F']), abs(F5 - st['F']) / 10)
            assert st['F'] >= math.log(stats(U, W, 1.0, h)['gamma']) - 1e-12
say(f"max discrepancy among F formulas (intercept via FD, integral via trapezoid/10): {worst:.2e}")

say("== (2) replication ==")
for (p, n) in [(5, 2), (7, 2), (5, 3)]:
    U, W = pos_net(p, n)
    nu = n + 1
    for s in [1.3, nu / 2]:
        b = nu / s
        st = stats(U, W, s)
        for k in [2, 3]:
            lam = k ** (-1 / b)
            Uk = np.concatenate([U * lam] * k); Wk = np.concatenate([W * lam] * k)
            sk = stats(Uk, Wk, s)
            say(f"(p,n,s,k)=({p},{n},{s:.3f},{k}): dlog gamma - (1-s)log k = {math.log(sk['gamma']/st['gamma']) - (1-s)*math.log(k):.2e}, "
                f"dH - log k = {sk['H']-st['H']-math.log(k):.2e}, dF = {sk['F']-st['F']:.2e}")
        for t in [0.5, 0.2, 0.05]:
            l1, l2 = t ** (1 / b), (1 - t) ** (1 / b)
            Ut = np.concatenate([U * l1, U * l2]); Wt = np.concatenate([W * l1, W * l2])
            sk = stats(Ut, Wt, s)
            pred = (s - 1) * (H([t, 1 - t]) - Hrenyi([t, 1 - t], s))
            say(f"   uneven replication t={t}: dF = {sk['F']-st['F']:.6f}, predicted (s-1)(H-H_s) = {pred:.6f}, "
                f"dH = {sk['H']-st['H']:.4f}, dlog gamma = {math.log(sk['gamma']/st['gamma']):.4f}")

say("== (3) merging two positively parallel neurons ==")
pos_dF = 0; neg_dF = 0; worst_iso = 0
for trial in range(200):
    p, n = [(5, 2), (7, 2), (5, 3)][trial % 3]
    nu = n + 1
    s = rng.uniform(1.05, nu / 2); b = nu / s
    U, W = pos_net(p, n, extra=1)
    i = rng.integers(len(W)); lam = rng.uniform(0.1, 1.5)
    Ua = np.concatenate([U, U[i:i + 1] * lam]); Wa = np.concatenate([W, W[i:i + 1] * lam])   # parallel copy
    mu = (1 + lam ** nu) ** (1 / nu)
    Um = U.copy(); Wm = W.copy(); Um[i] *= mu; Wm[i] *= mu                                      # merged
    assert np.allclose(table(Ua, Wa), table(Um, Wm))
    h = margin(Um, Wm)
    if h <= 0:
        continue
    sa = stats(Ua, Wa, s, h); sm = stats(Um, Wm, s, h)
    assert sm['Z'] < sa['Z']
    if sm['F'] > sa['F']:
        pos_dF += 1
    else:
        neg_dF += 1
    # isolated pair formula
    c1 = costs(U[i:i + 1], W[i:i + 1], b)[0]; c2 = c1 * lam ** b; t = c1 / (c1 + c2)
    s2 = stats(np.concatenate([U[i:i + 1], U[i:i + 1] * lam]), np.concatenate([W[i:i + 1], W[i:i + 1] * lam]), s, 1.0)
    s1 = stats(U[i:i + 1] * mu, W[i:i + 1] * mu, s, 1.0 * (1 + lam ** nu) / 1.0)
    # same table: h(merged) = h(pair); use h=1 for the pair, h = 1 for merged too (identical tables)
    s1 = stats(U[i:i + 1] * mu, W[i:i + 1] * mu, s, 1.0)
    worst_iso = max(worst_iso, abs((s2['F'] - s1['F']) - (s - 1) * (H([t, 1 - t]) - Hrenyi([t, 1 - t], s))))
say(f"merging always lowers cost (asserted). In-network sign of F(merged)-F(pair): positive {pos_dF}, negative {neg_dF}")
say(f"isolated pair: F(pair)-F(merged) = (s-1)(H(t)-H_s(t)) to {worst_iso:.2e}")

say("== (4) frequency split: cost, entropy chain rule with escort reweighting ==")
worst = 0
for trial in range(60):
    p, n = [(5, 2), (7, 2), (7, 3), (11, 2)][trial % 4]
    nu = n + 1
    b = [2.0, 2.4, 3.0][trial % 3]
    if b > nu:
        b = nu
    s = nu / b
    U = rng.standard_normal((4, n, p)); W = rng.standard_normal((4, p))
    if trial % 2 == 0:   # remove DC
        U = U - U.mean(-1, keepdims=True); W = W - W.mean(-1, keepdims=True)
    e = piece_energies(U, W)          # (m, |F|+1)
    E = e.sum(1)
    rho = e / E[:, None]
    c = E ** (b / 2); w = c / c.sum()
    lamb = (rho[:, 1:] ** (b / 2)).sum(1)
    Us, Ws = split(U, W)              # pieces ordered (i, k)
    cs = costs(Us, Ws, b)
    assert abs(cs.sum() - (c * lamb).sum()) < 1e-9 * c.sum()
    ws = cs / cs.sum()
    what = w * lamb / (w * lamb).sum()
    pis = rho[:, 1:] ** (b / 2) / lamb[:, None]
    Hpred = H(what) + float((what * np.array([H(pi) for pi in pis])).sum())
    worst = max(worst, abs(H(ws) - Hpred))
    if b == 2.0 and trial % 2 == 0:
        dH = H(ws) - H(w); pred = float((w * np.array([H(r[1:]) for r in rho])).sum())
        worst = max(worst, abs(dH - pred)); assert dH >= -1e-12
say(f"split cost = sum_i c_i lambda_i and H(split) = H(what) + sum what_i H(pi_i): max err {worst:.2e}; b=2 no-DC dH>=0 asserted")
# explicit decrease examples
p, n = 11, 2; nu = 3; a = np.arange(p)
K = 5
u_mix = sum(np.cos(2 * np.pi * k * a / p) for k in range(1, K + 1)); u_mix /= np.linalg.norm(u_mix)
u_pure = np.cos(2 * np.pi * a / p); u_pure /= np.linalg.norm(u_pure)
eps = 0.1   # neuron 1 (mixed over 5 frequencies) has norm^2 = eps * norm^2 of neuron 2 (pure)
U = np.array([[u_mix * np.sqrt(eps), u_mix * np.sqrt(eps)], [u_pure, u_pure]]); W = np.array([u_mix * np.sqrt(eps), u_pure])
for b in [2.0, 2.5, 3.0]:
    Us, Ws = split(U, W)
    say(f"  b={b}: H(theta)={H(shares(costs(U, W, b))):.4f}, H(split)={H(shares(costs(Us, Ws, b))):.4f}  (small neuron spread over 5 freqs, big neuron pure)")
# DC example at b=2: neuron 1 has the same norm as neuron 2 but 90% of its energy is DC
v = u_pure * np.sqrt(0.1) + np.sqrt(0.9) * np.ones(p) / np.sqrt(p)
U = np.array([[v, v], [u_pure, u_pure]]); W = np.array([v, u_pure])
Us, Ws = split(U, W)
say(f"  b=2 with DC (equal norms, neuron 1 is 90% DC): H(theta)={H(shares(costs(U, W, 2.0))):.4f}, H(split)={H(shares(costs(Us, Ws, 2.0))):.4f}")

say("== (5) dilation-orbit union ==")
worst_H = 0; min_gain = 1e9
for trial in range(30):
    p, n = [(7, 2), (11, 2), (7, 3)][trial % 3]
    nu = n + 1
    s = rng.uniform(1.05, nu / 2); b = nu / s
    y = rng.uniform(0, 1, len(Fset(p))) * (rng.uniform(0, 1, len(Fset(p))) < 0.7); y[0] = 1.0
    U, W = if_network(p, n, y)
    U = U + 0.03 * rng.standard_normal(U.shape); W = W + 0.03 * rng.standard_normal(W.shape)
    P = costs(U, W, b).sum()
    mprof = profile(U, W); h = mprof[1:].min()
    if h <= 0:
        continue
    J = [j for j in Fset(p) if rng.uniform() < 0.6] or [1]
    z = rng.uniform(0.2, 1, len(J)); z /= z.sum()
    Us_, Ws_ = [], []
    for j, zj in zip(J, z):
        Uj, Wj = dilate(U, W, j); Uj, Wj = scale_to_cost(Uj, Wj, b, zj * P)
        Us_.append(Uj); Ws_.append(Wj)
    Uu = np.concatenate(Us_); Wu = np.concatenate(Ws_)
    su = stats(Uu, Wu, s); st = stats(U, W, s, h)
    worst_H = max(worst_H, abs(su['H'] - st['H'] - H(z)))
    lb = min(sum(zj ** s * mprof[(j * d) % p] for j, zj in zip(J, z)) for d in range(1, p))
    assert su['h'] >= lb - 1e-9
    zhat = escort(z, s)
    bound = (s - 1) * (H(z) - Hrenyi(z, s)) + math.log(min(sum(zh * mprof[(j * d) % p] / h for j, zh in zip(J, zhat)) for d in range(1, p)))
    assert su['F'] - st['F'] >= bound - 1e-9 and bound >= -1e-12
    min_gain = min(min_gain, su['F'] - st['F'] - bound)
say(f"H(union) = H(theta)+H(z) to {worst_H:.2e}; h(union) >= min_delta sum z^s m(j delta) and dF >= (s-1)(H(z)-H_s(z)) + log(min escort-avg profile/h) >= 0 asserted; min slack {min_gain:.2e}")

say("== (6) block split = block-sign-group average; cost non-increasing; own data kept ==")
for (p, n, blocks) in [(7, 2, [[0, 1], [2], [3]]), (7, 2, [[0], [1, 2], [3]]), (5, 3, [[0, 2], [1]]), (11, 2, [[0, 1, 3], [2, 4], [5]])]:
    U = rng.standard_normal((3, n, p)); W = rng.standard_normal((3, p))
    Ub, Wb = block_split(U, W, blocks)
    nb = len(blocks)
    avg = np.zeros((p,) * (n + 1))
    signs = list(itertools.product([1, -1], repeat=nb * n))
    for S in signs:
        Uf, Wf = block_flip(U, W, blocks, np.array(S).reshape(nb, n))
        avg += table(Uf, Wf)
        assert abs(costs(Uf, Wf, 2.5).sum() - costs(U, W, 2.5).sum()) < 1e-9
    avg /= len(signs)
    e1 = np.abs(avg - table(Ub, Wb)).max()
    b0, bT0, _ = own_coeffs(U, W); b1, bT1, _ = own_coeffs(Ub, Wb)
    e2 = max(max(abs(b0[j] - b1[j]) for j in b0), max(abs(bT0[k] - bT1[k]) for k in bT0))
    cmax = max(costs(Ub, Wb, bb).sum() - costs(U, W, bb).sum() for bb in [2.0, 2.5, n + 1.0])
    say(f"p={p} n={n} blocks={blocks}: |avg table - block split table| = {e1:.1e}, own-data change = {e2:.1e}, max cost(block split)-cost = {cmax:.3f} (<=0)")

say("== (7) F is unbounded above: cancelling pairs ==")
p, n = 7, 2; nu = 3; s = 1.4; b = nu / s
U, W = if_network(p, n, np.ones(3)); st = stats(U, W, s)
Z = st['Z']
for N in [1, 10, 100, 1000, 10000]:
    eps = 0.05
    v = rng.standard_normal((N, n, p)); ww = rng.standard_normal((N, p))
    Up = np.concatenate([v, v]); Wp = np.concatenate([ww, -ww])
    Up, Wp = scale_to_cost(Up, Wp, b, eps * Z)
    Ua = np.concatenate([U, Up]); Wa = np.concatenate([W, Wp])
    c = np.concatenate([st['c'], costs(Up, Wp, b)]); w = c / c.sum()
    g = st['h'] / c.sum() ** s; Hh = H(w)
    say(f"N={N:5d} cancelling pairs (5% cost): gamma/gamma0 = {g/st['gamma']:.4f}, H = {Hh:.3f} (H0={st['H']:.3f}), F-F0 = {math.log(g)+(s-1)*Hh-st['F']:.3f}")
open("check_calculus.out", "w").write("\n".join(out) + "\n")
