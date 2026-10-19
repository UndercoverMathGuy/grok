# v1: pointwise geometry of K_h and D_h (Theorem G).
#  (G1) slack (Choquet) coordinates: phi = (2/p)(h - sum_d sigma_d rho^{l d}), sigma = g_phi - h.
#  (G2) fibre description: phi = a + i b in K_h  <=> |S b| <= kappa a - h ; in D_h <=> cot(pi/p) kappa |b| <= kappa a - h.
#  (G3) the sine matrix S (F x F) and the coverage matrix kappa (F x F) are invertible.
#  (G4) Gap(b, d) = cot(pi/p) sum_l |b_l| kappa_l(d) - |sum_l b_l sin(2 pi l d/p)| >= 0, = 0 iff b = t e_l, l d = +-1.
#  (G5) over a binding allocation (sigma^+ = 0 on B != empty) the D-fibre is {0} and the K-fibre is
#       S^{-1}(box), of dimension |F| - rank(S_B); computed at programme maximisers.
#  (G6) Re maps K_h into P_h = K_h cap R^F = D_h cap R^F.
import numpy as np
from fmlib import *
from cjlib import gamma_exact

rng = np.random.default_rng(7)
out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

# ---------- (G3)
worst_cond = 0
for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 199]:
    F = (p - 1) // 2
    d = np.arange(1, F + 1)[:, None]; l = np.arange(1, F + 1)[None, :]
    Smat = np.sin(TAU * d * l / p); Kmat = 1 - np.cos(TAU * d * l / p)
    sS = np.linalg.svd(Smat, compute_uv=False); sK = np.linalg.svd(Kmat, compute_uv=False)
    worst_cond = max(worst_cond, sS.max() / sS.min(), sK.max() / sK.min())
    # exact facts: S^2 = (p/4) I (DST-I type) ; kappa = J - C with C^2 = (p/4)I - J/2
    e1 = np.abs(Smat @ Smat - (p / 4) * np.eye(F)).max()
    Cm = np.cos(TAU * d * l / p); e2 = np.abs(Cm @ Cm - (p / 4) * np.eye(F) + 0.5).max()
    if p in (5, 11, 101, 199):
        say(f'(G3) p={p}: min sing S={sS.min():.4f} (sqrt(p)/2={np.sqrt(p)/2:.4f}), min sing kappa={sK.min():.4f};'
            f' |S^2-(p/4)I|={e1:.1e}, |C^2-(p/4)I+J/2|={e2:.1e}')
say(f'(G3) max condition number over primes <= 199: {worst_cond:.3f}')

# ---------- (G1),(G2),(G4),(G6)
errG1 = 0; mism = 0; tot = 0; gapmin = np.inf; gap_single_ok = True; reproj_bad = 0
for p in [5, 7, 11, 13, 17]:
    F = (p - 1) // 2; d = np.arange(1, p)
    Kfull = kappa_mat(p)                                           # (p-1, F)
    Sfull = np.sin(TAU * d[:, None] * np.arange(1, F + 1)[None, :] / p)
    cot = 1 / np.tan(np.pi / p)
    for trial in range(4000):
        a = rng.normal(size=F) + rng.uniform(0, 2); b = rng.normal(size=F) * rng.choice([1e-3, 0.1, 0.5, 2])
        if trial % 3 == 0: b[rng.random(F) < 0.6] = 0
        phi = a + 1j * b
        g = np.real(np.exp(0j) * (phi[None, :] * (1 - np.exp(-1j * TAU * d[:, None] * np.arange(1, F + 1)[None, :] / p)))).sum(1)
        h = rng.uniform(-1, 1) * np.abs(g).max() * 0.5 + g.min() * 0.5
        sig = g - h
        rec = (2 / p) * (h - (sig[:, None] * np.exp(1j * TAU * d[:, None] * np.arange(1, F + 1)[None, :] / p)).sum(0))
        errG1 = max(errG1, np.abs(rec - phi).max() / np.abs(phi).max())
        inK = g.min() >= h
        inK2 = np.all(np.abs(Sfull @ b) <= Kfull @ a - h + 1e-12)
        cmv = a - cot * np.abs(b)
        inD = (Kfull @ cmv).min() >= h
        inD2 = np.all(cot * (Kfull @ np.abs(b)) <= Kfull @ a - h + 1e-12)
        tot += 1; mism += (inK != inK2) + (inD != inD2) + (inD and not inK)
        gap = cot * (Kfull @ np.abs(b)) - np.abs(Sfull @ b)
        gapmin = min(gapmin, gap.min())
        if inK and (Kfull @ a - h).min() < -1e-9: reproj_bad += 1          # Re phi must be in P_h
    # single-support equality cases
    for l in range(1, F + 1):
        b = np.zeros(F); b[l - 1] = 1.7
        gap = cot * (Kfull @ np.abs(b)) - np.abs(Sfull @ b)
        zero_at = [int(x) for x in d[np.abs(gap) < 1e-12]]
        expect = sorted({int((pow(l, -1, p) * e) % p) for e in (1, p - 1)})
        if sorted(zero_at) != expect: gap_single_ok = False
say(f'(G1) slack-coordinate reconstruction max rel error {errG1:.2e}')
say(f'(G2) membership mismatches (K, D, D subset K) over {tot} random points: {mism}')
say(f'(G4) min Gap = {gapmin:.3e} (>= 0); zero set of single-support b is exactly l d = +-1: {gap_single_ok}')
say(f'(G6) points of K_h whose real part leaves P_h: {reproj_bad}')

# ---------- (G5) fibres over programme maximisers
for (p, s) in [(11, 1.3), (11, 1.5), (13, 1.5), (13, 2.0), (17, 1.5), (19, 1.8), (23, 1.4)]:
    F = (p - 1) // 2
    G, ys = gamma_exact(p, s)
    y = ys[0]; Kfull = kappa_mat(p)
    cov = Kfull[:F] @ y; B = [int(dd) for dd in np.arange(1, F + 1)[cov <= cov.min() + 1e-9]]
    Smat = np.sin(TAU * np.array(B)[:, None] * np.arange(1, F + 1)[None, :] / p)
    rk = np.linalg.matrix_rank(Smat, tol=1e-9)
    say(f'(G5) p={p} s={s}: K*={int((y>0).sum())}, binding offsets (mod +-) {B}; '
        f'dim K-fibre = |F|-rank = {F}-{rk} = {F-rk}; dim D-fibre = 0')

open('v1_geometry.out', 'w').write('\n'.join(out) + '\n')

# ---------- (G7) compensation lineality at a decoupled IF optimum: i I_B, I_B = {b : S b = 0 on B}
#   a + t i b in K_h for |t| small (two-sided), never in D_h; first-order cost change of IF blocks is 0
#   (|a_l + i t b_l|^r - a_l^r = O(t^2)) and the programme dual mu* (on B) does not see it.
out2 = []
for (p, s) in [(11, 1.5), (13, 2.0), (17, 1.5), (23, 1.4)]:
    F = (p - 1) // 2; r = 1 / s
    G, ys = gamma_exact(p, s); y = ys[0]
    a = y / G                                       # allocation with margin h = M(a) = 1
    h = 1.0
    Kfull = kappa_mat(p)[:F]
    sig = Kfull @ a - h; B = np.where(sig <= 1e-9)[0]
    Smat = np.sin(TAU * np.arange(1, F + 1)[:, None] * np.arange(1, F + 1)[None, :] / p)
    # basis of I_B: b = (4/p) S v with v supported off B
    off = [i for i in range(F) if i not in B]
    ok_K = ok_notD = True; cost1 = 0
    for trial in range(200):
        v = np.zeros(F); v[off] = rng.normal(size=len(off)); b = (4 / p) * Smat @ v
        assert np.abs((Smat @ b)[B]).max() < 1e-10
        tmax = 0.5 * sig[off].min() / max(1e-12, np.abs(v[off]).max())
        for t in [tmax, -tmax]:
            if np.any(np.abs(Smat @ (t * b)) > sig + 1e-12): ok_K = False
            cot = 1 / np.tan(np.pi / p)
            if np.all(cot * (Kfull @ np.abs(t * b)) <= sig + 1e-12): ok_notD = False
        tt = 1e-4 * tmax
        dc = np.sum(np.abs(a + 1j * tt * b) ** r - a ** r)
        cost1 = max(cost1, abs(dc) / tt)
    out2.append(f'(G7) p={p} s={s}: |B|={len(B)}, dim i*I_B = {F-len(B)}; two-sided in K_h: {ok_K}; outside D_h: {ok_notD}; '
                f'|first-order cost slope| <= {cost1:.1e}')
for s_ in out2: print(s_)
with open('v1_geometry.out', 'a') as fh: fh.write('\n'.join(out2) + '\n')
