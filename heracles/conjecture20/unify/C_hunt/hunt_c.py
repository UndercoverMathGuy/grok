# HUNT (c): atom cancellation at n in {3,4,5}, p in {5,7}. Usage: python hunt_c.py p n b nrest seed
# Part 1  class-C single feature (frequency 1), CHEAP atom systems with cancellation, frozen:
#         width m in {nu-1, nu-2}, masses ~equal, atom phases +-theta (cancellation c0 = 1 - cos theta), only the sign-flip
#         phases xi optimised (best ratio at prescribed gain/cancellation), then everything released.
# Part 2  class-C multi-feature networks over F, every feature seeded cheap (width nu-1 or nu-2, cancellation), released.
# Part 3  general single features (outputs at frequency 1 only, inputs arbitrary incl. DC): seeds from the Note's 1-D
#         shared-sign family u_i(a) = eps(a) exp(lambda_i e(a)) (width nu-1, nu, nu+1) and DC-carrying random neurons.
# Ratios: single features gamma/(nu^{1-s} B_p kappa_min); multi-feature gamma/(nu^{1-s} B_p Gamma_s) (IF value).
import sys, os, json, time
import numpy as np
import hlib as H, glib as Gb
GL = H.GL

p = int(sys.argv[1]); n = int(sys.argv[2]); bexp = float(sys.argv[3]); nrest = int(sys.argv[4]); seed = int(sys.argv[5])
nu = n + 1; s = nu / bexp; r = 1 / s; F = (p - 1) // 2
rng = np.random.default_rng(seed)
kmin = 1 - np.cos(2 * np.pi / p)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
tag = f"c_p{p}_n{n}_b{bexp}_s{seed}"
G, yst, bind = H.programme(p, s)
Kst = int((yst > 0).sum())
DSF = nu ** (1 - s) * kmin; DMF = nu ** (1 - s) * G
print(f"# {tag}: Gamma={G:.10f} K*={Kst} y*={np.round(yst, 6).tolist()} bind={bind}", flush=True)
t0 = time.time()


def cheap_atoms(m):
    """width m: nu-1 masses ~1 and m-nu+1 small masses (unequal => cheap even at width >= nu); phases +-theta
    (cancellation c0 = 1 - cos theta); returns b (|sum| = 1) or None if no cheap draw"""
    for _ in range(200):
        k = min(m, nu - 1)
        mags = np.r_[np.exp(0.15 * rng.standard_normal(k)), 10 ** rng.uniform(-2.5, -0.1, m - k)]
        c0 = float(rng.choice([0.0, 0.02, 0.06, 0.12, 0.2]))
        th = np.arccos(1 - c0)
        ph = th * np.where(np.arange(m) % 2 == 0, 1.0, -1.0) * (1 + 0.2 * rng.standard_normal(m))
        b = mags * np.exp(1j * ph)
        A = abs(b.sum()); Q = np.sum(np.abs(b) ** r)
        if A / (nu ** (1 - s) * Q ** s) > 1 + 1e-3:
            return b / A, c0
    return None, None


# ---------------- Part 1
rec1 = []; best1 = -np.inf; bestfz = -np.inf
for it in range(nrest):
    m = [nu - 1, nu, nu + 1, nu, nu + 2, max(2, nu - 2)][it % 6]
    b, c0 = cheap_atoms(m)
    if b is None: continue
    cc = H.CC(p, n, [1] * m, s)
    phi = (2 * np.pi * np.arange(m)[:, None] / m * np.ones((1, n)) + [0.1, 0.5, 2.0][it % 3] * rng.standard_normal((m, n)))
    x = cc.pack(b, phi)
    free = np.r_[np.zeros(2 * m, bool), np.ones(m * n, bool)]
    xf = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=1000, free=free)
    xf, vf = cc.slp(xf, iters=80, free=free)
    fd = cc.feature_data(xf)[1]
    frozen = dict(ratio=vf / DSF, gain=fd['gain'], cancel=1 - fd['A'] / fd['L'])
    xm = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=1000, fix_mag=True)     # masses frozen, atom phases free
    fdm = cc.feature_data(xm)[1]
    fixmass = dict(ratio=cc.gnorm(xm) / DSF, gain=fdm['gain'], cancel=1 - fdm['A'] / fdm['L'])
    xr = cc.lbfgs(xf, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=1000)
    xr, vr = cc.slp(xr, iters=150)
    fd2 = cc.feature_data(xr)[1]
    rr = vr / DSF
    rec = dict(part=1, it=it, m=m, c0=c0, frozen=frozen, fixmass=fixmass,
               free=dict(ratio=rr, gain=fd2['gain'], cancel=1 - fd2['A'] / fd2['L'], width=fd2['width']))
    rec1.append(rec); best1 = max(best1, rr); bestfz = max(bestfz, frozen['ratio'])
    print(json.dumps(rec, default=float), flush=True)
    if rr > 1 + 1e-9 or frozen['ratio'] > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_{tag}_p1_{it}.npz"), freqs=cc.freqs, x=xr, xf=xf, p=p, n=n, b=bexp, S=[1])

# ---------------- Part 2
rec2 = []; best2 = -np.inf
for it in range(nrest):
    sup = [l for l in range(1, F + 1) if yst[l - 1] > 0]
    if it % 3 == 2 and F > len(sup):
        sup = sorted(set(sup) | {int(rng.choice([l for l in range(1, F + 1) if l not in sup]))})
    freqs = []; bs = []; phs = []
    for l in sup:
        m = int(rng.choice([nu - 1, nu, nu, nu + 1]))
        if rng.random() < 0.8:
            b, _ = cheap_atoms(m)
            if b is None: b = np.ones(m) / m
        else:
            b = np.ones(m) / m
        amp = yst[l - 1] if yst[l - 1] > 0 else 0.2 * yst.max()
        freqs += [l] * m; bs += list(b * amp)
        phs += list(2 * np.pi * np.arange(m)[:, None] / m * np.ones((1, n)) + 0.4 * rng.standard_normal((m, n)))
    cc = H.CC(p, n, freqs, s)
    x = cc.pack(np.array(bs), np.array(phs))
    x = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=1000)
    x, v = cc.slp(x, iters=150)
    rr = v / DMF
    fd = cc.feature_data(x); Amax = max(d['A'] for d in fd.values())
    feats = [l for l in fd if fd[l]['A'] > 1e-6 * Amax]
    rec = dict(part=2, it=it, start_support=sup, ratio=rr, feats=feats, count=len(feats), Kstar=Kst,
               gains={l: round(fd[l]['gain'], 6) for l in feats}, widths={l: fd[l]['width'] for l in feats},
               cancel={l: round(1 - fd[l]['A'] / fd[l]['L'], 6) for l in feats})
    rec2.append(rec); best2 = max(best2, rr)
    print(json.dumps(rec, default=float), flush=True)
    if rr > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_{tag}_p2_{it}.npz"), freqs=cc.freqs, x=x, p=p, n=n, b=bexp)

# ---------------- Part 3
a = np.arange(p); e = np.where(a < p / 2, 2 * a, 2 * a - p).astype(float); eps = np.where(a < p / 2, 1.0, -1.0)
ALL = set(range(0, F + 1))
rec3 = []; best3 = -np.inf
for it in range(max(2, nrest // 2)):
    m = int(rng.choice([nu - 1, nu, nu + 1]))
    lam = np.sort(rng.normal(0, 0.6, m))
    U = np.array([[eps * np.exp(l * e) * (1 + 0.05 * rng.standard_normal(p)) for _ in range(n)] for l in lam])
    if it % 2 == 1:   # DC-carrying random inputs instead
        U = np.array([[0.7 + 0.5 * np.cos(2 * np.pi * a / p + rng.uniform(0, 6.3)) + 0.2 * rng.standard_normal(p)
                       for _ in range(n)] for _ in range(m)])
    W = np.array([np.cos(2 * np.pi * a / p + rng.uniform(0, 2 * np.pi)) for _ in range(m)])
    g = Gb.GNet(p, n, bexp, [[ALL] * n + [{1}] for _ in range(m)])
    x = g.from_UW(U, W)
    x = g.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4), iters=1500)
    x, v = g.slp(x, iters=100)
    rr = v / (H.Bp(p, n) * DSF)
    nt = g.net(x); A_, Q_, a_ = nt.cheapness()
    dc = float(np.sum(nt.U.mean(2) ** 2) * p / np.sum(nt.U ** 2))
    rec = dict(part=3, it=it, m=m, seed='1D' if it % 2 == 0 else 'DC', ratio=rr, gain=float(A_[0] / a_[0]), dc_share=dc)
    rec3.append(rec); best3 = max(best3, rr)
    print(json.dumps(rec, default=float), flush=True)
    if rr > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_{tag}_p3_{it}.npz"), U=nt.U, W=nt.W, p=p, n=n, b=bexp)

near = [z for z in rec2 if z['ratio'] > 1 - 1e-6]
cheapfree = [z['free']['ratio'] for z in rec1 if z['free']['gain'] > 1 + 1e-4]
print(f"# SUMMARY {tag}: P1 frozen-cheap best={bestfz:.6f} released best={best1:.10f} best released-still-cheap="
      f"{max(cheapfree) if cheapfree else None} | P2 multi best={best2:.10f} near-opt counts="
      f"{sorted(set(z['count'] for z in near))} K*={Kst} | P3 general sf best={best3:.10f} | time={time.time() - t0:.0f}s",
      flush=True)
