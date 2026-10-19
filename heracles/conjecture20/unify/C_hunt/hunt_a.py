# HUNT (a): compensation at n = 2 in class C. Usage: python hunt_a.py p b Smode nrest seed
#   Smode: F (all features allowed), Sstar (S = spt y*), AP<d>_<K>_<a> (S = {a + k d : k < K}, reduced to F)
# Seed families:
#   if_jit     IF optimum on S* (restricted programme) + jitter
#   massdes    IF with per-neuron mass perturbation eps and random neuron deletions (cheap features), then
#              PHASE DESIGN with masses frozen (maximise the true margin over phases only = flat-bottom design),
#              then full release
#   widths     random sub-support, random widths 2..4 per feature, m-gon phases + jitter
#   extra      S* plus 1..4 extra features of S at small amplitude (compensators), widths 2..3
#   cheapfix   every feature of the support has masses (a, a, c a) with c in {0.6, 0.35, 0.15}: phase design only
#              (records how close a compensated network with PRESCRIBED cheapness gets), then release
import sys, os, json, time
import numpy as np
import hlib as H

p = int(sys.argv[1]); bexp = float(sys.argv[2]); Smode = sys.argv[3]; nrest = int(sys.argv[4]); seed = int(sys.argv[5])
n = 2; nu = 3; s = nu / bexp; r = 1 / s; F = (p - 1) // 2
rng = np.random.default_rng(seed)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
tag = f"a_p{p}_b{bexp}_{Smode}_s{seed}"


def red(l):
    l %= p
    return min(l, p - l)


if Smode == 'F':
    S = list(range(1, F + 1))
elif Smode == 'Sstar':
    G0, y0, _ = H.programme(p, s)
    S = [l for l in range(1, F + 1) if y0[l - 1] > 0]
elif Smode.startswith('AP'):
    d, K, a = [int(v) for v in Smode[2:].split('_')]
    S = sorted(set(red(a + k * d) for k in range(K)) - {0})
else:
    raise SystemExit('bad Smode')

GS, yS, bindS = H.programme(p, s, S=S, restarts=60)
Sstar = [l for l in S if yS[l - 1] > 0]
D = nu ** (1 - s) * GS          # W/D denominator in units B_p = 1 (pi = 3^{1-s} at n = 2)
print(f"# {tag}: S={S} Gamma^S={GS:.10f} supp={Sstar} K*_S={len(Sstar)} bind={bindS}", flush=True)


def finish(cc, x):
    x = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=800)
    x, v = cc.slp(x, iters=150)
    return x, v / D


def summarize(cc, x, ratio):
    fd = cc.feature_data(x)
    Amax = max(d['A'] for d in fd.values())
    feats = [l for l in fd if fd[l]['A'] > 1e-6 * Amax]
    gains = {l: round(fd[l]['gain'], 6) for l in feats}
    widths = {l: fd[l]['width'] for l in feats}
    b, _ = cc.unpack(x)
    # dead features: neurons present but A ~ 0 (pure interference fields)
    ghost = [l for l in fd if fd[l]['A'] <= 1e-6 * Amax and fd[l]['L'] > 1e-6 * np.abs(b).sum()]
    Qt = sum(fd[l]['Q'] for l in fd)
    shares = {l: round(fd[l]['Q'] / Qt, 5) for l in feats}
    sig = [gains[l] for l in feats if shares[l] >= 0.01]
    return dict(ratio=ratio, feats=feats, count=len(feats), gains=gains, widths=widths, ghost=ghost, shares=shares,
                maxgain=max(gains.values()), maxgain_sig=max(sig) if sig else 0.0)


def model(freqs):
    return H.CC(p, n, freqs, s)


def if_x(cc, ys, jit, wid=None):
    b = np.zeros(cc.m, complex); phi = np.zeros((cc.m, n)); cnt = {}
    for i, l in enumerate(cc.freqs):
        k = cnt.get(l, 0); cnt[l] = k + 1
        ml = wid[l] if wid else nu
        b[i] = ys[l - 1] / ml
        phi[i] = 2 * np.pi * k / ml
    b = b * (1 + jit * rng.standard_normal(cc.m)) * np.exp(1j * jit * rng.standard_normal(cc.m))
    phi = phi + jit * rng.standard_normal(phi.shape)
    return cc.pack(b, phi)


records = []; best = (-np.inf, None, None)
t0 = time.time()
fams = ['if_jit', 'massdes', 'widths', 'extra', 'cheapfix']
for it in range(nrest):
    fam = fams[it % len(fams)]
    rec = dict(fam=fam, it=it)
    if fam == 'if_jit':
        jit = [0.02, 0.1, 0.3, 0.6][(it // len(fams)) % 4]
        freqs = [l for l in Sstar for _ in range(nu)]
        cc = model(freqs); x = if_x(cc, yS, jit); rec['jit'] = jit
    elif fam == 'massdes':
        eps = [0.1, 0.25, 0.5, 0.8][(it // len(fams)) % 4]
        freqs = []; bl = []
        for l in Sstar:
            for k in range(nu):
                if rng.random() < 0.25 and k == nu - 1: continue        # delete a neuron: width-2 feature
                freqs.append(l)
        cc = model(freqs)
        cnt = {}
        b = np.zeros(cc.m, complex); phi = np.zeros((cc.m, n))
        for i, l in enumerate(freqs):
            k = cnt.get(l, 0); cnt[l] = k + 1
            b[i] = yS[l - 1] / nu * np.exp(eps * rng.standard_normal())
            phi[i] = 2 * np.pi * k / nu + eps * rng.standard_normal(n)
        x = cc.pack(b, phi)
        free = np.r_[np.zeros(2 * cc.m, bool), np.ones(cc.m * n, bool)]      # atoms frozen & aligned: gain fixed
        xd = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=800, free=free)
        xd, _ = cc.slp(xd, iters=60, free=free)
        rec['eps'] = eps; rec['designed'] = cc.gnorm(xd) / D
        fd = cc.feature_data(xd); rec['designed_gain'] = max(d['gain'] for d in fd.values())
        x = xd
    elif fam == 'widths':
        k = int(rng.integers(max(1, len(Sstar) - 1), min(len(S), len(Sstar) + 3) + 1))
        sub = sorted(rng.choice(S, size=k, replace=False).tolist())
        Gs, ys, _ = H.programme(p, s, S=sub, restarts=10, seed=int(rng.integers(1 << 30)))
        sub = [l for l in sub if ys[l - 1] > 0]
        wid = {l: int(rng.integers(2, 5)) for l in sub}
        freqs = [l for l in sub for _ in range(wid[l])]
        cc = model(freqs); x = if_x(cc, ys, 0.15, wid); rec['sub'] = sub; rec['wid'] = wid
    elif fam == 'extra':
        cand = [l for l in S if l not in Sstar]
        if not cand:
            cand = list(Sstar)
        ne = int(rng.integers(1, min(4, len(cand)) + 1))
        ext = rng.choice(cand, size=ne, replace=False).tolist()
        ys = yS.copy(); amp = float(rng.choice([0.03, 0.1, 0.3]))
        wid = {l: nu for l in Sstar}
        for l in ext:
            ys[l - 1] += amp * yS.max(); wid[l] = int(rng.integers(2, 4))
        sup = sorted(set(Sstar) | set(ext))
        freqs = [l for l in sup for _ in range(wid[l])]
        cc = model(freqs); x = if_x(cc, ys, 0.2, wid); rec['ext'] = ext; rec['amp'] = amp
    else:  # cheapfix
        c = [0.6, 0.35, 0.15][(it // len(fams)) % 3]
        freqs = [l for l in Sstar for _ in range(nu)]
        cc = model(freqs)
        b = np.zeros(cc.m, complex); phi = np.zeros((cc.m, n)); cnt = {}
        for i, l in enumerate(freqs):
            k = cnt.get(l, 0); cnt[l] = k + 1
            w = [1, 1, c][k]; b[i] = yS[l - 1] * w / (2 + c)
            phi[i] = 2 * np.pi * k / nu + [0.1, 0.4, 1.0][(it // 15) % 3] * rng.standard_normal(n)
        x = cc.pack(b, phi)
        free = np.r_[np.zeros(2 * cc.m, bool), np.ones(cc.m * n, bool)]
        xd = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=800, free=free)
        xd, _ = cc.slp(xd, iters=60, free=free)
        rec['c'] = c; rec['designed'] = cc.gnorm(xd) / D
        fd = cc.feature_data(xd); rec['designed_gain'] = max(d['gain'] for d in fd.values())
        x = xd
    x, ratio = finish(cc, x)
    rec.update(summarize(cc, x, ratio))
    records.append(rec)
    if ratio > best[0]:
        best = (ratio, cc.freqs.copy(), x.copy())
    if ratio > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_{tag}_{it}.npz"), freqs=cc.freqs, x=x, p=p, n=n, b=bexp, S=S, ratio=ratio)
    print(json.dumps(rec, default=float), flush=True)

np.savez(os.path.join(OUT, f"best_{tag}.npz"), freqs=best[1], x=best[2], p=p, n=n, b=bexp, S=S, ratio=best[0])
rs = [z['ratio'] for z in records]
near = [z for z in records if z['ratio'] > 1 - 1e-6]
cheap = [z['ratio'] for z in records if z['maxgain_sig'] > 1 + 1e-3]
des = [z['designed'] for z in records if 'designed' in z]
print(f"# SUMMARY {tag}: runs={len(rs)} best={max(rs):.10f} K*_S={len(Sstar)} "
      f"near-opt counts={sorted(set(z['count'] for z in near))} near-opt supports={sorted(set(tuple(z['feats']) for z in near))} "
      f"best_with_significant_cheap_feature={max(cheap) if cheap else None} best_designed={max(des) if des else None} "
      f"time={time.time() - t0:.0f}s", flush=True)
