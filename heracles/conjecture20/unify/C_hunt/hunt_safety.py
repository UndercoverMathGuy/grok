# SAFETY CURVE (class C, any n): best ratio gamma / (nu^{1-s} B_p Gamma^{(S)}) over networks that CONTAIN cheap features of a
# prescribed atom profile. Usage: python hunt_safety.py p n b Smode restarts seed
#   cheap set C = one feature l of S* (each in turn) or all of S*; its atoms are frozen & phase-aligned with masses
#   (1, 1, c) (width 3, gain G(c) > 1 exactly) or (1, 1) (c = 0, width 2); everything else is free: the other features
#   (3 neurons each), their phases, the cheap features' sign-flip phases xi, and 2 compensator neurons at every
#   frequency of S outside S* (free: they may form features or pure zero-mean interference).
import sys, os, json, time
import numpy as np
import hlib as H

p = int(sys.argv[1]); n = int(sys.argv[2]); bexp = float(sys.argv[3]); Smode = sys.argv[4]
nrest = int(sys.argv[5]); seed = int(sys.argv[6])
nu = n + 1; s = nu / bexp; r = 1 / s; F = (p - 1) // 2
rng = np.random.default_rng(seed)
S = list(range(1, F + 1)) if Smode == 'F' else None
if Smode == 'Sstar':
    G0, y0, _ = H.programme(p, s); S = [l for l in range(1, F + 1) if y0[l - 1] > 0]
GS, yS, bindS = H.programme(p, s, S=S, restarts=60)
Sstar = [l for l in S if yS[l - 1] > 0]
D = nu ** (1 - s) * GS
comp = [l for l in S if l not in Sstar]
print(f"# safety p={p} n={n} b={bexp} S={Smode}:{S} Gamma={GS:.10f} S*={Sstar} compensators={comp}", flush=True)


def gain_of(c):
    w = np.array([1.0, 1.0, c] + [c] * (nu - 3))[:nu] if c > 0 else np.array([1.0, 1.0])
    w = w / w.sum(); Q = np.sum(w ** r)
    return 1.0 / (nu ** (1 - s) * Q ** s)


t0 = time.time()
cheapsets = [[l] for l in Sstar] + ([list(Sstar)] if len(Sstar) > 1 else [])
results = {}
for C in cheapsets:
    for c in [0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.0]:
        bestv = -np.inf
        for rr in range(nrest):
            freqs = []; b = []; ph = []
            for l in Sstar:
                if l in C:
                    w = [1.0, 1.0] + [c] * (nu - 2) if c > 0 else [1.0, 1.0]
                    w = np.array(w) / sum(w)
                else:
                    w = np.ones(nu) / nu
                m = len(w)
                for k in range(m):
                    freqs.append(l); b.append(yS[l - 1] * w[k])
                    ph.append(2 * np.pi * k / m + [0.05, 0.3, 1.0, 3.0][rr % 4] * rng.standard_normal(n))
            for l in comp:
                for k in range(2):
                    freqs.append(l); b.append(0.05 * yS.max() * np.exp(1j * rng.uniform(0, 2 * np.pi)))
                    ph.append(rng.uniform(0, 2 * np.pi, n))
            cc = H.CC(p, n, freqs, s)
            x = cc.pack(np.array(b, complex), np.array(ph))
            frozen = np.array([cc.freqs[i] in C for i in range(cc.m)])
            free = np.r_[~frozen, ~frozen, np.ones(cc.m * n, bool)]
            x = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=800, free=free)
            x, v = cc.slp(x, iters=120, free=free)
            v /= D
            if v > bestv:
                bestv = v; fd = cc.feature_data(x)
                info = {l: (round(fd[l]['A'] / max(d['A'] for d in fd.values()), 4), round(fd[l]['gain'], 5)) for l in fd}
        results[(tuple(C), c)] = bestv
        print(json.dumps(dict(cheap=C, c=c, gain=gain_of(c), best=bestv, feat_relA_gain=info), default=float), flush=True)
print(f"# done {time.time() - t0:.0f}s", flush=True)
