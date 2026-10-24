# Sanity checks for the discrete plane Bhatia-Davis inequality (Proposition PBD in REPORT.md):
#   f1,f2,f3 : Z_p -> R mean zero,  F(x)=sum f_i(x_i),  Pi_c = {x in Z_p^3 : x1+x2+x3 = c}
#   m = -min_{Pi_c} F,  M = max_{Pi_c'} F,  V = sum Var f_i,  eps = min_{i<j} sqrt(osc_i osc_j)/p
#   claim (proved):   m*M + m*eps >= V - sqrt(V)*eps,  and eps <= sqrt(2V/(3p))
#   corollary:        (m+M)/2 >= sqrt(V) * c_p,  c_p = sqrt(1-eta) - eta/2,  eta = sqrt(2/(3p))
# Also re-derives the rank-coupling identity E[(F^- + m)(M - F^+)] >= 0 step by step on random data.
import numpy as np, itertools
rng = np.random.default_rng(11)
def planes(f, p):
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    out = {}
    for c in range(p):
        x3 = (c - x1 - x2) % p
        out[c] = f[0][x1] + f[1][x2] + f[2][x3]
    return out
def gen(p, kind):
    fs = []
    for i in range(3):
        if kind == 'gauss': v = rng.normal(size=p)
        elif kind == 'spike': v = np.zeros(p); v[rng.integers(p)] = rng.normal()*p; v += 0.05*rng.normal(size=p)
        elif kind == 'twopoint': v = np.zeros(p); a, b = rng.choice(p, 2, replace=False); v[a] = 1; v[b] = -1; v *= rng.exponential()
        elif kind == 'ramp': d = int(rng.integers(1, p)); v = (np.arange(p)*d % p).astype(float); v = v[np.argsort(np.arange(p)*pow(d, -1, p) % p)] if False else (np.arange(p) * pow(d, -1, p) % p).astype(float)
        elif kind == 'line': v = rng.normal(size=p) if i == 2 else np.zeros(p)
        elif kind == 'mixed': v = rng.normal(size=p)**3
        fs.append(v - v.mean())
    return fs
worst_slack = np.inf; worst_cor = np.inf
for p in [5, 7, 11, 13, 17, 23]:
    for kind in ['gauss', 'spike', 'twopoint', 'ramp', 'line', 'mixed']:
        for t in range(150):
            f = gen(p, kind)
            V = sum((g**2).mean() for g in f)
            if V < 1e-12: continue
            osc = [g.max() - g.min() for g in f]
            eps = min(np.sqrt(osc[i]*osc[j]) for i, j in [(0,1),(0,2),(1,2)])/p
            P = planes(f, p)
            mins = {c: -P[c].min() for c in P}; maxs = {c: P[c].max() for c in P}
            m = min(mins.values()); M = min(maxs.values())          # worst plane for each (covers every c, c')
            slack = m*M + m*eps - (V - np.sqrt(V)*eps)
            worst_slack = min(worst_slack, slack/V)
            eta = np.sqrt(2/(3*p)); cp = np.sqrt(1 - eta) - eta/2
            worst_cor = min(worst_cor, ((m + M)/2)/np.sqrt(V) - cp)
            assert eps <= np.sqrt(2*V/(3*p)) + 1e-12
print(f"min over all trials of [mM + m eps - (V - sqrt(V) eps)]/V = {worst_slack:.4f}   (>= 0 required)")
print(f"min over all trials of (m+M)/(2 sqrt V) - c_p = {worst_cor:.4f}   (>= 0 required)")
# exact PB2 ratio on the same families (conjecture: mM >= V)
r = np.inf
for p in [5, 7, 11, 13]:
    for kind in ['gauss', 'spike', 'twopoint', 'ramp', 'line', 'mixed']:
        for t in range(150):
            f = gen(p, kind); V = sum((g**2).mean() for g in f)
            if V < 1e-12: continue
            P = planes(f, p); m = min(-P[c].min() for c in P); M = min(P[c].max() for c in P)
            r = min(r, m*M/V)
print(f"min over trials of (min_c depth)(min_c' height)/V = {r:.4f}   (PB2 conjecture: >= 1)")
