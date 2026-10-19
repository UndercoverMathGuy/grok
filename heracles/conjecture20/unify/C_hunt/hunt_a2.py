# HUNT (a), resumed: compensation at n = 2 in class C, with the exact programme (B&B) and the full-active-set polish.
# Usage: python hunt_a2.py p b Smode nrest seed [families]
#   Smode: F | Sstar | AP<K> (S = {1..K}, a relation-rich class) | REL (S* plus every q = |2l - j| for l, j in S*:
#          relations 2l = j + q inside S) | sub<d> (multiplicative subgroup of order d)
# Families (comma list; default all):
#   if_jit   IF optimum of the class + jitter (baseline: must return to ratio 1)
#   single   a random nonempty subset of features made SINGLE NEURONS (gain 3^{s-1}: the most L2-efficient cheap feature,
#            ptest2.py), the rest IF blocks; PHASE DESIGN with masses frozen (prescribed gains), then full release
#   removed  IF blocks with one atom removed (2 atoms at cube-root phases: fluctuation on all three characters, gain (3/2)^{s-1})
#   punct    punctured 2-atom blocks (label-only fluctuation, gain (3/2)^{s-1}; Lemma L2 applies: consistent family)
#   mixed    per feature a random choice among IF / single / removed / punct / (1,1,c) triangle; phase design, release
#   ghost    IF optimum plus zero-mean "ghost" pairs (b, -b) at frequencies of S outside the support (pure compensators)
#   extra    support plus 1-3 extra small features of S (compensators that carry clean mass)
# Per restart: seed -> [phase design with masses frozen: 'designed' ratio, prescribed gains] -> release (L-BFGS, mass LP,
#   full SLP) -> final ratio vs D_S = 3^{1-s} B_p Gamma^{(S)}; for ratio > 0.999 also vs D_T, T = the final feature set
#   (exact B&B Gamma^{(T)}).  Networks with ratio_T > 1 + 1e-9 are saved as VIOL_*.npz.
import sys, os, json, time
import numpy as np
import hlib as H, hlib2 as H2

p = int(sys.argv[1]); bexp = float(sys.argv[2]); Smode = sys.argv[3]; nrest = int(sys.argv[4]); seed = int(sys.argv[5])
fams = sys.argv[6].split(',') if len(sys.argv) > 6 else ['if_jit', 'cheapc', 'single', 'cheapc', 'removed', 'punct', 'cheapc', 'mixed', 'ghost', 'extra']
n = 2; nu = 3; s = nu / bexp; r = 1 / s; F = (p - 1) // 2
rng = np.random.default_rng(seed)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out2')
tag = f"a2_p{p}_b{bexp}_{Smode}_s{seed}"


def red(l):
    l %= p
    return min(l, p - l)


def prog(S, nodes=4000):
    return H2.prog_bb(p, s, S=S, maxnodes=nodes) if len(S) <= 10 else H2.prog_best(p, s, S=S, restarts=200)


R0 = prog(list(range(1, F + 1)))
Sst0 = R0['supp']
if Smode == 'F': S = list(range(1, F + 1))
elif Smode == 'Sstar': S = Sst0
elif Smode.startswith('AP'): S = list(range(1, min(F, int(Smode[2:])) + 1))
elif Smode == 'REL':
    S = sorted(set(Sst0) | {red(2 * l - j) for l in Sst0 for j in Sst0 if red(2 * l - j) != 0} | {red(2 * l + j) for l in Sst0 for j in Sst0 if red(2 * l + j) != 0})
elif Smode.startswith('sub'):
    d = int(Smode[3:]); assert (p - 1) % d == 0 and d % 2 == 0
    g = next(g for g in range(2, p) if len({pow(g, k, p) for k in range(p - 1)}) == p - 1)
    h = pow(g, (p - 1) // d, p); S = sorted({red(pow(h, k, p)) for k in range(d)})
else: raise SystemExit('bad Smode')
RS = R0 if S == list(range(1, F + 1)) else prog(S)
GS, yS, Sst = RS['G'], RS['y'], RS['supp']
D = nu ** (1 - s) * GS
print(f"# {tag}: S={S} Gamma^S={GS:.12f} (B&B gap {RS['gap']}) supp={Sst} K*_S={len(Sst)} bind={RS['bind']} | "
      f"Gamma(F)={R0['G']:.12f} K*={len(Sst0)} supp*={Sst0}", flush=True)
CACHE = {}


def gamma_T(T):
    T = tuple(sorted(T))
    if T not in CACHE: CACHE[T] = prog(list(T), nodes=100000)
    return CACHE[T]


# ---------------------------------------------------------------------------------------------- feature atom systems
def blk_if(A, k0=0):           # nu atoms, cube-root phases: no fluctuation
    return [(A / nu, [2 * np.pi * k / nu + k0, 2 * np.pi * k / nu + k0]) for k in range(nu)]


def blk_single(A):              # one neuron: gain 3^{s-1}, fluctuation |a_T| = A on all three characters
    return [(A, list(rng.uniform(0, 2 * np.pi, 2)))]


def blk_removed(A):             # IF block minus one atom, masses renormalised: all three characters, |a_T| = A/2
    k0 = rng.uniform(0, 2 * np.pi)
    return [(A / 2, [2 * np.pi * k / 3 + k0, 2 * np.pi * k / 3 + k0]) for k in range(2)]


def blk_punct(A):               # 2 atoms xi = (1,1), (-1,-1): label-only fluctuation a_{12} = A
    k0 = rng.uniform(0, 2 * np.pi)
    return [(A / 2, [k0, -k0]), (A / 2, [k0 + np.pi, -k0 + np.pi])]


def blk_cheapc(A, c):         # IF phases, masses (1, 1, c): gain G(c) = 3^{s-1}(2+c)/(2+c^r)^s, phases perturbed
    m = np.array([1.0, 1.0, c]) / (2 + c); k0 = rng.uniform(0, 2 * np.pi); jt = float(rng.choice([0.05, 0.3, 1.0]))
    return [(A * m[k], list(2 * np.pi * k / nu + k0 + jt * rng.standard_normal(2))) for k in range(nu)]


def blk_tri(A):                 # (1,1,c) triangle: single-character fluctuation of size (1-c) A
    c = float(rng.choice([0.2, 0.5, 0.8])); beta = np.arccos(c / 2)
    x2 = np.array([np.exp(1j * beta), np.exp(-1j * beta), -1.0]); ph = rng.uniform(0, 2 * np.pi)
    x1 = np.exp(1j * ph) * (np.conj(x2) ** 2 if rng.random() < 0.5 else x2)
    m = np.array([1, 1, c]) / (2 + c)
    return [(A * m[i], [np.angle(x1[i]), np.angle(x2[i])]) for i in range(3)]


def assemble(blocks):
    freqs = []; b = []; ph = []
    for l, atoms in blocks:
        for (bb, phi) in atoms:
            freqs.append(l); b.append(bb); ph.append(phi)
    cc = H.CC(p, n, freqs, s)
    return cc, cc.pack(np.array(b, complex), np.array(ph))


def summarize(cc, x):
    fd = cc.feature_data(x)
    Amax = max(d['A'] for d in fd.values())
    feats = [l for l in fd if fd[l]['A'] > 1e-7 * Amax]
    b, _ = cc.unpack(x)
    Qt = sum(fd[l]['Q'] for l in fd)
    sig = [l for l in feats if fd[l]['Q'] / Qt > 1e-4]
    ghost = [l for l in fd if fd[l]['A'] <= 1e-7 * Amax and fd[l]['L'] > 1e-6 * np.abs(b).sum()]
    return dict(feats=feats, count=len(feats), count_sig=len(sig), gains={l: round(fd[l]['gain'], 6) for l in feats},
                widths={l: fd[l]['width'] for l in feats}, shares={l: round(fd[l]['Q'] / Qt, 5) for l in feats}, ghost=ghost,
                maxgain_sig=max([fd[l]['gain'] for l in sig] or [0]))


recs = []; best = (-np.inf, None, None); t0 = time.time()
for it in range(nrest):
    fam = fams[it % len(fams)]
    rec = dict(fam=fam, it=it)
    A = {l: yS[l - 1] for l in Sst}
    design = fam in ('single', 'removed', 'punct', 'mixed', 'cheapc')
    if fam == 'if_jit':
        blocks = [(l, blk_if(A[l])) for l in Sst]
        cc, x = assemble(blocks)
        jit = [0.05, 0.2, 0.5][(it // len(fams)) % 3]
        x = x + jit * rng.standard_normal(x.size) * np.r_[np.full(2 * cc.m, np.abs(cc.unpack(x)[0]).mean()), np.ones(cc.m * n)]
        rec['jit'] = jit
    elif design:
        k = int(rng.integers(1, len(Sst) + 1)) if fam not in ('mixed', 'cheapc') else len(Sst)
        if fam == 'cheapc':
            c = [0.95, 0.85, 0.7, 0.5, 0.3][(it // len(fams)) % 5]; rec['c'] = c
            if rng.random() < 0.5: k = int(rng.integers(1, len(Sst) + 1))
        chosen = set(rng.choice(Sst, size=k, replace=False).tolist())
        blocks = []; kinds = {}
        for l in Sst:
            if l not in chosen: blocks.append((l, blk_if(A[l], rng.uniform(0, 6.3)))); kinds[l] = 'if'; continue
            kind = fam if fam != 'mixed' else str(rng.choice(['if', 'single', 'removed', 'punct', 'tri']))
            kinds[l] = kind
            if kind == 'cheapc': blocks.append((l, blk_cheapc(A[l], c))); continue
            blocks.append((l, {'if': lambda a: blk_if(a, rng.uniform(0, 6.3)), 'single': blk_single, 'removed': blk_removed,
                               'punct': blk_punct, 'tri': blk_tri}[kind](A[l])))
        cc, x = assemble(blocks)
        rec['kinds'] = kinds
    elif fam == 'ghost':
        out = [l for l in S if l not in Sst]
        if not out: out = [l for l in range(1, F + 1) if l not in Sst]   # ghosts are allowed anywhere (A = 0)
        ng = int(rng.integers(1, min(4, len(out)) + 1)); gl = rng.choice(out, size=ng, replace=False).tolist()
        blocks = [(l, blk_if(A[l], rng.uniform(0, 6.3))) for l in Sst]
        amp = float(rng.choice([0.05, 0.15, 0.4])) * max(A.values())
        for l in gl:
            ph = list(rng.uniform(0, 2 * np.pi, 2))
            blocks.append((l, [(amp, ph), (-amp, [ph[0] + rng.uniform(0.3, 3), ph[1] + rng.uniform(0.3, 3)])]))
        cc, x = assemble(blocks); rec['ghosts'] = gl
    else:  # extra
        out = [l for l in S if l not in Sst]
        if not out: out = list(Sst)
        ne = int(rng.integers(1, min(3, len(out)) + 1)); ext = rng.choice(out, size=ne, replace=False).tolist()
        blocks = [(l, blk_if(A[l], rng.uniform(0, 6.3))) for l in Sst]
        for l in ext:
            a = float(rng.choice([0.05, 0.15, 0.3])) * max(A.values())
            blocks.append((l, [blk_single, blk_removed, blk_punct][int(rng.integers(3))](a)))
        cc, x = assemble(blocks); rec['ext'] = ext
    rec['m'] = cc.m
    if design:
        free = np.r_[np.zeros(2 * cc.m, bool), np.ones(cc.m * n, bool)]
        xd = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=800, free=free)
        xd, vd = H2.slp_full(cc, xd, iters=200, free=free)
        fd = cc.feature_data(xd)
        rec['designed'] = vd / D
        rec['designed_gains'] = {l: round(fd[l]['gain'], 5) for l in fd}
        x = xd
    x, v = H2.polish(cc, x)
    ratio = v / D
    rec['ratio'] = ratio
    rec.update(summarize(cc, x))
    if ratio > 0.999:
        RT = gamma_T(rec['feats'])
        rec['ratio_T'] = ratio * GS / RT['G']; rec['GammaT_gap'] = RT['gap']
    if ratio > best[0]:
        best = (ratio, cc.freqs.copy(), x.copy())
        np.savez(os.path.join(OUT, f"best_{tag}.npz"), freqs=best[1], x=best[2], p=p, n=n, b=bexp, S=S, ratio=best[0])
    if rec.get('ratio_T', 0) > 1 + 1e-9 or ratio > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_{tag}_{it}.npz"), freqs=cc.freqs, x=x, p=p, n=n, b=bexp, S=S, ratio=ratio)
    rec['t'] = round(time.time() - t0)
    recs.append(rec)
    print(json.dumps(rec, default=float), flush=True)

np.savez(os.path.join(OUT, f"best_{tag}.npz"), freqs=best[1], x=best[2], p=p, n=n, b=bexp, S=S, ratio=best[0])
rs = [z['ratio'] for z in recs]
near = [z for z in recs if z['ratio'] > 1 - 1e-6]
des = [z['designed'] for z in recs if 'designed' in z]
cheap = [z['ratio'] for z in recs if z['maxgain_sig'] > 1 + 1e-3]
print(f"# SUMMARY {tag}: runs={len(rs)} best={max(rs):.12f} best_ratio_T={max([z.get('ratio_T', 0) for z in recs]):.12f} "
      f"K*_S={len(Sst)} near-opt counts={sorted(set(z['count'] for z in near))} near-opt feature sets={sorted(set(tuple(z['feats']) for z in near))} "
      f"best designed (prescribed cheap)={max(des) if des else None} best final with a significant cheap feature={max(cheap) if cheap else None} "
      f"time={time.time() - t0:.0f}s", flush=True)
