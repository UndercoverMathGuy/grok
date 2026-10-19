# HUNT (b): |S| = 1 (feature 1 only) with foreign zero-mean fields. Usage: python hunt_b.py p n b nrest seed [mode]
#   mode struct (default): exact class N_{1}: every neuron misses each foreign frequency j != 1 in at least one slot,
#                          so beta_ij = 0 exactly (sufficient, not necessary, for beta_j = 0).
#   mode pen: fully general neurons, penalty mu * sum_{j != 1} |beta_j|^2 / P^{2s} (allows cancellation across neurons).
#   mode sf : single-feature networks (outputs at frequency 1 only, inputs arbitrary): the single-feature price search.
# Seeds: IF block at 1 (or a cheap own block: 2 neurons / masses (1,1,c)) plus
#   channel neurons (T, j): inputs |2-j| on T, j off T, output j (produce exactly the characters 2*1_T of feature 1),
#   partner neurons (T): inputs 3 on T, 1 off T, output 1 (own-frequency non-circular content),
#   DC-carrying random neurons, and fully random neurons.
# Ratio: gamma / (nu^{1-s} B_p kappa_min); at n = 2 this is W/D with pi = 3^{1-s} (Note Thm q1n2).
import sys, os, json, time, itertools
import numpy as np
import hlib as H, glib as Gb
GL = H.GL

p = int(sys.argv[1]); n = int(sys.argv[2]); bexp = float(sys.argv[3]); nrest = int(sys.argv[4]); seed = int(sys.argv[5])
mode = sys.argv[6] if len(sys.argv) > 6 else 'struct'
nu = n + 1; s = nu / bexp; F = (p - 1) // 2
rng = np.random.default_rng(seed)
kmin = 1 - np.cos(2 * np.pi / p)
D = nu ** (1 - s) * H.Bp(p, n) * kmin
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
tag = f"b_p{p}_n{n}_b{bexp}_{mode}_s{seed}"
foreign = [j for j in range(1, F + 1) if j != 1]
ALL = set(range(0, F + 1))


def red(k):
    k %= p
    return min(k, p - k)


def struct_allowed(base):
    """base: list of n+1 frequency sets; remove every foreign j from one slot unless already missing somewhere"""
    al = [set(A) for A in base]
    for j in foreign:
        if any(j not in A for A in al): continue
        t = int(rng.integers(0, n + 1)); al[t].discard(j)
    return al


def vec(freqset, amp):
    a = np.arange(p); v = np.zeros(p)
    for k in freqset:
        if k == 0: v += amp * rng.standard_normal()
        else: v += amp * np.cos(2 * np.pi * k * a / p + rng.uniform(0, 2 * np.pi))
    return v


def build(it):
    """returns (allowed list, U, W, description)"""
    al = []; Us = []; Ws = []; desc = {}
    kind = ['if', 'cheap2', 'cheapc'][it % 3] if mode != 'sf' else ['if', 'cheap2', 'cheapc'][it % 3]
    # own block
    U0, W0 = GL.if_block(p, n, 1, 1.0, phase0=rng.uniform(0, np.pi))
    if kind == 'if':
        own = list(range(nu)); amps = np.ones(nu)
    elif kind == 'cheap2':
        own = [0, 1]; amps = np.ones(2)
    else:
        c = float(rng.choice([0.3, 0.6, 0.85])); own = list(range(nu)); amps = np.r_[np.ones(nu - 1), c]
        desc['c'] = c
    desc['own'] = kind
    for k, i in enumerate(own):
        a = amps[k] ** (1.0 / nu)
        Us.append(U0[i] * a + 0.05 * rng.standard_normal((n, p))); Ws.append(W0[i] * a + 0.05 * rng.standard_normal(p))
        A_own = [ALL if mode == 'pen' else ({0, 1} if mode == 'sf' else ALL)] * (n + 1)
        if mode == 'sf':
            A_own = [ALL] * n + [{1}]
        al.append(A_own if mode != 'struct' else struct_allowed([ALL] * (n + 1)))
    # foreign / extra neurons
    nf = int(rng.integers(1, 2 * nu + 1)); amp = float(rng.choice([0.05, 0.2, 0.5]))
    kinds = []
    for q in range(nf):
        r = rng.random()
        if mode == 'sf':
            fs = [set(rng.choice(sorted(ALL), size=int(rng.integers(1, 3)), replace=False).tolist()) for _ in range(n)]
            Us.append(np.array([vec(f, amp) for f in fs])); Ws.append(vec({1}, amp)); al.append([ALL] * n + [{1}])
            kinds.append('sf-rand'); continue
        T = [t for t in range(n) if rng.random() < 0.5] or [int(rng.integers(0, n))]
        if r < 0.45 and foreign:
            j = int(rng.choice(foreign))
            fs = [{red(2 - j)} if t in T else {j} for t in range(n)]
            Us.append(np.array([vec(f, amp) for f in fs])); Ws.append(vec({j}, amp)); kinds.append(f'chan{j}T{T}')
            base = [ALL] * n + [ALL]
        elif r < 0.65:
            fs = [{red(3)} if t in T else {1} for t in range(n)]
            Us.append(np.array([vec(f, amp) for f in fs])); Ws.append(vec({1}, amp)); kinds.append(f'partT{T}')
            base = [ALL] * (n + 1)
        elif r < 0.85:
            fs = [set(rng.choice(sorted(ALL), size=2, replace=False).tolist()) | {0} for _ in range(n)]
            Us.append(np.array([vec(f, amp) for f in fs])); Ws.append(vec(set(rng.choice(sorted(ALL), size=2).tolist()), amp))
            kinds.append('dc'); base = [ALL] * (n + 1)
        else:
            Us.append(amp * rng.standard_normal((n, p))); Ws.append(amp * rng.standard_normal(p)); kinds.append('rand')
            base = [ALL] * (n + 1)
        al.append(struct_allowed(base) if mode == 'struct' else base)
    desc['extra'] = kinds
    return al, np.array(Us), np.array(Ws), desc


t0 = time.time(); best = (-np.inf, None); recs = []
pen = foreign if mode == 'pen' else ()
for it in range(nrest):
    al, U, W, desc = build(it)
    g = Gb.GNet(p, n, bexp, al)
    x = g.from_UW(U, W)
    if mode == 'pen':
        x = g.lbfgs(x, taus=(1e-2, 3e-3, 1e-3), iters=1000, pen=pen, mu=10.0 / H.Bp(p, n))
        x = g.lbfgs(x, taus=(3e-4, 1e-4), iters=1500, pen=pen, mu=1e5 / H.Bp(p, n))
    else:
        x = g.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4), iters=1200)
        x, _ = g.slp(x, iters=100)
    ratio = g.gamma(x) / D
    nt = g.net(x); cl = np.abs(nt.clean())
    U2, W2 = nt.U, nt.W
    # output energy by frequency, cheapness of feature 1
    A, Q, a = nt.cheapness()
    oute = [float(np.sum((W2 @ GL.freq_proj(p, k)) ** 2)) for k in range(0, F + 1)]
    rec = dict(it=it, ratio=ratio, beta=[float(v) for v in cl], rel_foreign_beta=float(cl[1:].max() / cl[0]) if F > 1 else 0.0,
               gain1=float(A[0] / a[0]), out_energy=[round(e / sum(oute), 6) for e in oute], m=g.m, **desc)
    recs.append(rec)
    print(json.dumps(rec, default=float), flush=True)
    if ratio > best[0]: best = (ratio, (U2, W2, al))
    if ratio > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_{tag}_{it}.npz"), U=U2, W=W2, p=p, n=n, b=bexp, ratio=ratio)
np.savez(os.path.join(OUT, f"best_{tag}.npz"), U=best[1][0], W=best[1][1], p=p, n=n, b=bexp, ratio=best[0])
rs = [r['ratio'] for r in recs]
nonif = [r['ratio'] for r in recs if max(r['out_energy'][2:] + [0]) > 1e-4 or r['gain1'] > 1 + 1e-4]
print(f"# SUMMARY {tag}: runs={len(rs)} best={max(rs):.10f} best_with_foreign_or_cheap={max(nonif) if nonif else None} "
      f"time={time.time() - t0:.0f}s", flush=True)
