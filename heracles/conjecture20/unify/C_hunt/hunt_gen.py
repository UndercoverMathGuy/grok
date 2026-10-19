# HUNT (a''): general (non-class-C) networks at n = 2: mixed modes, shared neurons, DC.  Usage: python hunt_gen.py p b nrest seed
# Seeds: the IF optimum (balanced blocks) with
#   'merge'  : pairs of neurons from DIFFERENT features merged into one neuron (u = u_a + u_b slot-wise, w = w_a + w_b;
#              cost-neutral at b = 2 for orthogonal frequencies), pairs chosen preferentially with an F2 relation
#              2l = +-j +-q between the features (the free-mixing channels);
#   'dc'     : DC added to inputs of some neurons; 'mixjit': all-frequency noise on every neuron; 'cheap': one feature at width 2.
# All coordinates free (all frequencies, DC); smooth-min L-BFGS + SLP.  Ratio = gamma / (3^{1-s} B_p Gamma_s).
import sys, os, json, time, itertools
import numpy as np
import hlib as H, glib as Gb
GL = H.GL

p = int(sys.argv[1]); bexp = float(sys.argv[2]); nrest = int(sys.argv[3]); seed = int(sys.argv[4])
n = 2; nu = 3; s = nu / bexp; F = (p - 1) // 2
rng = np.random.default_rng(seed)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
G, y, bind = H.programme(p, s, restarts=120)
Sst = [l for l in range(1, F + 1) if y[l - 1] > 0]
D = nu ** (1 - s) * H.Bp(p, n) * G
ALL = set(range(0, F + 1))


def red(k):
    k %= p
    return min(k, p - k)


rel = [(l, j) for l in Sst for j in Sst if j != l and any(red(2 * l) == red(j + e * q) for q in Sst for e in (1, -1))]
print(f"# gen p={p} b={bexp}: Gamma={G:.10f} S*={Sst} F2-related pairs={rel[:8]}", flush=True)
t0 = time.time(); recs = []; best = -np.inf
for it in range(nrest):
    kind = ['merge', 'merge', 'dc', 'mixjit', 'cheap'][it % 5]
    net = GL.if_network(p, n, y, bexp); U = net.U.copy(); W = net.W.copy()
    fr = [l for l in Sst for _ in range(nu)]
    if kind == 'merge':
        k = int(rng.integers(1, len(Sst) + 1)); used = set(); newU = []; newW = []
        for _ in range(k):
            if rel and rng.random() < 0.7:
                la, lb = rel[int(rng.integers(len(rel)))]
            else:
                la, lb = rng.choice(Sst, 2, replace=False) if len(Sst) > 1 else (Sst[0], Sst[0])
            ia = [i for i in range(len(fr)) if fr[i] == la and i not in used]
            ib = [i for i in range(len(fr)) if fr[i] == lb and i not in used]
            if not ia or not ib: continue
            a_, b_ = int(rng.choice(ia)), int(rng.choice(ib)); used |= {a_, b_}
            newU.append(U[a_] + U[b_]); newW.append(W[a_] + W[b_])
        keep = [i for i in range(len(fr)) if i not in used]
        U = np.concatenate([U[keep]] + ([np.array(newU)] if newU else [])); W = np.concatenate([W[keep]] + ([np.array(newW)] if newW else []))
    elif kind == 'dc':
        for i in rng.choice(len(U), max(1, len(U) // 3), replace=False):
            U[i] += rng.normal(0, 0.3, (n, 1)) * np.abs(U[i]).max()
    elif kind == 'mixjit':
        U = U + 0.15 * np.abs(U).max() * rng.standard_normal(U.shape); W = W + 0.15 * np.abs(W).max() * rng.standard_normal(W.shape)
    else:
        l0 = int(rng.choice(Sst)); idx = [i for i in range(len(fr)) if fr[i] == l0]
        drop = idx[-1]; keep = [i for i in range(len(fr)) if i != drop]
        U = U[keep]; W = W[keep]
    U = U + 0.01 * rng.standard_normal(U.shape); W = W + 0.01 * rng.standard_normal(W.shape)
    g = Gb.GNet(p, n, bexp, [[ALL] * 3 for _ in range(len(U))])
    x = g.from_UW(U, W); r0 = g.gamma(x) / D
    x = g.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4), iters=1500)
    x, v = g.slp(x, iters=100)
    nt = g.net(x); cl = np.abs(nt.clean()); feats = [l for l in range(1, F + 1) if cl[l - 1] > 1e-6 * cl.max()]
    sp = GL.split(nt); rs = sp.margin() / sp.cost() ** s / D if sp.margin() > -1e9 else None
    dc = float(np.sum(nt.U.mean(2) ** 2) * p / np.sum(nt.U ** 2))
    rec = dict(it=it, kind=kind, start=r0, ratio=v / D, feats=feats, count=len(feats), Kstar=len(Sst), split_ratio=rs, dc_share=dc, m=len(U))
    recs.append(rec); best = max(best, v / D)
    print(json.dumps(rec, default=float), flush=True)
    if v / D > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_gen_p{p}_b{bexp}_{it}.npz"), U=nt.U, W=nt.W, p=p, n=n, b=bexp, S=list(range(1, F + 1)))
near = [z for z in recs if z['ratio'] > 1 - 1e-6]
nonsplit = [z['ratio'] for z in recs if z['split_ratio'] is not None and z['ratio'] - z['split_ratio'] > 1e-4]
print(f"# SUMMARY gen p={p} b={bexp}: runs={len(recs)} best={best:.10f} near-opt counts={sorted(set(z['count'] for z in near))} "
      f"K*={len(Sst)} best mixing-reliant (ratio > split + 1e-4)={max(nonsplit) if nonsplit else None} time={time.time() - t0:.0f}s", flush=True)
