# Re-evaluation of saved networks: explicit product network (U, W) -> margin over ALL x in Z_p^n and ALL c (mpmath, 30
# digits) and cost in mpmath; ratio against nu^{1-s} B_p Gamma^{(S)} with Gamma^{(S)} from a FEASIBLE programme point
# (a lower bound on Gamma^{(S)}, so the printed ratio is an UPPER bound on the true one; for |S| <= 8 exact vertices).
# Usage: python verify.py file1.npz [file2.npz ...]   (class-C files carry freqs, x; general files carry U, W)
import sys, os, glob
import numpy as np
import mpmath as mp
import hlib as H
GL = H.GL
mp.mp.dps = 30
files = sys.argv[1:]
for f in files:
    z = np.load(f, allow_pickle=True)
    p = int(z['p']); n = int(z['n']); b = float(z['b']); nu = n + 1; s = nu / b
    if 'freqs' in z.files:
        cc = H.CC(p, n, list(z['freqs']), s); x = z['x']
        U, W = cc.to_net(x); model = cc.gnorm(x)
        S = list(z['S']) if 'S' in z.files else list(range(1, (p - 1) // 2 + 1))
    else:
        U, W = z['U'], z['W']; model = None; S = [1]
    if 'S' in z.files or 'freqs' in z.files:
        Sl = [int(v) for v in S]
        if len(Sl) == 1: Gm = 1 - np.cos(2 * np.pi / p)      # one feature: Gamma^{({l})} = kappa_min (dilation)
        elif len(Sl) <= 8: Gm, _ = H.programme_exact(p, s, Sl)
        else: Gm, _, _ = H.programme(p, s, S=Sl, restarts=80)
    else:
        Gm = 1 - np.cos(2 * np.pi / p)
    nt = GL.Net(U, W, b)
    hf = nt.margin(); cf = nt.cost()
    hm, ct = H.margin_mp(U, W, dps=30)
    cm = mp.fsum([c ** (mp.mpf(b) / 2) for c in ct])
    gm = hm / cm ** mp.mpf(s)
    D = mp.mpf(nu) ** (1 - mp.mpf(s)) * mp.mpf(H.Bp(p, n)) * mp.mpf(Gm)
    print(f"{os.path.basename(f)}: p={p} n={n} b={b} |S|={len(S)} float ratio={hf / cf ** s / float(D):.12f} "
          f"mp ratio={mp.nstr(gm / D, 15)} model={'' if model is None else f'{model * H.Bp(p, n) / float(D):.12f}'} "
          f"stored={float(z['ratio']) if 'ratio' in z.files else ''}", flush=True)
