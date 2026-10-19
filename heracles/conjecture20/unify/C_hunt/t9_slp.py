# Does the full-active-set SLP polish reach the IF value from jittered IF seeds where the old pipeline stalled?
import sys, time, numpy as np
import hlib as H, hlib2 as H2
p = int(sys.argv[1]); bexp = float(sys.argv[2]); jit = float(sys.argv[3])
n = 2; nu = 3; s = nu / bexp
R = H2.prog_bb(p, s, maxnodes=20000) if p <= 23 else None
if R is None:
    G, y, bind = H.programme(p, s, restarts=600); supp = [l for l in range(1, (p-1)//2+1) if y[l-1] > 0]
else:
    G, y, supp = R['G'], R['y'], R['supp']
D = nu ** (1 - s) * G
freqs = [l for l in supp for _ in range(nu)]
cc = H.CC(p, n, freqs, s)
rng = np.random.default_rng(1)
x0 = H.if_seed(cc, y)
print(f"p={p} b={bexp} G={G:.10f} supp={supp} IF ratio={cc.gnorm(x0)/D:.12f}", flush=True)
x = H.if_seed(cc, y, jit, rng)
print("jittered", cc.gnorm(x) / D, flush=True)
t = time.time(); x1 = cc.lbfgs(x, taus=(3e-3, 1e-3, 3e-4), iters=600); print("lbfgs", cc.gnorm(x1) / D, f"{time.time()-t:.1f}s", flush=True)
t = time.time(); x2, v2 = H2.mass_lp(cc, x1); print("mass_lp", v2 / D, f"{time.time()-t:.1f}s", flush=True)
t = time.time(); x3, v3 = H2.slp_full(cc, x2, verbose=False); print("slp_full", v3 / D, f"{time.time()-t:.1f}s", flush=True)
t = time.time(); x4, v4 = H2.polish(cc, x3, lbfgs=False); print("polish", v4 / D, f"{time.time()-t:.1f}s", flush=True)
