# Single-T family search with a self-consistent programme reference:
#  Gamma_best = best programme value found by (reweighted LP, 60 restarts) and by the same optimiser with beta=0.
#  Then search with interference. Report R relative to Gamma_best (exact Gamma for p<=23).
import sys, numpy as np, time
from cjlib import TL, tl_optimise, gamma_exact, gamma_lp
p = int(sys.argv[1]); n = int(sys.argv[2]); s = float(sys.argv[3]); starts = int(sys.argv[4])
F = (p-1)//2; t0 = time.time(); rng = np.random.default_rng(7*p+n)
tl = TL(p, n, s, list(range(1, F+1)))
if p <= 23:
    Gam, ys = gamma_exact(p, s); y = ys[0]; src = 'exact'
else:
    Gam, y = gamma_lp(p, s, restarts=60); src = 'lp'
    for k in range(2*starts):
        al = rng.exponential(1, F)*(rng.random(F) < rng.uniform(0.1, 0.6)) + 1e-6
        x0 = np.concatenate([np.log(al), np.zeros(F), rng.uniform(-np.pi, np.pi, F)])
        R, (a1, b1, p1) = tl_optimise(tl, Gam, x0, ifonly=True)
        if R > 1: Gam *= R; y = a1/(np.sum(a1**(1/s))**s); src = 'opt'
tau = (1+1/n)**(s-1)
print(f"p={p} n={n} s={s:.3f} Gamma_ref={Gam:.6f} ({src}) K*={np.sum(y>1e-6*y.max())} tau_n={tau:.4f} c_n={1-1/tau:.4f}", flush=True)
best = (0, None); Rs = []
for k in range(starts):
    yy = np.maximum(y, 1e-3*y.max())
    if k % 3 == 0: al = yy; be = rng.uniform(0.05, 1.0, F)*yy
    elif k % 3 == 1: al = yy*rng.uniform(0, 0.3, F) + 1e-6; be = yy*rng.uniform(0.5, 1.5, F) + rng.exponential(0.3*y.max(), F)*(rng.random(F) < 0.3)
    else: al = rng.exponential(1, F)*(rng.random(F) < 0.4) + 1e-6; be = rng.exponential(1, F)*(rng.random(F) < 0.4) + 1e-6
    x0 = np.concatenate([np.log(al), np.log(be), rng.uniform(-np.pi, np.pi, F)])
    R, (a1, b1, p1) = tl_optimise(tl, Gam, x0); Rs.append(R)
    if R > best[0]: best = (R, (a1, b1, p1))
R, (a1, b1, p1) = best
m = max(a1.max(), b1.max())
print(f"  BEST R={R:.6f} (starts: median {np.median(Rs):.4f}) time {time.time()-t0:.0f}s; beta-mass share {b1.sum()/(a1.sum()+b1.sum()):.4f}")
np.savez(f"tl2_p{p}_n{n}_s{s:.3f}.npz", alpha=a1, beta=b1, phi=p1, R=R, Gamma=Gam, p=p, n=n, s=s)
for l in range(F):
    if a1[l] > 1e-3*m or b1[l] > 1e-3*m: print(f"     l={l+1}: alpha={a1[l]/m:.4f} beta={b1[l]/m:.4f} phi={p1[l]%(2*np.pi):.4f}")
