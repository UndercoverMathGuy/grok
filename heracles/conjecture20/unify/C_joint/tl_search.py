# Counterexample search in the single-T family TL_n (all n, here b=2 and others):
# maximise R = h/(Gamma_s (sum alpha^r + lam beta^r)^s) over alpha, beta >= 0, phases. R > 1 would beat the IF value.
import sys, numpy as np, time
from cjlib import TL, tl_optimise, gamma_exact, gamma_lp
p = int(sys.argv[1]); n = int(sys.argv[2]); s = float(sys.argv[3]) if len(sys.argv) > 3 else (n+1)/2
starts = int(sys.argv[4]) if len(sys.argv) > 4 else 12
F = (p-1)//2
t0 = time.time()
if p <= 23: Gam, ys = gamma_exact(p, s); y = ys[0]
else: Gam, y = gamma_lp(p, s, restarts=40)
S = list(range(1, F+1)); tl = TL(p, n, s, S)
rng = np.random.default_rng(p*1000+n)
tau = 1/(1-(1-(n/(n+1))**(s-1)))
print(f"p={p} n={n} s={s:.3f} Gamma={Gam:.6f} K*={np.sum(y>0)} tau_n={tau:.4f} c_n={1-1/tau:.4f}", flush=True)
best = (0, None)
for k in range(starts):
    if k == 0:   # start at the IF optimum with tiny interference
        al = np.maximum(y, 1e-3*y.max()); be = 1e-2*al; ph = rng.uniform(-np.pi, np.pi, F)
    elif k % 3 == 1:  # IF optimum + substantial interference on the support
        al = np.maximum(y, 1e-3*y.max()); be = rng.uniform(0.2, 1.0, F)*al; ph = rng.uniform(-np.pi, np.pi, F)
    elif k % 3 == 2:  # mostly single-T features
        al = np.maximum(y, 1e-3*y.max())*rng.uniform(0, 0.3, F) + 1e-4; be = np.maximum(y, 1e-2*y.max())*rng.uniform(0.5, 1.5, F); ph = rng.uniform(-np.pi, np.pi, F)
    else:  # random support
        al = rng.exponential(1, F)*(rng.random(F) < 0.5) + 1e-4; be = rng.exponential(1, F)*(rng.random(F) < 0.5) + 1e-4; ph = rng.uniform(-np.pi, np.pi, F)
    x0 = np.concatenate([np.log(al), np.log(be), ph])
    R, (a1, b1, p1) = tl_optimise(tl, Gam, x0)
    tag = ''
    if R > best[0]: best = (R, (a1, b1, p1)); tag = ' *'
    print(f"  start {k}: R={R:.6f}  #alpha>1e-3max={np.sum(a1>1e-3*max(a1.max(),b1.max()))} #beta>1e-3max={np.sum(b1>1e-3*max(a1.max(),b1.max()))}{tag}", flush=True)
R, (a1, b1, p1) = best
print(f"BEST R={R:.6f}  time {time.time()-t0:.0f}s")
np.savez(f"tl_best_p{p}_n{n}_s{s:.3f}.npz", alpha=a1, beta=b1, phi=p1, R=R, Gamma=Gam, p=p, n=n, s=s)
m = max(a1.max(), b1.max())
for l in range(F):
    if a1[l] > 1e-3*m or b1[l] > 1e-3*m: print(f"   l={l+1}: alpha={a1[l]/m:.4f} beta={b1[l]/m:.4f} phi={p1[l]%(2*np.pi):.4f}")
