# check the LP-dual gradient of rho(psi) in jtest2 against finite differences (at a point where the LP solution is unique)
import sys, numpy as np
sys.argv = ['jtest2.py', '7', '2', '1', '0']
src = open('jtest2.py').read().split("t0 = time.time()")[0]
exec(src)
rng2 = np.random.default_rng(5)
for trial in range(3):
    psi = rng2.uniform(0, 6.28, nc)
    v, r, g = lp(psi)
    e = 1e-6; num = np.zeros(nc)
    for k in range(nc):
        pp = psi.copy(); pp[k] += e; pm = psi.copy(); pm[k] -= e
        num[k] = (lp(pp)[0] - lp(pm)[0]) / (2 * e)
    print("rho", v, "max|grad - fd|", np.abs(g - num).max(), "max|fd|", np.abs(num).max())
