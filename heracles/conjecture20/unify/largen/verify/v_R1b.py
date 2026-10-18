# R1 check on MIXED sign patterns: IF blocks (u_i(a) = cos(2 pi a/p + t_i), t_i = pi i/nu, whose sign patterns differ
# across neurons) perturbed in every input coordinate (DC and all frequencies) and in the output phases; keep the
# positive-margin ones and run v_R1.check_alternation (field from the logit table, mpmath).
import sys, os, math, numpy as np, mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v_R1.py')).read().split('print("(a) certified')[0]
ns = {}; exec(src, ns)
rng = np.random.default_rng(31337)
tot = 0
for (p, n, eps) in [(5, 6, 0.006), (5, 7, 0.004), (7, 6, 0.004), (3, 9, 0.004), (11, 4, 0.01), (5, 9, 0.002)]:
    nu = n + 1; a = np.arange(p); t0 = np.pi * np.arange(nu) / nu; got = 0
    for trial in range(400):
        U = np.array([np.cos(2 * np.pi * a / p + t) for t in t0]) + eps * rng.standard_normal((nu, p))
        Wc = np.exp(1j * (n * t0 + eps * rng.standard_normal(nu))) * (1 + eps * rng.standard_normal(nu))
        Uq = [[mp.mpf(float(v)) for v in row] for row in U]
        rho = mp.expj(2 * mp.pi / p)
        W = [[mp.re(mp.mpc(complex(Wc[i])) * rho ** c) for c in range(p)] for i in range(nu)]
        c = ns['margin_cm'](Uq, W, p, n)
        if c > 0:
            pats = len(set(tuple(np.sign(r)) for r in U))
            print(f"  p={p} n={n}: positive margin {mp.nstr(c, 4)}, distinct sign patterns {pats}")
            ns['check_alternation'](Uq, W, p, n, f'perturbed IF p={p} n={n}')
            got += 1; tot += 1
            if got >= 2: break
print('total mixed-sign positive-margin networks checked:', tot)
