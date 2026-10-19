# W8 at larger n (b = 2 is the worst case of the bound): aligned class-C single features near and far from IF.
import numpy as np
from elib import *
from c8_aligned import build, analyse
rng = np.random.default_rng(11)
for (p, n) in [(5, 5), (5, 6), (7, 4), (5, 7)]:
    nu = n + 1; s = nu / 2; worst = -np.inf; cnt = 0; best = 0
    for trial in range(12):
        m = nu + rng.integers(0, 3)
        base = np.array([np.pi * (l % nu) / nu for l in range(m)])
        ph = np.repeat(base[:, None], n, axis=1) + rng.normal(scale=[0.005, 0.02, 0.05][trial % 3], size=(m, n))
        amp = np.where(np.arange(m) < nu, 1.0, 0.25) * np.exp(rng.normal(scale=[0.0, 0.01, 0.03][trial % 3], size=m))
        net = Net(*build(p, n, amp, ph))
        if net.margin() <= 0: continue
        ratio, bound, coneL2, c, A, L, kp = analyse(net, p, n, s)
        x = 1 - c / A
        worst = max(worst, ratio - bound, ratio - (1 - x) * (1 + 2 * kp * x * x / nu) ** (s - 1), ratio - 1, -coneL2)
        best = max(best, ratio); cnt += 1
    print(f"(p,n)=({p},{n}) b=2: {cnt} positive-margin aligned networks; best gamma/gIF={best:.4f}; max violation of (i),(iii) = {worst:.2e} (<= 0)")
