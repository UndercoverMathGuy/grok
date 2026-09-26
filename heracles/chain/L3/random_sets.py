# Exact Riesz-with-errors bounds (riesz_bound.bounds, exact rational at rho on a 1/200 grid) for random K-subsets
# of {1..h}, K = round(1.3 ln p), plus K = round(ln p) for comparison. NUMERICAL distribution over sets; each value CERTIFIED.
import sys, math, numpy as np
from riesz_bound import bounds
from fractions import Fraction as Fr
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
p = int(sys.argv[1]); h = (p - 1) // 2
rhos = [Fr(i, 100) for i in range(5, 51)]
for c in (1.0, 1.3):
    K = round(c * math.log(p)); one, two = [], []
    for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 20):
        H = sorted(rng.choice(np.arange(1, h + 1), K, replace=False).tolist())
        b1, b2 = bounds(p, H, rhos)[:2]; one.append(float(b1[0])); two.append(float(b2[0]))
    q = lambda a: f"min {min(a):.3f} med {np.median(a):.3f} max {max(a):.3f}"
    print(f"p={p} c={c} K={K}: one-sided [{q(one)}] frac>=0.3476: {np.mean(np.array(one) >= 0.3476):.2f} | "
          f"two-sided [{q(two)}] frac>=0.3476: {np.mean(np.array(two) >= 0.3476):.2f}", flush=True)
