# True single-pattern depth of the pair-offset body N_D (independent densities at +D and -D), per offset D, for a set T,
# vs the flip alone (A_s, D-independent) and the proven independent-density bound min_k c1 sqrt(2/cov_k(D)) with c1 = A*.
import sys, json, numpy as np
from lever import pair_depth
p = int(sys.argv[1]); T = [int(x) for x in sys.argv[2].split(',')]; As = float(sys.argv[3]); Ast = float(sys.argv[4])
rng = np.random.default_rng(3)
h = (p - 1) // 2
for D in range(1, h + 1):
    cov = 1 - np.cos(2 * np.pi * np.array(T) * D / p)
    pdD = pair_depth(p, T, D, rng, starts=6, iters=120)
    print(json.dumps(dict(D=D, pair_depth=round(pdD, 4), flip=As, indep_bound=round(float((Ast * np.sqrt(2 / cov)).min()), 4),
                          min_cov=round(float(cov.min()), 3), max_cov=round(float(cov.max()), 3))), flush=True)
