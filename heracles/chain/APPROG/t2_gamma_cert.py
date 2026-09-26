# B&B upper certificate for Gamma*(p): W_unfloored(Gam_cand * (1 + 1e-7)) >= 1  =>  Gamma* <= Gam_cand (1 + 1e-7) (fp64 LP B&B).
import sys, time, numpy as np
from approg_lib import *
p = int(sys.argv[1]); G = float(sys.argv[2])
t = time.time()
st_up, n_up = decide(p, [], 1.0, 0.0, G * (1 + 1e-7), max_nodes=int(sys.argv[3]) if len(sys.argv) > 3 else 200000)
st_lo, n_lo = decide(p, [], 1.0, 0.0, G * (1 - 1e-7))
print(f"p={p}: at Gam*(1+1e-7): {st_up} ({n_up} nodes);  at Gam*(1-1e-7): {st_lo}  [{time.time() - t:.0f}s]", flush=True)
