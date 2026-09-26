"""pax_ceil_big.py -- extend FINAL_RUNS/PAX/ceiling.json to the large primes (no training needed).
gamma_con(K, p) = 3 * M_K(p) / (9 p K)^{3/2}  (closed form verified exactly in pax_ceiling.py)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from kstar import best_subset                      # noqa: E402
OUT = ROOT / "FINAL_RUNS" / "PAX" / "ceiling.json"
C = json.load(open(OUT)); rng = np.random.default_rng(0)
for p in [int(x) for x in sys.argv[1:]] or [509, 1009]:
    if str(p) in C:
        print("have", p); continue
    Ms, Ss = {}, {}
    for K in range(1, 13):
        M, S, how = best_subset(p, K, rng, restarts=60)
        Ms[K] = M; Ss[K] = S
    g = {K: 3 * Ms[K] / (9 * p * K) ** 1.5 for K in Ms}
    Kst = max(g, key=g.get)
    C[str(p)] = dict(nf=(p - 1) // 2, Kstar=Kst, gcon={str(k): v for k, v in g.items()},
                     S={str(k): s for k, s in Ss.items()}, M={str(k): m for k, m in Ms.items()})
    print(f"p {p} nf {(p-1)//2} K*_con {Kst} gamma_con(K*) {g[Kst]:.4e}  "
          f"rungs " + " ".join(f"{K}:{g[K]/g[Kst]:.2f}" for K in range(3, 12)))
json.dump(C, open(OUT, "w"), indent=1)
