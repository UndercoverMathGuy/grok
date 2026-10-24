# Is the programme optimum dense (K* = F, uniform y) near the convex end?  Exact B&B for F <= 10, reweighted LP + local
# support search otherwise; compare with the uniform allocation value F^{-s}(F + 1/2).
import numpy as np, hlib2 as H2, time
for p in [11, 13, 17, 23, 31, 41, 61]:
    F = (p - 1) // 2
    for s in [1.02, 1.05, 1.1]:
        t = time.time()
        R = H2.prog_bb(p, s, maxnodes=20000, restarts=40) if F <= 10 else H2.prog_bb(p, s, maxnodes=100, restarts=150)
        unif = (F ** (-s)) * (F + 0.5)
        print(f"p={p} s={s}: Gamma(best found)={R['G']:.10f} K*={len(R['supp'])} (F={F}) uniform={unif:.10f} uniform optimal: {abs(R['G']-unif) < 1e-9 * unif} B&B gap={R['gap']:.1e} ({time.time()-t:.0f}s)", flush=True)
