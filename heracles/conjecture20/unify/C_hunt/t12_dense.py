import numpy as np, hlib2 as H2, time
for p in [11, 13, 17, 23, 31]:
    F = (p - 1) // 2
    for s in [1.02, 1.05, 1.1, 1.2]:
        t = time.time()
        R = H2.prog_bb(p, s, maxnodes=30000, restarts=60) if F <= 11 else H2.prog_best(p, s, restarts=300)
        unif = (F ** (-s)) * (F + 0.5)
        print(f"p={p} s={s}: Gamma={R['G']:.10f} K*={len(R['supp'])} (F={F}) uniform value={unif:.10f} gap={R['gap']} ({time.time()-t:.0f}s)", flush=True)
