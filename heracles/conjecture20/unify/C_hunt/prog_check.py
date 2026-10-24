# Reliability of the programme solver: default (60 restarts, seed 0, as in the hunts) vs 600 restarts (3 seeds).
import numpy as np, hlib as H
for p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]:
    for b in [2.0, 2.5]:
        s = 3 / b; F = (p - 1) // 2
        G0, y0, B0 = H.programme(p, s, restarts=60, seed=0)
        best = (G0, y0)
        for sd in (1, 2, 3):
            G1, y1, _ = H.programme(p, s, restarts=200, seed=sd)
            if G1 > best[0] + 1e-12: best = (G1, y1)
        S0 = [l for l in range(1, F + 1) if y0[l - 1] > 0]; S1 = [l for l in range(1, F + 1) if best[1][l - 1] > 0]
        print(f"p={p} b={b}: Gamma(60)={G0:.10f} K={len(S0)} S={S0} | Gamma(600)={best[0]:.10f} K={len(S1)} S={S1} | rel gap={(best[0]-G0)/best[0]:.2e}", flush=True)
