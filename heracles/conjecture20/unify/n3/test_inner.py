import sys
sys.argv = ['cert_pt.py', 'pt', '1.8552', '0.84']
from fractions import Fraction as Fr
import numpy as np
import cert_pt as cp
from ptinner import F
r = 1/1.8552
rng = np.random.default_rng(2)
worst = -1
for t in range(60):
    P = rng.uniform(0.01, 3.0); T = -rng.uniform(0, 4)*P**1.5
    Pf, Tf = Fr(P), Fr(T)
    lo = float(cp.inner_low(Pf, Tf))
    fl = F(P, T, r)
    worst = max(worst, lo - fl)
    if t < 6: print(f"P={P:.4f} T={T:.4f}: rigorous low {lo:.6f}  float closed-form {fl:.6f}")
print("max(rigorous_low - float) =", worst, "(should be <= ~1e-10)")
