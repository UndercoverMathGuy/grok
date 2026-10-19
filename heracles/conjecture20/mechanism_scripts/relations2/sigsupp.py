import numpy as np, sys
import sys; sys.path.insert(0,"../cu")
from sigma import sigma, inner
from progx import exhaustive
s=float(sys.argv[1])
for p in [int(a) for a in sys.argv[2:]]:
    F=(p-1)//2; G0,S0,y0=exhaustive(p,s,min(F,5))
    v,th=sigma(p,S0,restarts=8)
    val,b=inner(p,S0,th)
    print(f"p={p} supp y*={S0} plain sigma(supp) <= {v:.4f} b={np.round(b,3)}",flush=True)
