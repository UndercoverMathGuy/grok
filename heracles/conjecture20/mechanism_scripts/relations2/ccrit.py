# Critical bonus constant: largest c such that sup_{y,t,th} V/(Gamma_s (sum y^r)^s) = 1 (numerically), tau = 3^{s-1}.
import numpy as np, sys
import sys; sys.path.insert(0,"..")
from relax import search
from progx import exhaustive
s=float(sys.argv[1]); 
for p in [int(a) for a in sys.argv[2:]]:
    F=(p-1)//2; G0,S0,y0=exhaustive(p,s,min(F,5)); tau=3**(s-1)
    lo,hi=0.0,2.0
    for it in range(9):
        c=(lo+hi)/2; v,_=search(p,s,c,tau,G0,restarts=8,maxfe=8000)
        if v>1+1e-4: hi=c
        else: lo=c
    print(f"p={p} s={s} K*={len(S0)}  critical c in [{lo:.3f},{hi:.3f}]   (needed 0.2887)",flush=True)
