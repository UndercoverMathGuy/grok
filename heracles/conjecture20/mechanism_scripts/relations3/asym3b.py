# Asymptotic LD model (p->oo, random-like optimised supports, uniform weights, K=kap ln p):
# efficiency e(kap), Theorem-1 requirement req_c = c-(1/e-1)/tau, tilt certificate (inf over profiles) and LD truth.
import numpy as np, sys
sys.path.insert(0,'.')
from asym import law, ldtruth
from asym2 import setup, val
from scipy.optimize import minimize
TAU=3**0.5; CA=1/(2*3**0.5); C12=1-(2/3)**0.5
kaps=[3.2,4.0,4.5,7.0,8.0,9.0,17.0,18.0]
vals={}
for k in kaps:
    gam,W,Sb=setup(k); vals[k]=gam/np.sqrt(k)
gmax=max(vals.values())
print(" kap   gamma   e      reqA   req12 | tilt_inf  tilt(b~s) | LDtruth(b~s)")
for k in kaps:
    gam,W,Sb=setup(k); e=vals[k]/gmax
    best=9; rng=np.random.default_rng(1)
    for st in range(8):
        b0=Sb**rng.uniform(-1,3)*np.exp(rng.normal(0,0.5,len(Sb)))
        r=minimize(lambda b: val(b,W,Sb,k),b0,method='Nelder-Mead',options={'maxiter':4000})
        best=min(best,r.fun)
    print(f"{k:5.1f} {gam:6.3f} {e:6.3f}  {CA-(1/e-1)/TAU:6.3f} {C12-(1/e-1)/TAU:6.3f} |  {best:6.3f}   {val(Sb,W,Sb,k):6.3f}  |  {ldtruth(k):6.3f}",flush=True)
