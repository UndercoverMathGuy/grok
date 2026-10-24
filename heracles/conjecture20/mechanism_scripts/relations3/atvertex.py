# At the programme vertex y*, v = tau y*: certified (1/2)osc D_delta / (K v)_delta at the best binding delta, for
# Parseval, log-tilt and tilted-mean; relation split at the tilted-mean optimum.  Needed: >= c (0.2887 Thm A, 0.1835 P12).
import numpy as np, sys
sys.path.insert(0,'.')
from core import *
D=load()
print(" p    K  K/lnp  gamma  maxSk  | Pars   logT   TiltM | rho0   rho1  (at TiltM opt, best delta)")
for p in sorted(D):
    S=np.array(D[p]['S']); y=np.array(D[p]['y']); B,cov=binding(p,S,y); C=cosmat(p,S)
    gam=cov.min()/(y.sum()*p/(p-1))
    best=None
    for d in B:
        s=np.abs(np.sin(np.pi*S*d/p)); b=y*s; pay=(b*s).sum()
        par=np.sqrt((b**2).sum()/2)/pay; tm,lam=tilted_mean_bound(b,C); lt,_=log_tilt_bound(b,C)
        r0,r1,E=relation_split(lam*b,C,p)
        rec=(max(par,tm/pay),par,lt/pay,tm/pay,r0,r1,(2*s**2).sum())
        if best is None or rec[0]>best[0]: best=rec
    _,par,lt,tm,r0,r1,sk=best
    print(f"{p:5d} {len(S):3d} {len(S)/np.log(p):5.2f} {gam:5.3f} {sk:6.2f} | {par:5.3f} {lt:6.3f} {tm:6.3f} | {r0:6.3f} {r1:6.3f}")
