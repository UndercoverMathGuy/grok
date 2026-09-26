# Best "killing measure": probability mu on T^2 with E(1+s)(1+t)=0 minimising
#   kappa(mu) = sup_{|alpha|=|beta|=1} |E_mu (1+alpha s)(1+beta t)|.
# If kappa* <= 3/2, then m e1 <= 3 e2 (R1) follows for every population (rotate atom a to v_a=w_a=1).
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(0)
g=np.exp(2j*np.pi*np.arange(360)/360); AL,BE=np.meshgrid(g,g,indexing='ij')
def kappa(ms,mt,mst): return np.abs(1+AL*ms+BE*mt+AL*BE*mst).max()
def unpack(z,K):
    wts=np.exp(z[:K]); wts/=wts.sum(); s=np.exp(1j*z[K:2*K]); t=np.exp(1j*z[2*K:3*K]); return wts,s,t
def obj(z,K):
    wts,s,t=unpack(z,K); ms=(wts*s).sum(); mt=(wts*t).sum(); mst=(wts*s*t).sum()
    kill=abs(1+ms+mt+mst)
    return kappa(ms,mt,mst)+50*kill
best=np.inf
for trial in range(60):
    K=int(rng.integers(2,7)); z=np.r_[rng.normal(0,1,K),2*np.pi*rng.random(2*K)]
    z=minimize(obj,z,args=(K,),method='Nelder-Mead',options={'maxiter':6000,'fatol':1e-10,'xatol':1e-9}).x
    wts,s,t=unpack(z,K); ms=(wts*s).sum(); mt=(wts*t).sum(); mst=(wts*s*t).sum()
    k=kappa(ms,mt,mst); kill=abs(1+ms+mt+mst)
    if kill<1e-4 and k<best: best=k; print(f"trial {trial} K={K}: kappa={k:.5f} kill={kill:.1e} moments s,t,st = {np.round([ms,mt,mst],4)}",flush=True)
print("best kappa:",best,"  target 1.5")
