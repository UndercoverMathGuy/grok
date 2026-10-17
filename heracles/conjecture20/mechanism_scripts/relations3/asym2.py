# inf over amplitude profiles beta(s) of the asymptotic tilted-mean certificate (relation mass -> 0 regime).
import numpy as np, sys
sys.path.insert(0,'.')
from asym import law, A, g, kU
from scipy.optimize import brentq, minimize
nb=24
def setup(kap):
    gam,w=law(kap); s=np.sqrt(kU/2)
    q=np.cumsum(w); edges=np.searchsorted(q,np.linspace(0,1,nb+1)[1:-1]); idx=np.split(np.arange(len(s)),edges)
    W=np.array([w[i].sum() for i in idx]); Sb=np.array([(w[i]*s[i]).sum()/w[i].sum() for i in idx])
    return gam,W,Sb
def val(beta,W,Sb,kap):
    beta=np.abs(beta)+1e-12
    f=lambda x: (W*g(x*beta)).sum()-1/kap
    x=brentq(f,1e-9,1e6)
    return (W*beta*A(x*beta)).sum()/(W*beta*Sb).sum()
if __name__=="__main__":
    for kap in [1.5,2.0,2.5,3.0,3.2,3.5,4.0,5.0]:
        gam,W,Sb=setup(kap); best=9
        rng=np.random.default_rng(0)
        for st in range(12):
            b0=Sb**rng.uniform(-1,3)*np.exp(rng.normal(0,0.5,nb))
            r=minimize(lambda b: val(b,W,Sb,kap),b0,method='Nelder-Mead',options={'maxiter':6000,'xatol':1e-6,'fatol':1e-7})
            best=min(best,r.fun)
        print(f"kap={kap:4.2f} gamma={gam:5.3f}  inf_beta TiltM/sum(b s) = {best:6.3f}   (b~s: {val(Sb,W,Sb,kap):6.3f})",flush=True)
