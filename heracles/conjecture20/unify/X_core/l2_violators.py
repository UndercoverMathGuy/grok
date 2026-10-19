# Locate L2-RT violators (phase-free bound fails) and test whether RT_n holds once the network
# is allowed its WORST (flattest) phases: maximise over phases min_delta[cov - D_delta(phi)] / (Gamma B^s).
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize
rng=np.random.default_rng(5)
def setup(p,n,b):
    F=(p-1)//2; K=G.kappa_mat(p)[:F]; s=(n+1)/b; lam=(n/(n+1))**(1-1/s); r=1/s
    Gam=G.programme(p,s,restarts=40)[0]; l=np.arange(1,F+1); W=np.arange(p)
    def l2ratio(al,be):
        return np.min(K@(al+be)-np.sqrt(K@(be**2)))/(Gam*((al**r).sum()+lam*(be**r).sum())**s)
    def rtratio(al,be,ph):
        U=(be[None,:]*np.cos(2*np.pi*np.outer(W,l)/p+ph[None,:])).sum(1)
        dep=np.array([np.max(np.abs(U-np.roll(U,d))) for d in range(1,F+1)])
        return np.min(K@(al+be)-dep)/(Gam*((al**r).sum()+lam*(be**r).sum())**s)
    return F,K,s,lam,r,Gam,l2ratio,rtratio
for (p,n,b) in [(23,3,2.0),(29,3,2.0),(29,2,2.5)]:
    F,K,s,lam,r,Gam,l2ratio,rtratio=setup(p,n,b)
    best=(0,None)
    for t in range(40):
        x0=np.concatenate([rng.uniform(0,1,F)*(rng.uniform(size=F)<0.5),rng.uniform(0,1,F)])
        res=minimize(lambda x:-l2ratio(x[:F]**2,x[F:]**2),x0,method='L-BFGS-B')
        if -res.fun>best[0]: best=(-res.fun,res.x)
    al,be=best[1][:F]**2,best[1][F:]**2; sc=1/ (al+be).sum(); al,be=al*sc,be*sc
    print(f"\n(p,n,b)=({p},{n},{b}) L2-RT max ratio {best[0]:.4f}")
    print("  alpha=",np.round(al,3)); print("  beta =",np.round(be,3))
    # worst phases for the real RT ratio
    wr=0
    for t in range(30):
        ph0=rng.uniform(0,2*np.pi,F)
        res=minimize(lambda ph:-rtratio(al,be,ph),ph0,method='Nelder-Mead',options={'maxiter':3000})
        wr=max(wr,-res.fun)
    print(f"  RT_n ratio with the flattest phases found: {wr:.4f}   (<=1 means phases rescue it)")
    # where does the cheap part sit: number of features, coverage profile
    print("  #cheap features:",int((be>1e-3).sum()),"  #IF features:",int((al>1e-3).sum()))
