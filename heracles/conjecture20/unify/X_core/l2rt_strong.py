# Systematic test of the phase-free core
#   L2-RT: min_delta [ sum (alpha+beta) kappa(delta) - sqrt(sum beta^2 kappa(delta)) ] <= Gamma_s (sum alpha^r + lambda sum beta^r)^s
# Adversary uses a moderately strong Gamma; every apparent violation is re-checked with a strong Gamma
# (300+ reweighted-LP starts, plus starts at the violator's own allocations).
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize, linprog
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
rng=np.random.default_rng(seed)
def rwlp(K,r,s,w):
    for it in range(300):
        res=linprog(w,A_ub=-K,b_ub=-np.ones(K.shape[0]),bounds=(0,None),method='highs'); y=res.x
        wn=(y+1e-10)**(r-1)
        if np.allclose(wn,w,rtol=1e-10): break
        w=wn
    return np.sum(y**r)**(-s)
def gamma(p,s,nstarts,extra=()):
    F=(p-1)//2; r=1/s; K=G.kappa_mat(p)[:F]
    starts=[np.ones(F)]+[rng.uniform(0.05,2,F) for _ in range(nstarts)]+[np.maximum(a,1e-4) for a in extra]
    return max(rwlp(K,r,s,(w+1e-10)**(r-1)) for w in starts)
def ratio(x,K,s,lam,Gam):
    F=K.shape[1]; al=x[:F]**2; be=x[F:]**2; r=1/s
    return np.min(K@(al+be)-np.sqrt(K@(be**2)))/(Gam*((al**r).sum()+lam*(be**r).sum())**s)
for (p,n,b) in [(11,2,2.0),(13,2,2.0),(17,2,2.0),(19,3,2.0),(23,2,2.0),(23,3,2.0),(29,2,2.0),(29,3,2.0),(31,2,2.5),(37,2,2.0),(37,3,2.0),(41,2,2.0),(41,4,2.0),(43,2,2.2)]:
    F=(p-1)//2; K=G.kappa_mat(p)[:F]; s=(n+1)/b; lam=(n/(n+1))**(1-1/s); r=1/s
    Gm=gamma(p,s,60)
    best=(0,None)
    for t in range(50):
        mode=t%3
        a0=rng.uniform(0,1,F)*(rng.uniform(size=F)<0.5) if mode else np.zeros(F)
        b0=rng.uniform(0,1,F)*(rng.uniform(size=F)<(0.3 if mode==2 else 1.0))
        x0=np.concatenate([a0,b0])
        res=minimize(lambda x:-ratio(x,K,s,lam,Gm),x0,method='L-BFGS-B',options={'maxiter':4000})
        if -res.fun>best[0]: best=(-res.fun,res.x)
    v,x=best; al=x[:F]**2; be=x[F:]**2
    msg=f"(p,n,b)=({p},{n},{b}) max L2 ratio {v:.4f} (Gamma~{Gm:.5f})"
    if v>0.995:
        Gs=gamma(p,s,400,extra=[al+be,be,al,(al+be)**2,np.sqrt(al+be+1e-12)])
        msg+=f" -> strong Gamma {Gs:.5f}: {v*Gm/Gs:.4f}"
    msg+=f"  #cheap={int((be>1e-3*max(be.max(),1e-12)).sum())} #IF={int((al>1e-3*max(al.max(),1e-12)).sum())} cheap share={be.sum()/(al+be).sum():.2f}"
    print(msg, flush=True)
