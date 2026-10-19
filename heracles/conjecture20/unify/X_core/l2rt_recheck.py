# Recheck the earlier L2-RT "violations" with a much stronger Gamma_s estimate.
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize, linprog
rng=np.random.default_rng(1)
def gamma_strong(p,s,extra_starts=()):
    # reweighted-LP from many starts incl. given allocations
    F=(p-1)//2; r=1/s; K=G.kappa_mat(p)[:F]; best=0
    starts=[np.ones(F)]+[rng.uniform(0.05,2,F) for _ in range(300)]+[np.maximum(a,1e-3) for a in extra_starts]
    for w0 in starts:
        w=(w0+1e-10)**(r-1)
        for it in range(300):
            res=linprog(w,A_ub=-K,b_ub=-np.ones(F),bounds=(0,None),method='highs'); y=res.x
            wn=(y+1e-10)**(r-1)
            if np.allclose(wn,w,rtol=1e-10): break
            w=wn
        best=max(best,np.sum(y**r)**(-s))
    return best
def ratio(x,K,s,lam,Gam):
    F=K.shape[1]; al=x[:F]**2; be=x[F:]**2; r=1/s
    return np.min(K@(al+be)-np.sqrt(K@(be**2)))/(Gam*((al**r).sum()+lam*(be**r).sum())**s)
for (p,n,b) in [(23,3,2.0),(29,3,2.0),(29,2,2.5),(29,2,2.0),(31,3,2.0)]:
    F=(p-1)//2; K=G.kappa_mat(p)[:F]; s=(n+1)/b; lam=(n/(n+1))**(1-1/s); r=1/s
    Gw=G.programme(p,s,restarts=30)[0]
    found=[]
    for t in range(60):
        x0=np.concatenate([rng.uniform(0,1,F)*(rng.uniform()<0.5), rng.uniform(0,1,F)])
        res=minimize(lambda x:-ratio(x,K,s,lam,Gw),x0,method='L-BFGS-B',options={'maxiter':3000})
        if -res.fun>1.0: found.append((-res.fun,res.x))
    found.sort(key=lambda z:-z[0])
    if not found: print((p,n,b),'no violator with weak Gamma',round(Gw,5)); continue
    v,x=found[0]; al=x[:F]**2; be=x[F:]**2
    Gs=gamma_strong(p,s,extra_starts=[al+be,be,al])
    print((p,n,b),f"weak Gamma {Gw:.5f} -> L2 ratio {v:.4f};  strong Gamma {Gs:.5f} -> L2 ratio {v*Gw/Gs:.4f};  #cheap={int((be>1e-3*be.max()).sum())} #IF={int((al>1e-3*max(al.max(),1e-12)).sum())}")
