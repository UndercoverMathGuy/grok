# L2-RT with pinned cheap share theta = sum beta / sum(alpha+beta), and the structured family
# "beta uniform on a k-subset of F (random or interval-like), alpha = 0" at larger p.
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize, linprog
rng=np.random.default_rng(2)
def rwlp(K,r,s,w):
    for it in range(400):
        res=linprog(w,A_ub=-K,b_ub=-np.ones(K.shape[0]),bounds=(0,None),method='highs'); y=res.x
        wn=(y+1e-10)**(r-1)
        if np.allclose(wn,w,rtol=1e-10): break
        w=wn
    return np.sum(y**r)**(-s)
def gamma(p,s,nstarts,extra=()):
    F=(p-1)//2; r=1/s; K=G.kappa_mat(p)[:F]
    starts=[np.ones(F)]+[rng.uniform(0.05,2,F) for _ in range(nstarts)]+[np.maximum(a,1e-4) for a in extra]
    return max(rwlp(K,r,s,(w+1e-10)**(r-1)) for w in starts)
def l2lhs(al,be,K): return np.min(K@(al+be)-np.sqrt(K@(be**2)))
print("== pinned cheap share (adversarial over alpha,beta) ==")
for (p,n,b) in [(13,2,2.0),(23,2,2.0),(23,3,2.0),(31,2,2.0),(37,2,2.5)]:
    F=(p-1)//2; K=G.kappa_mat(p)[:F]; s=(n+1)/b; lam=(n/(n+1))**(1-1/s); r=1/s; Gm=gamma(p,s,80)
    out=[]
    for th in [0.2,0.5,1.0]:
        def rat(x):
            al=x[:F]**2; be=x[F:]**2
            if th>=1: al=0*al; be=be/be.sum()
            else: al=al*(1-th)/al.sum(); be=be*th/be.sum()
            return l2lhs(al,be,K)/(Gm*((al**r).sum()+lam*(be**r).sum())**s)
        best=0
        for t in range(25):
            x0=np.concatenate([rng.uniform(0.05,1,F),rng.uniform(0.05,1,F)])
            res=minimize(lambda x:-rat(x),x0,method='Nelder-Mead',options={'maxiter':6000,'xatol':1e-8,'fatol':1e-12})
            best=max(best,-res.fun)
        out.append(round(best,4))
    print(f"(p,n,b)=({p},{n},{b}) max L2-RT ratio at cheap share 0.2/0.5/1.0: {out}",flush=True)
print("== beta uniform on k-subsets of F, alpha=0, n=2, s near 1 (the window) ==")
for p in [61,101,151,199]:
    F=(p-1)//2; K=G.kappa_mat(p)[:F]
    for s in [1+3/np.sqrt(p),1+6/np.sqrt(p),1+12/np.sqrt(p),1.5]:
        n=2; lam=(n/(n+1))**(1-1/s); r=1/s; Gm=gamma(p,s,40)
        worst=0
        for k in [F, (3*F)//4, F//2, F//4]:
            for t in range(6):
                S=rng.choice(F,k,replace=False) if k<F else np.arange(F)
                be=np.zeros(F); be[S]=1.0; al=np.zeros(F)
                v=l2lhs(al,be,K)/(Gm*(lam*(be**r).sum())**s); worst=max(worst,v)
        print(f"  p={p} s={s:.4f}: max L2-RT ratio over uniform-on-subset cheap parts = {worst:.4f}",flush=True)
