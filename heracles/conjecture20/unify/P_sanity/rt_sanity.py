import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/E_reduce')
import elib as E
from scipy.optimize import minimize
rng=np.random.default_rng(0)
def ratio(x,p,n,s,G):
    F=(p-1)//2; al=np.abs(x[:F]); be=np.abs(x[F:2*F]); ph=x[2*F:]
    r=1/s; lam=(n/(n+1))**(1-r); l=np.arange(1,F+1); w=np.arange(p)
    U=lambda ww: (be[None,:]*np.cos(2*np.pi*np.outer(ww,l)/p+ph[None,:])).sum(1)
    best=np.inf
    for d in range(1,p):
        cov=((al+be)*(1-np.cos(2*np.pi*l*d/p))).sum()
        dep=(U(w-d)-U(w)).max()
        best=min(best,cov-dep)
    return best/(G*((al**r).sum()+lam*(be**r).sum())**s)
for (p,n,s) in [(7,2,1.5),(11,2,1.5),(13,2,1.5),(11,3,2.0),(13,2,1.2)]:
    G=E.programme_vertices(p,s)[0]; F=(p-1)//2; mx=0
    for t in range(40):
        x0=rng.normal(size=3*F)
        res=minimize(lambda x:-ratio(x,p,n,s,G),x0,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-9,'fatol':1e-12})
        mx=max(mx,-res.fun)
    print((p,n,s),'Gamma',round(G,5),'max LHS/RHS found',round(mx,6))
