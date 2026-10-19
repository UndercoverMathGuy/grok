# Danger test: all-cheap network sitting on a programme-optimal allocation (beta = y*, alpha = 0).
# RT_n here needs  min_delta [ sum y* kappa - D_delta ] <= lambda^s * M(y*)   (Gamma_s is attained by y*).
# L2 version uses D >= sqrt(sum y*^2 kappa). Report K*, the L2 ratio, and the ratio with Kronecker-type
# (sup-norm) depth for the flattest phases we can find.
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize, linprog
rng=np.random.default_rng(4)
def rwlp(K,r,s,w):
    for it in range(400):
        res=linprog(w,A_ub=-K,b_ub=-np.ones(K.shape[0]),bounds=(0,None),method='highs'); y=res.x
        wn=(y+1e-10)**(r-1)
        if np.allclose(wn,w,rtol=1e-10): break
        w=wn
    return np.sum(y**r)**(-s), y/np.sum(y**r)**s
def gamma(p,s,nstarts):
    F=(p-1)//2; r=1/s; K=G.kappa_mat(p)[:F]; best=(0,None)
    for w0 in [np.ones(F)]+[rng.uniform(0.05,2,F) for _ in range(nstarts)]:
        v,y=rwlp(K,r,s,(w0+1e-10)**(r-1))
        if v>best[0]: best=(v,y)
    return best
n=2
for (p,s) in [(101,1.5),(199,1.5),(401,1.5),(809,1.5),(1601,1.5),(401,1.25),(1601,1.25)]:
    F=(p-1)//2; K=G.kappa_mat(p)[:F]; lam=(n/(n+1))**(1-1/s)
    Gm,y=gamma(p,s,30 if p<500 else 12)
    supp=y>1e-9*y.max(); Ks=int(supp.sum())
    cov=K@y; Mv=cov.min()
    l2=np.min(cov-np.sqrt(K@(y**2)))
    msg=f"p={p} s={s} K*={Ks} Gamma={Gm:.5f}  L2 ratio={l2/(lam**s*Mv):.4f}"
    if p<=401:
        # flattest phases on this support: maximise min_delta[cov - D_delta(phi)]
        l=np.arange(1,F+1)[supp]; yb=y[supp]; W=np.arange(p)
        def lhs(ph):
            U=(yb[None,:]*np.cos(2*np.pi*np.outer(W,l)/p+ph[None,:])).sum(1)
            dep=np.array([np.max(np.abs(U-np.roll(U,d))) for d in range(1,F+1)])
            return np.min(cov-dep)
        best=-1e9
        for t in range(8):
            res=minimize(lambda ph:-lhs(ph),rng.uniform(0,2*np.pi,Ks),method='Nelder-Mead',options={'maxiter':3000})
            best=max(best,-res.fun)
        msg+=f"  true-depth ratio (flattest phases found)={best/(lam**s*Mv):.4f}"
    print(msg,flush=True)
