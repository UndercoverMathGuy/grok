# Same as pp_adversarial.py but with the cheap share pinned: sum beta = theta * sum(alpha+beta), theta in {0.3,0.6,1}.
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize
rng=np.random.default_rng(11)
def run(p,n,b,theta,trials):
    F=(p-1)//2; K=G.kappa_mat(p)[:F]; s=(n+1)/b; lam=(n/(n+1))**(1-1/s); r=1/s
    l=np.arange(1,F+1); W=np.arange(p)
    cs=np.concatenate([np.linspace(0,2,41),[2.5,3,4,6,10]])
    def unpack(x):
        al=x[:F]**2; be=x[F:2*F]**2; ph=x[2*F:]
        sa,sb=al.sum(),be.sum()
        if theta>=1: al=0*al; be=be/sb
        else: al=al*(1-theta)/sa; be=be*theta/sb
        return al,be,ph
    def evalc(x):
        al,be,ph=unpack(x)
        U=(be[None,:]*np.cos(2*np.pi*np.outer(W,l)/p+ph[None,:])).sum(1)
        dep=np.array([np.max(np.abs(U-np.roll(U,d))) for d in range(1,F+1)])
        L=np.min(K@(al+be)-dep)
        if L<=1e-6: return 10.0
        B=(al**r).sum()+lam*(be**r).sum(); best=0
        PP=np.maximum(np.cos(2*np.pi*np.outer(W,l)/p+ph[None,:]),0)
        for w0 in range(p):
            Y=al[None,:]+cs[:,None]*(be*PP[w0])[None,:]
            cov=(Y@K.T).min(1); cost=(Y**r).sum(1); ok=cost>0
            if ok.any(): best=max(best,np.max(cov[ok]*B**s/cost[ok]**s))
        return best/L
    worst=np.inf
    for t in range(trials):
        x0=np.concatenate([rng.uniform(0.1,1,F),rng.uniform(0.1,1,F),rng.uniform(0,2*np.pi,F)])
        res=minimize(evalc,x0,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-7,'fatol':1e-10})
        worst=min(worst,res.fun)
    return worst
for (p,n,b) in [(7,2,2.0),(11,2,2.0),(11,3,2.0),(13,2,2.0),(13,2,2.5),(17,2,2.0),(19,3,2.0)]:
    out=[]
    for theta in [0.3,0.6,1.0]:
        out.append(round(run(p,n,b,theta,8),4))
    print(f"(p,n,b)=({p},{n},{b})  min certificate at cheap share 0.3/0.6/1.0: {out}   (10 = no positive-margin config found)")
