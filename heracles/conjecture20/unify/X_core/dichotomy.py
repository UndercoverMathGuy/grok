# Combined certificate family for RT_n:  y = alpha + c1*beta + c2*beta*(cos theta(w0))_+ ,  best over w0, c1, c2.
# cert(config) = max_y [ M(y) B^s / (sum y^r)^s ] / LHS   (>= 1 certifies RT_n without knowing Gamma_s).
# Adversary minimises cert over configurations with pinned cheap share.  Also reports which mechanism wins.
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
C1=np.linspace(0,1,11); C2=np.linspace(0,4,17)
def make(p,n,b,theta):
    F=(p-1)//2; K=G.kappa_mat(p)[:F]; s=(n+1)/b; lam=(n/(n+1))**(1-1/s); r=1/s
    l=np.arange(1,F+1); W=np.arange(p)
    def unpack(x):
        al=x[:F]**2; be=x[F:2*F]**2; ph=x[2*F:]
        if theta>=1: al=0*al; be=be/be.sum()
        else: al=al*(1-theta)/al.sum(); be=be*theta/be.sum()
        return al,be,ph
    def cert(x,detail=False):
        al,be,ph=unpack(x)
        Th=2*np.pi*np.outer(W,l)/p+ph[None,:]
        U=(be[None,:]*np.cos(Th)).sum(1)
        dep=np.array([np.max(np.abs(U-np.roll(U,d))) for d in range(1,F+1)])
        L=np.min(K@(al+be)-dep)
        if L<=1e-7: return 10.0
        B=(al**r).sum()+lam*(be**r).sum()
        PP=np.maximum(np.cos(Th),0)                                   # (p,F)
        # linear-only and positive-part-only and combined
        Y=al[None,None,None,:]+C1[None,:,None,None]*be+C2[None,None,:,None]*(be[None,None,None,:]*PP[:,None,None,:])  # (p,11,17,F)
        Y=Y.reshape(-1,F); cost=(Y**r).sum(1); ok=cost>0
        cov=(Y@K.T).min(1); val=np.where(ok,cov*B**s/np.maximum(cost,1e-300)**s,0)/L
        best=val.max()
        if detail:
            V=val.reshape(p,11,17)
            lin=V[:,:,0].max(); pp=V[:,0,:].max()
            return best,lin,pp,L/((K@(al+be)).min())
        return best
    return cert,F
for (p,n,b) in [(17,2,2.0),(23,2,2.0),(29,2,2.0),(31,3,2.0),(37,2,2.5)]:
    rows=[]
    for theta in [0.4,0.8,1.0]:
        cert,F=make(p,n,b,theta); worst=(np.inf,None)
        for t in range(5):
            x0=np.concatenate([rng.uniform(0.1,1,F),rng.uniform(0.1,1,F),rng.uniform(0,2*np.pi,F)])
            res=minimize(cert,x0,method='Nelder-Mead',options={'maxiter':2500,'xatol':1e-6,'fatol':1e-9})
            if res.fun<worst[0]: worst=(res.fun,res.x)
        if worst[0]<10:
            b_,lin,pp,flat=cert(worst[1],detail=True); rows.append((theta,round(b_,4),round(lin,4),round(pp,4),round(flat,3)))
        else: rows.append((theta,'no L>0'))
    print(f"(p,n,b)=({p},{n},{b}):", rows, "  [theta, best cert, linear-only, pos-part-only, L/min cov]")
