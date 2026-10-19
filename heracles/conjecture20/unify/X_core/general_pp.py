# Does the positive-part compression certify RT_n for GENERAL interference?
# Certificate: y(w0,c) = alpha + c * beta * (cos theta(w0))_+ ; best over w0 in Z_p, c >= 0.
# RT_n holds at the configuration if  M(y)/(sum y^r)^s * B^s >= LHS  for some (w0,c), B = sum alpha^r + lambda sum beta^r.
# We search for configurations maximising LHS / [Gamma_s B^s] (the RT_n ratio), then report
#   cert = max_{w0,c} M(y)B^s/(sum y^r)^s / LHS   (>= 1 means certified without knowing Gamma_s).
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize
rng=np.random.default_rng(3)
def parts(x,F): return x[:F]**2, x[F:2*F]**2, x[2*F:]
def lhs(al,be,ph,p,K):
    F=len(al); l=np.arange(1,F+1); w=np.arange(p)
    U=(be[None,:]*np.cos(2*np.pi*np.outer(w,l)/p+ph[None,:])).sum(1)
    dep=np.array([np.max(np.abs(U-np.roll(U,d))) for d in range(1,F+1)])
    return np.min(K@(al+be)-dep)
def M(y,K): return np.min(K@y)
def cert(al,be,ph,p,s,lam,K):
    F=len(al); l=np.arange(1,F+1); r=1/s; B=(al**r).sum()+lam*(be**r).sum(); L=lhs(al,be,ph,p,K)
    if L<=0: return np.inf
    best=0
    for w0 in range(p):
        pp=np.maximum(np.cos(2*np.pi*l*w0/p+ph),0)
        for c in np.linspace(0,3,61):
            y=al+c*be*pp
            if y.sum()<=0: continue
            best=max(best,M(y,K)*B**s/((y**r).sum())**s)
    return best/L
for (p,n,b) in [(11,2,2.0),(13,2,2.0),(13,3,2.0),(17,2,2.0),(23,3,2.0),(29,2,2.5),(29,3,2.0)]:
    F=(p-1)//2; K=G.kappa_mat(p)[:F]; s=(n+1)/b; lam=(n/(n+1))**(1-1/s); r=1/s
    Gam=G.programme(p,s,restarts=30)[0]
    def ratio(x):
        al,be,ph=parts(x,F); B=(al**r).sum()+lam*(be**r).sum()
        return lhs(al,be,ph,p,K)/(Gam*B**s)
    worst_cert=np.inf; best_ratio=0
    for t in range(12):
        x0=np.concatenate([rng.uniform(0,1,F)*(rng.uniform(size=F)<0.5),rng.uniform(0.2,1,F),rng.uniform(0,2*np.pi,F)])
        res=minimize(lambda x:-ratio(x),x0,method='Nelder-Mead',options={'maxiter':4000})
        al,be,ph=parts(res.x,F); rt=-res.fun
        if rt>0.5:
            cc=cert(al,be,ph,p,s,lam,K); worst_cert=min(worst_cert,cc); best_ratio=max(best_ratio,rt)
    print(f"(p,n,b)=({p},{n},{b})  max RT ratio found {best_ratio:.4f}   worst positive-part certificate (>=1 certifies) {worst_cert:.4f}")
