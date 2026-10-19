# Riding model with the cheap feature FORCED on: y_l fixed at a fraction f of the total share; everything else optimised.
# Start from the IF optimum (rho=0) so the optimiser can always reach the IF value at f->0.
import numpy as np, sys
from scipy.optimize import minimize
from prog2 import solve
s=float(sys.argv[1]); r=1/s; boost=(1.5)**(s-1); eta=1.0
p=int(sys.argv[2])
G,y0,cov=solve(p,s); S=np.nonzero(y0)[0]+1; K=len(S); ys=y0[S-1]
d=np.arange(1,p); rho_=np.exp(2j*np.pi/p)
kap=1-np.cos(2*np.pi*np.outer(d,S)/p); C=1-rho_**(-np.outer(d,S))
def margin(v,li,soft=None):
    yy=np.abs(v[:K]); yy=yy/np.sum(yy**r)**s; rh=np.abs(v[K:2*K]); th=v[2*K:3*K]
    amp=yy/np.sqrt(1+rh**2); amp[li]=boost*yy[li]
    mode=eta*yy*rh/np.sqrt(1+rh**2); mode[li]=boost*yy[li]
    g=kap@amp-np.abs(C@(mode*np.exp(1j*th)))
    return g.min() if soft is None else g.min()-soft*np.log(np.mean(np.exp(-(g-g.min())/soft)))
print(f"p={p} K*={K} Gamma={G:.6f}",flush=True)
for li in range(K):
    out=[]
    for f in [1.0,0.5,0.2]:
        best=-9
        for trial in range(6):
            rng=np.random.default_rng(100*li+trial)
            v=np.concatenate([ys*(1+0.05*rng.standard_normal(K)),np.zeros(K)+0.3*rng.uniform(0,1,K)*(trial>0),rng.uniform(0,2*np.pi,K)])
            v[li]=ys[li]*f
            def obj(w,soft):
                vv=w.copy(); vv[li]=ys[li]*f*np.sum(np.abs(w[:K])**r)**s/np.sum(np.abs(ys)**r)**s  # keep cheap share ~ fixed
                return -margin(vv,li,soft)
            for soft in [0.02,0.005,0.001]:
                res=minimize(lambda w:obj(w,soft*G),v,method='L-BFGS-B',options={'maxiter':3000}); v=res.x
            vv=v.copy(); vv[li]=ys[li]*f*np.sum(np.abs(v[:K])**r)**s/np.sum(np.abs(ys)**r)**s
            best=max(best,margin(vv,li)/G)
        out.append(best)
    print(f"  cheap feature {S[li]}: best margin/IF with its share at f=1,0.5,0.2 of programme share: {np.round(out,4)}",flush=True)
