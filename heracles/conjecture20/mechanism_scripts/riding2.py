# Reduced exact model of 'cheap feature + parasitic riding orbits' at n=2 (large-p limit of the single-character min).
# margin(delta) = sum_{k!=l} y_k kappa_k / sqrt(1+rho_k^2) + boost*y_l*kappa_l - | boost*y_l e^{i t_l} c_l + sum_j eta*y_j rho_j/sqrt(1+rho_j^2) e^{i t_j} c_j |
# with sum_k y_k^r = 1 (IF prices).  IF value = Gamma_s.  Ratio > 1 means the mechanism beats IF.
import numpy as np, sys
from scipy.optimize import minimize
from prog2 import solve
s=float(sys.argv[1]); r=1/s; boost=(1.5)**(s-1); eta=1.0
for p in [int(a) for a in sys.argv[2:]]:
    G,y0,cov=solve(p,s); F=(p-1)//2; S=np.nonzero(y0)[0]+1; K=len(S)
    d=np.arange(1,p); rho_=np.exp(2j*np.pi/p)
    kap=1-np.cos(2*np.pi*np.outer(d,S)/p); C=1-rho_**(-np.outer(d,S))
    best=-1
    for li in range(K):
        def margin(v,soft=None):
            yy=np.abs(v[:K]); yy=yy/np.sum(yy**r)**s; rh=np.abs(v[K:2*K]); th=v[2*K:3*K]
            amp=yy/np.sqrt(1+rh**2); amp[li]=boost*yy[li]
            mode=eta*yy*rh/np.sqrt(1+rh**2); mode[li]=boost*yy[li]
            cover=kap@amp; Z=np.abs(C@(mode*np.exp(1j*th)))
            g=cover-Z
            return g.min() if soft is None else -soft*np.log(np.mean(np.exp(-(g-g.min())/soft)))+g.min()
        rng=np.random.default_rng(li)
        for trial in range(8):
            v0=np.concatenate([y0[S-1]*(1+0.1*rng.standard_normal(K)),rng.uniform(0,1.5,K),rng.uniform(0,2*np.pi,K)])
            for soft in [0.02,0.005,0.001]:
                res=minimize(lambda v:-margin(v,soft*G),v0,method='L-BFGS-B',options={'maxiter':3000})
                v0=res.x
            val=margin(v0)/G
            if val>best: best=val; bv=(li,v0.copy())
    li,v=bv; yy=np.abs(v[:K]); yy=yy/np.sum(yy**r)**s
    print(f"p={p} s={s} K*={K}: best margin/IF = {best:.4f}  cheap l={S[li]}  y={np.round(yy,4)} y0={np.round(y0[S-1],4)} rho={np.round(np.abs(v[K:2*K]),3)}",flush=True)
    np.save(f"riding_p{p}.npy",np.concatenate([[li],v]))
