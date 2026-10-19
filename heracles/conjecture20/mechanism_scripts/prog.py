# Programme Gamma_s(p) by reweighted LP (concave l_r minimisation over the coverage polytope), random restarts.
import numpy as np, sys
from scipy.optimize import linprog
def kappa(p):
    F=(p-1)//2; j=np.arange(1,F+1); d=np.arange(1,F+1)
    return 1-np.cos(2*np.pi*np.outer(d,j)/p)   # rows delta=1..F (symmetric), cols j
def solve(p,s,restarts=6,seed=0):
    r=1/s; K=kappa(p); F=K.shape[1]; rng=np.random.default_rng(seed); best=(np.inf,None)
    for t in range(restarts):
        w=rng.uniform(0.5,1.5,F) if t else np.ones(F)
        for it in range(60):
            res=linprog(w,A_ub=-K,b_ub=-np.ones(F),bounds=(0,None),method='highs')
            y=res.x; wn=(y+1e-9)**(r-1)
            if np.allclose(wn,w,rtol=1e-6): break
            w=wn
        Q=np.sum(y[y>1e-12]**r)
        if Q<best[0]: best=(Q,y.copy())
    Q,y=best; y=y*(y>1e-12); yn=y/Q**s; cov=K@yn
    return Q**(-s), yn, cov
if __name__=="__main__":
    s=float(sys.argv[1])
    for p in [int(a) for a in sys.argv[2:]]:
        G,y,cov=solve(p,s); S=np.nonzero(y)[0]+1; K=kappa(p)
        B=np.nonzero(cov<G*(1+1e-6))[0]+1
        kb=[K[d-1,S-1].sum() for d in B]
        print(f"p={p} s={s} Gamma={G:.5f} K*={len(S)} |B|={len(B)} sum_S kappa at binding: min {min(kb):.2f} max {max(kb):.2f}",flush=True)
