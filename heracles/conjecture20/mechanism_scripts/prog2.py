# Robust programme solver: Gamma_s(p) = max_{sum y^r<=1} min_delta sum_j y_j kappa_j(delta).
# Concave minimisation of sum y^r over {y>=0: K y >= 1} -> vertices.  Many reweighted-LP restarts from random
# weights and random supports + local support moves; keeps the best vertex.  Returns (Gamma, y, coverage).
import numpy as np, sys
from scipy.optimize import linprog
def kappa(p):
    F=(p-1)//2; j=np.arange(1,F+1); return 1-np.cos(2*np.pi*np.outer(j,j)/p)
def rlp(K,r,w,allowed=None):
    F=K.shape[1]
    bounds=[(0,None) if (allowed is None or allowed[j]) else (0,0) for j in range(F)]
    for it in range(80):
        res=linprog(w,A_ub=-K,b_ub=-np.ones(K.shape[0]),bounds=bounds,method='highs')
        if res.status!=0: return None
        y=res.x; wn=(y+1e-10)**(r-1)
        if np.allclose(wn,w,rtol=1e-7): break
        w=wn
    return y
def solve(p,s,restarts=200,seed=0):
    r=1/s; K=kappa(p); F=K.shape[1]; rng=np.random.default_rng(seed); best=(np.inf,None)
    def val(y): yy=y[y>1e-12]; return np.sum(yy**r)
    for t in range(restarts):
        if t%2==0: w=rng.exponential(1,F)
        else:
            k=rng.integers(1,max(2,F//2)+1); allowed=np.zeros(F,bool); allowed[rng.choice(F,k,replace=False)]=True
            y=rlp(K,r,rng.exponential(1,F),allowed)
            if y is None: continue
            w=(y+1e-6)**(r-1)
        y=rlp(K,r,w)
        if y is None: continue
        if val(y)<best[0]: best=(val(y),y.copy())
    # local support moves: try dropping each support element
    improved=True
    while improved:
        improved=False; Q,y=best; S=np.nonzero(y>1e-12)[0]
        for j in S:
            allowed=np.ones(F,bool); allowed[j]=False
            for t in range(5):
                yy=rlp(K,r,rng.exponential(1,F)*~(~allowed),allowed)
                if yy is not None and val(yy)<Q-1e-12: best=(val(yy),yy.copy()); improved=True; break
            if improved: break
    Q,y=best; y=y*(y>1e-12); yn=y/Q**s
    return Q**(-s),yn,K@yn
if __name__=="__main__":
    s=float(sys.argv[1])
    for p in [int(a) for a in sys.argv[2:]]:
        G,y,cov=solve(p,s); S=np.nonzero(y)[0]+1; B=np.nonzero(cov<G*(1+1e-6))[0]+1
        print(f"p={p} s={s} Gamma={G:.6f} K*={len(S)} S={list(S)} |B|={len(B)}",flush=True)
