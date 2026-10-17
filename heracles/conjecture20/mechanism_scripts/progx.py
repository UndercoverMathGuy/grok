# Exhaustive vertex enumeration for the programme: vertices of {y>=0, K y>=1} with support size k <= kmax.
import numpy as np, itertools, sys
from prog2 import kappa
def exhaustive(p,s,kmax):
    r=1/s; K=kappa(p); F=K.shape[0]; best=(np.inf,None)
    for k in range(1,kmax+1):
        for S in itertools.combinations(range(F),k):
            KS=K[:,S]
            for B in itertools.combinations(range(F),k):
                M=KS[list(B),:]
                if abs(np.linalg.det(M))<1e-12: continue
                y=np.linalg.solve(M,np.ones(k))
                if np.any(y<-1e-12): continue
                if np.any(KS@y<1-1e-9): continue
                Q=np.sum(np.clip(y,0,None)**r)
                if Q<best[0]: best=(Q,(S,y))
    Q,(S,y)=best
    return Q**(-s),[j+1 for j in S],y/Q**s
if __name__=="__main__":
    p=int(sys.argv[1]); s=float(sys.argv[2]); kmax=int(sys.argv[3])
    G,S,y=exhaustive(p,s,kmax); print(f"p={p} s={s} exhaustive(k<={kmax}): Gamma={G:.6f} support={S} y={np.round(y,4)}",flush=True)
