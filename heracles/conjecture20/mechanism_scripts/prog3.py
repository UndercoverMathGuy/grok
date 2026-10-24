# Programme solver v3: local search over supports S with an accurate restricted solve on each S
# (reweighted LP restricted to S, many restarts -- cheap because |S| is small), swap/add/drop moves,
# random restarts.  Value = Q^{-s} with Q = min sum y^r over {y supported on S, K y >= 1}.
import numpy as np, sys
from scipy.optimize import linprog
from prog2 import kappa
def restricted(K,r,S,rng,tries=12):
    KS=K[:,S]; best=(np.inf,None)
    for t in range(tries):
        w=np.ones(len(S)) if t==0 else rng.exponential(1,len(S))
        for it in range(60):
            res=linprog(w,A_ub=-KS,b_ub=-np.ones(K.shape[0]),bounds=(0,None),method='highs')
            if res.status!=0: return (np.inf,None)
            y=res.x; wn=(y+1e-10)**(r-1)
            if np.allclose(wn,w,rtol=1e-8): break
            w=wn
        Q=np.sum(y[y>1e-12]**r)
        if Q<best[0]: best=(Q,y)
    return best
def solve(p,s,starts=20,seed=0,maxk=None):
    r=1/s; K=kappa(p); F=K.shape[1]; rng=np.random.default_rng(seed); glob=(np.inf,None,None)
    for st in range(starts):
        k=rng.integers(2,min(F,12)+1); S=sorted(rng.choice(F,k,replace=False))
        Q,y=restricted(K,r,S,rng)
        while Q==np.inf:
            S=sorted(set(S)|{int(rng.integers(F))}); Q,y=restricted(K,r,S,rng)
        improved=True
        while improved:
            improved=False
            Sact=[S[i] for i in range(len(S)) if y[i]>1e-12]
            moves=[]
            for a in Sact: moves.append(sorted(set(Sact)-{a}))
            for b in range(F):
                if b not in Sact: moves.append(sorted(set(Sact)|{b}))
            rng.shuffle(moves)
            for M in moves[:60]:
                if len(M)==0: continue
                Qm,ym=restricted(K,r,M,rng,tries=6)
                if Qm<Q-1e-10: S,Q,y=M,Qm,ym; improved=True; break
        if Q<glob[0]: glob=(Q,S,y)
    Q,S,y=glob; yy=np.zeros(F); yy[S]=y; yy=yy*(yy>1e-12)/Q**s
    return Q**(-s),yy,K@yy
if __name__=="__main__":
    s=float(sys.argv[1])
    for p in [int(a) for a in sys.argv[2:]]:
        G,y,cov=solve(p,s); S=list(np.nonzero(y)[0]+1)
        print(f"p={p} s={s} Gamma={G:.6f} K*={len(S)} S={S}",flush=True)
