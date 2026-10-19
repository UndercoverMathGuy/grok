# Near-optimal programme supports at large p (heuristic): local search on min coverage with uniform weights,
# then reweighted LP for the restricted programme  Gamma_S = max min_delta (K_S y)_delta, sum y^{2/3} <= 1  (s=1.5).
import numpy as np, sys, json, time
from scipy.optimize import linprog
r=2/3; s=1.5
def kap_rows(p,ls,d):   # len(ls) x F
    return 1-np.cos(2*np.pi*np.outer(ls,d)/p)
def local_search(p,K,rng,iters=20000,batch=256):
    F=(p-1)//2; d=np.arange(1,F+1)
    S=list(rng.choice(np.arange(1,F+1),K,replace=False)); KS=kap_rows(p,S,d); cov=KS.sum(0); m=cov.min()
    stall=0; it=0
    while it<iters and stall<8*K:
        it+=1; j=it%K
        cand=rng.integers(1,F+1,batch); cand=cand[~np.isin(cand,S)]
        Kc=kap_rows(p,cand,d); base=cov-KS[j]
        mins=(base[None,:]+Kc).min(1); i=np.argmax(mins)
        if mins[i]>m+1e-12:
            S[j]=int(cand[i]); KS[j]=Kc[i]; cov=base+Kc[i]; m=mins[i]; stall=0
        else: stall+=1
    return sorted(S), m
def restricted(p,S,iters=60):
    F=(p-1)//2; d=np.arange(1,F+1); KS=kap_rows(p,S,d).T   # F x K
    w=np.ones(len(S))
    for it in range(iters):
        res=linprog(w,A_ub=-KS,b_ub=-np.ones(F),bounds=(0,None),method='highs')
        y=res.x; wn=(y+1e-10)**(r-1)
        if np.allclose(wn,w,rtol=1e-7): break
        w=wn
    Q=np.sum(y[y>1e-12]**r); yn=y/Q**s; keep=yn>1e-9
    return Q**(-s), [int(a) for a in np.array(S)[keep]], [float(a) for a in yn[keep]]
if __name__=="__main__":
    p=int(sys.argv[1]); Ks=[int(a) for a in sys.argv[2].split(',')]; reps=int(sys.argv[3]); out={}
    rng=np.random.default_rng(p)
    best=(0,None)
    for K in Ks:
        for rep in range(reps):
            t=time.time(); S,m=local_search(p,K,rng); G,S2,y=restricted(p,S)
            print(f"p={p} K={K} rep={rep} unif-min-cov={m:.3f} gamma_unif={m/K:.3f} Gamma_S={G:.5f} |S'|={len(S2)} {time.time()-t:.0f}s",flush=True)
            out.setdefault('runs',[]).append({'K':K,'rep':rep,'Gamma':G,'S':S2,'y':y})
            if G>best[0]: best=(G,{'Gamma':G,'S':S2,'y':y})
            out['best']=best[1]; json.dump(out,open(f'large_{p}.json','w'))
    print("BEST",p,best[0],len(best[1]['S']),flush=True)
