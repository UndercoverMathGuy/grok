# For correct programme supports: can the sign-flipped characters 2l*1_{T_l} (n=2, T_l in {1},{2},{12}) be chosen
# dissociated (all 3^K signed sums distinct in Z_p^2)?  Greedy/random search over T-assignments.
import numpy as np, itertools, sys
from prog2 import solve
def sums_distinct(vecs,p):
    seen={(0,0)}; cur=[(0,0)]
    for v in vecs:
        new=[]
        for c in cur:
            for sg in (1,-1):
                t=((c[0]+sg*v[0])%p,(c[1]+sg*v[1])%p)
                if t in seen: return False
                seen.add(t); new.append(t)
        cur=cur+new
    return True
s=float(sys.argv[1])
for p in [int(a) for a in sys.argv[2:]]:
    G,y,cov=solve(p,s,restarts=120); S=list(np.nonzero(y)[0]+1); K=len(S)
    Tvec={0:(1,0),1:(0,1),2:(1,1)}
    rng=np.random.default_rng(0); ok=None
    for trial in range(4000 if 3**K<=3**9 else 1500):
        assign=rng.integers(0,3,K)
        vecs=[((2*l*Tvec[a][0])%p,(2*l*Tvec[a][1])%p) for l,a in zip(S,assign)]
        if sums_distinct(vecs,p): ok=assign; break
    print(f"p={p} K*={K} 3^K={3**K} p^2={p*p}  dissociated T-assignment {'FOUND '+str(list(ok)) if ok is not None else 'not found'}",flush=True)
