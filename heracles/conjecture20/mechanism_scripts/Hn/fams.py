import numpy as np, itertools
def collision_free_families(n):
    Ts=[T for k in range(1,n+1) for T in itertools.combinations(range(n),k)]
    V=np.array([[1 if t in T else 0 for t in range(n)] for T in Ts])
    targets=set()
    for v in V: targets.add(tuple(v)); targets.add(tuple(-v))
    out=[]
    for k in range(1,n+1):
        for F in itertools.combinations(range(len(Ts)),k):
            M=V[list(F)]
            if np.linalg.matrix_rank(M)<k: continue
            ok=True
            if k>=2:
                for sig in itertools.product((-1,0,1),repeat=k):
                    if sum(1 for x in sig if x)<2: continue
                    if tuple(np.array(sig)@M) in targets: ok=False; break
            if ok: out.append(F)
    return Ts,V,out
if __name__=="__main__":
    for n in (2,3,4):
        Ts,V,F=collision_free_families(n)
        mx=max(len(f) for f in F)
        print(n,"families:",len(F)," max size:",mx," examples of max:",[[Ts[i] for i in f] for f in F if len(f)==mx][:4])
