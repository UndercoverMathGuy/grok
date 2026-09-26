import numpy as np, itertools, sys
def vertices(p):
    n=(p-1)//2
    J=np.arange(1,n+1); D=np.arange(1,n+1)
    M=1-np.cos(2*np.pi*np.outer(D,J)/p)  # rows delta, cols j
    V=set()
    for k in range(1,n+1):
        for S in itertools.combinations(range(n),k):
            MS=M[:,S]
            for T in itertools.combinations(range(n),k):
                A=MS[list(T)]
                if abs(np.linalg.det(A))<1e-10: continue
                v=np.linalg.solve(A,np.ones(k))
                if (v<=1e-12).any(): continue
                if (MS@v<1-1e-9).any(): continue
                full=np.zeros(n); full[list(S)]=v
                V.add(tuple(np.round(full,10)))
    return np.array(sorted(V))
def phi(V,s): return (V**(1/s)).sum(1)
for p in [int(a) for a in sys.argv[1:]]:
    V=vertices(p); n=(p-1)//2
    F=V[(V>1e-12).all(1)]
    print(p,'full-support vertices',len(F),'uniform among them',any(np.allclose(f,2/p) for f in F))
    lo,hi=1.0001,4.0
    def dense_opt(s): return phi(F,s).min() <= phi(V,s).min()+1e-10
    if not dense_opt(lo): print(p,"not dense near 1"); continue
    for _ in range(60):
        mid=(lo+hi)/2
        (lo,hi)=(mid,hi) if dense_opt(mid) else (lo,mid)
    print(p, len(V), "dense up to s=%.4f"%lo, "ln-bound vacuous" if (p-1)<=2*np.ceil(8*np.log(p)) else "bound s-1<=%.3f"%(np.log(4)/np.log((p-1)/(2*np.ceil(8*np.log(p))))))
