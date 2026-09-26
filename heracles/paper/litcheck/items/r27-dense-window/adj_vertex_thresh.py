import numpy as np
from scipy.optimize import brentq
def adj_vertices(p):
    n=(p-1)//2; J=np.arange(1,n+1)
    M=1-np.cos(2*np.pi*np.outer(J,J)/p)  # rows delta, cols j
    out=[]
    for j0 in range(n):
        S=[j for j in range(n) if j!=j0]
        for d0 in range(n):
            T=[d for d in range(n) if d!=d0]
            A=M[np.ix_(T,S)]
            if abs(np.linalg.det(A))<1e-12: continue
            v=np.linalg.solve(A,np.ones(n-1))
            if (v<=1e-12).any(): continue
            if M[d0,S]@v<1-1e-9: continue
            full=np.zeros(n); full[S]=v; out.append(full)
    return np.array(out)
def thr(p):
    n=(p-1)//2; u=np.full(n,2/p); V=adj_vertices(p)
    if len(V)==0: return None
    f=lambda s: (V**(1/s)).sum(1).min()-(u**(1/s)).sum()
    return brentq(f,1.00001,20) if f(20)<0 else None
for p in [7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,79,83,89,97,101,113]:
    t=thr(p)
    if t: print(p, "adjacent-vertex threshold s=%.4f  (s-1)*p=%.3f  (s-1)*ln p=%.3f"%(t,(t-1)*p,(t-1)*np.log(p)))
    else: print(p,'none')
