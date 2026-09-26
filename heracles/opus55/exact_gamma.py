# Exact ansatz programme restricted to a support S: C_r(S) = min sum x^r over {x>=0 on S, A_{.,S} x >= 1}, all vertices
# enumerated (float LA, feasibility tol 1e-9). Gamma_S = C_r^{-1/r}. usage: exact_gamma.py p r k1 k2 ...
import numpy as np, itertools, sys
p=int(sys.argv[1]); r=float(sys.argv[2]); S=[int(a) for a in sys.argv[3:]]; h=(p-1)//2
D=np.arange(1,h+1); A=1-np.cos(2*np.pi*np.outer(D,np.arange(1,h+1))/p)
best=(np.inf,None)
for K in range(1,len(S)+1):
    for Sp in itertools.combinations(S,K):
        cols=[k-1 for k in Sp]
        Ts=np.array(list(itertools.combinations(range(h),K)))
        M=A[Ts][:,:,cols]                                   # [nT,K,K]
        det=np.linalg.det(M); ok=np.abs(det)>1e-12
        x=np.linalg.solve(M[ok],np.ones((ok.sum(),K,1)))[...,0]
        pos=(x>-1e-12).all(1); x=x[pos]
        if len(x)==0: continue
        feas=(A[:,cols]@x.T>=1-1e-9).all(0); x=x[feas]
        if len(x)==0: continue
        c=(np.clip(x,0,None)**r).sum(1); i=c.argmin()
        if c[i]<best[0]: best=(c[i],(Sp,x[i]))
C,(Sp,x)=best
print(f"p={p} r={r} S={S}: Gamma_S = {C**(-1/r):.9f}, optimal sub-support {Sp}, shares {np.round(x**r/C,5)}")
