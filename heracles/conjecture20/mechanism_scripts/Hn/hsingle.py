# single-T constant: sup (|A| - nu^{1-s}Q^s)/max_T|beta_T|. usage: n s Nmax restarts
import numpy as np, sys, itertools
from scipy.optimize import minimize
n=int(sys.argv[1]); s=float(sys.argv[2]); Nmax=int(sys.argv[3]); R=int(sys.argv[4]); nu=n+1; r=1/s
Ts=[T for k in range(1,n+1) for T in itertools.combinations(range(n),k)]
V=np.array([[1.0 if t in T else 0.0 for t in range(n)] for T in Ts])
def parts(v,N):
    b=np.abs(v[:N])*np.exp(1j*v[N:2*N]); th=v[2*N:].reshape(N,n)
    return b,th,np.exp(1j*(th@V.T)).T@b
def ratio(v,N):
    b,th,B=parts(v,N); A=abs(b.sum())
    return (A-nu**(1-s)*np.sum(np.abs(b)**r)**s)/max(np.abs(B).max(),1e-12)
rng=np.random.default_rng(3); ov=(-9,0,None)
for N in range(1,Nmax+1):
    best=(-9,None)
    for t in range(R):
        v0=np.concatenate([rng.uniform(0.1,1,N),rng.uniform(0,2*np.pi,N+N*n)])
        for rep in range(2):
            res=minimize(lambda v:-ratio(v,N),v0,method='Nelder-Mead',options={'maxiter':15000,'xatol':1e-11,'fatol':1e-14,'adaptive':True}); v0=res.x
        if -res.fun>best[0]: best=(-res.fun,res.x)
    print(f"n={n} s={s} width {N}: c1(single T) >= {best[0]:.5f}",flush=True)
    if best[0]>ov[0]: ov=(best[0],N,best[1])
b,th,B=parts(ov[2],ov[1]); A=b.sum()
print(" best width",ov[1],"|beta|/|A|",np.round(np.abs(b)/abs(A),4),"|beta_T|/|A|",np.round(np.abs(B)/abs(A),4))
