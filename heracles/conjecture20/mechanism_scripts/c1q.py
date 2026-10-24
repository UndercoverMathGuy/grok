import numpy as np, itertools, sys
from scipy.optimize import minimize
n=int(sys.argv[1]); nu=n+1; s=float(sys.argv[2]); r=1/s
Ts=[T for k in range(1,n+1) for T in itertools.combinations(range(n),k)]
V=np.array([[1 if t in T else 0 for t in range(n)] for T in Ts])
fams=[list(F) for k in range(1,n+1) for F in itertools.combinations(range(len(Ts)),k) if np.linalg.matrix_rank(V[list(F)])==k]
FM=np.zeros((len(fams),len(Ts)))
for a,F in enumerate(fams): FM[a,F]=1
def ratio(v,N):
    b=np.abs(v[:N])*np.exp(1j*v[N:2*N]); th=v[2*N:].reshape(N,n)
    A=abs(b.sum()); B=np.abs(np.exp(1j*(th@V.T)).T@b)
    den=(FM@B).max()
    return (A-nu**(1-s)*np.sum(np.abs(b)**r)**s)/max(den,1e-12)
rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 0)
for N in range(1,6):
    best=-9
    for t in range(60):
        v0=np.concatenate([rng.uniform(0.1,1,N),rng.uniform(0,2*np.pi,N+N*n)])
        res=minimize(lambda v:-ratio(v,N),v0,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-9,'fatol':1e-12})
        best=max(best,-res.fun)
    print(f"n={n} s={s} width {N}: c1 >= {best:.4f}",flush=True)
