import numpy as np, itertools, math
from scipy.optimize import minimize
def der(k): return round(math.factorial(k)*sum((-1)**j/math.factorial(j) for j in range(k+1)))
def Phi(nu,s): return sum(math.comb(nu,k)*der(k)*nu**((s-1)*(k-1)) for k in range(2,nu+1))
def bound(n,s): nu=n+1; return 2*(1-nu**(1-s))*Phi(nu,s)
for n in (2,3,4):
    nu=n+1
    ints=[tuple(range(a,b)) for a in range(n) for b in range(a+1,n+1)]
    V=np.array([[1.0 if t in T else 0.0 for t in range(n)] for T in ints])
    for s in (1.01,1.03):
        r=1/s; best=-9; rng=np.random.default_rng(0)
        for N in range(1,nu+2):
            for t in range(15):
                f=lambda v:-((abs((np.abs(v[:N])*np.exp(1j*v[N:2*N])).sum())-nu**(1-s)*np.sum(np.abs(v[:N])**r)**s)/max(np.abs(np.exp(1j*(v[2*N:].reshape(N,n)@V.T)).T@(np.abs(v[:N])*np.exp(1j*v[N:2*N]))).max(),1e-12))
                v0=np.concatenate([rng.uniform(0.1,1,N),rng.uniform(0,2*np.pi,N+N*n)])
                res=minimize(f,v0,method='Nelder-Mead',options={'maxiter':8000,'adaptive':True}); best=max(best,-res.fun)
        print(f"n={n} s={s}: numerical sup bonus/max_interval|beta_T| ~ {best:.5f}   proven bound {bound(n,s):.4f}   Phi={Phi(nu,s):.3f}",flush=True)
from scipy.optimize import brentq
for n in (2,3,4,5):
    print(f"n={n}: proven (H_n) at 1/2 for s <= {brentq(lambda s:bound(n,s)-0.5,1.0000001,2):.5f};  at cos(pi/5) for s <= {brentq(lambda s:bound(n,s)-np.cos(np.pi/5),1.0000001,2):.5f}")
