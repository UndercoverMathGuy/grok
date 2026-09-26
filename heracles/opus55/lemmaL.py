# Test (L): P(a)=sum_l c_l prod_j (1+v_lj a_j) on torus T^n. Claim: min Re P <= nu^{-(1-r)/r} ||c||_r
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(0)
def run(n,N,r,G=40,trials=40):
    th=np.linspace(0,2*np.pi,G,endpoint=False)
    grids=np.meshgrid(*([th]*n),indexing='ij'); A=np.stack([np.exp(1j*g.ravel()) for g in grids],1)
    nu=n+1; best=-1
    def val(z,soft=None):
        c=z[:N]*np.exp(1j*z[N:2*N]); V=np.exp(1j*z[2*N:].reshape(N,n))
        terms=np.prod(1+V[None,:,:]*A[:,None,:],axis=2)
        P=(terms*c[None,:]).real.sum(1)
        cost=(np.abs(c)**r).sum()**(1/r)
        m=-soft*np.log(np.mean(np.exp(-(P-P.min())/soft)))+P.min() if soft else P.min()
        return m/cost
    for t in range(trials):
        z=np.concatenate([rng.random(N)+.1,rng.uniform(0,2*np.pi,N),rng.uniform(0,2*np.pi,N*n)])
        for s in (0.05,0.01,0.002):
            z=minimize(lambda z:-val(z,s),z,method='L-BFGS-B').x
        best=max(best,val(z))
    return best, nu**(-(1-r)/r)
for n,r in [(1,0.5),(1,2/3),(2,2/3),(2,0.5),(3,0.5)]:
    for N in ([2,3,4] if n==1 else [3,4,5,6]):
        b,t=run(n,N,r,G=(60 if n<=2 else 16),trials=(30 if n<=2 else 10))
        print(f"n={n} r={r:.3f} N={N}: best {b:.5f}  claim-bound {t:.5f}  ratio {b/t:.4f}",flush=True)
