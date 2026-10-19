# k#(n,s) = sup (Re A - nu^{1-s} Q^s) / (sigma f(gamma_max)),  gamma_max = (3/4)|tau|/sigma^3,
# tau = sum_{T,T' disjoint nonempty, ordered} beta_T beta_T' conj(beta_{T u T'}), sigma^2=(1/2)sum_T |beta_T|^2,
# f(g) = (sqrt(g^2+4)-g)/2.  Also reports plain k* (= sup bonus/sigma) found on the way.
import numpy as np, itertools, sys
from scipy.optimize import minimize
n=int(sys.argv[1]); s=float(sys.argv[2]) if len(sys.argv)>2 else (n+1)/2; nu=n+1; r=1/s
masks=list(range(1,2**n)); V=np.array([[(m>>t)&1 for t in range(n)] for m in masks]); ix={m:i for i,m in enumerate(masks)}
trip=np.array([(ix[a],ix[b],ix[a|b]) for a in masks for b in masks if a&b==0])
f=lambda g:(np.sqrt(g*g+4)-g)/2
def parts(v,N):
    b=np.abs(v[:N])*np.exp(1j*v[N:2*N]); th=v[2*N:].reshape(N,n)
    A=b.sum(); BT=np.exp(1j*(th@V.T)).T@b
    bonus=A.real-nu**(1-s)*np.sum(np.abs(b)**r)**s
    sig=np.sqrt(0.5*np.sum(np.abs(BT)**2))
    tau=np.sum(BT[trip[:,0]]*BT[trip[:,1]]*np.conj(BT[trip[:,2]]))
    return bonus,sig,tau
def ks(v,N):
    bonus,sig,tau=parts(v,N)
    if sig<1e-9: return -9
    return bonus/(sig*f(0.75*abs(tau)/sig**3))
def kst(v,N):
    bonus,sig,tau=parts(v,N); return bonus/max(sig,1e-9) if sig>1e-9 else -9
rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 0); best=-9; bk=-9
for N in range(2,nu+3):
    for t in range(25):
        v0=np.concatenate([rng.uniform(0.1,1,N),rng.uniform(0,2*np.pi,N+N*n)])
        res=minimize(lambda v:-ks(v,N),v0,method='Nelder-Mead',options={'maxiter':8000,'xatol':1e-10,'fatol':1e-13})
        best=max(best,-res.fun)
        res=minimize(lambda v:-kst(v,N),v0,method='Nelder-Mead',options={'maxiter':8000,'xatol':1e-10,'fatol':1e-13})
        bk=max(bk,-res.fun)
print(f"n={n} s={s}: k# >= {best:.4f}   k* >= {bk:.4f}",flush=True)
