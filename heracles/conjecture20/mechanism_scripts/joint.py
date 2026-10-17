# Joint single-feature constant J(n,s,p) = sup bonus / max(k_p*sigma, cos(pi/p)*max_F sum_{T in F}|beta_T|)
# (collision-free families F), sigma^2 = (1/2) sum_{T nonempty}|beta_T|^2.  J<=1 => single-frequency single features
# cannot beat IF at (n,p) (L^2 bound and pointwise both-edges bound hold simultaneously).
import numpy as np, itertools, sys
from scipy.optimize import minimize
n=int(sys.argv[1]); s=float(sys.argv[2]); p=int(sys.argv[3]); nu=n+1; r=1/s
kp=np.sqrt(1/np.tan(np.pi/p)**2-1); cp_=np.cos(np.pi/p)
Ts=[T for k in range(1,n+1) for T in itertools.combinations(range(n),k)]
V=np.array([[1 if t in T else 0 for t in range(n)] for T in Ts])
def collision_free(F):
    for k in range(2,len(F)+1):
        for sub in itertools.combinations(F,k):
            for sg in itertools.product((1,-1),repeat=k-1):
                vec=V[sub[0]]+sum(e*V[i] for e,i in zip(sg,sub[1:]))
                if np.all((vec==0)|(vec==1)) and vec.sum()>0: return False
                if np.all((vec==0)|(vec==-1)) and vec.sum()<0: return False
    return True
fams=[list(F) for k in range(1,min(n,4)+1) for F in itertools.combinations(range(len(Ts)),k) if np.linalg.matrix_rank(V[list(F)])==k and collision_free(F)]
FM=np.zeros((len(fams),len(Ts)))
for i,F in enumerate(fams): FM[i,F]=1
def J(v,N):
    b=np.abs(v[:N])*np.exp(1j*v[N:2*N]); th=v[2*N:].reshape(N,n)
    A=abs(b.sum()); B=np.abs(np.exp(1j*(th@V.T)).T@b)
    bonus=A-nu**(1-s)*np.sum(np.abs(b)**r)**s
    den=max(kp*np.sqrt(0.5*np.sum(B**2)),cp_*(FM@B).max(),1e-12)
    return bonus/den
rng=np.random.default_rng(0); best=-9
for N in range(1,nu+2):
    for t in range(30):
        v0=np.concatenate([rng.uniform(0.1,1,N),rng.uniform(0,2*np.pi,N+N*n)])
        res=minimize(lambda v:-J(v,N),v0,method='Nelder-Mead',options={'maxiter':6000,'xatol':1e-10,'fatol':1e-13})
        best=max(best,-res.fun)
print(f"n={n} s={s} p={p}: joint constant J >= {best:.4f}  (k_p={kp:.3f}, cos={cp_:.3f}, families={len(fams)})",flush=True)
