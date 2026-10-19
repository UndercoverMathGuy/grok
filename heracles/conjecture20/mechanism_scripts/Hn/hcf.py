# c1 with collision-free families. usage: n s Nmax restarts seed
import numpy as np, sys
from scipy.optimize import minimize
from fams import collision_free_families
n=int(sys.argv[1]); s=float(sys.argv[2]); Nmax=int(sys.argv[3]); R=int(sys.argv[4]); seed=int(sys.argv[5]) if len(sys.argv)>5 else 0
nu=n+1; r=1/s
Ts,V,F=collision_free_families(n); Vf=V.astype(float)
FM=np.zeros((len(F),len(Ts)))
for a,f in enumerate(F): FM[a,list(f)]=1
def parts(v,N):
    b=np.abs(v[:N])*np.exp(1j*v[N:2*N]); th=v[2*N:].reshape(N,n)
    return b,th,np.exp(1j*(th@Vf.T)).T@b
def ratio(v,N):
    b,th,B=parts(v,N); A=abs(b.sum()); den=(FM@np.abs(B)).max()
    return (A-nu**(1-s)*np.sum(np.abs(b)**r)**s)/max(den,1e-12)
rng=np.random.default_rng(seed); overall=(-9,None,None)
for N in range(1,Nmax+1):
    best=(-9,None)
    for t in range(R):
        v0=np.concatenate([rng.uniform(0.1,1,N),rng.uniform(0,2*np.pi,N+N*n)])
        for rep in range(2):
            res=minimize(lambda v:-ratio(v,N),v0,method='Nelder-Mead',options={'maxiter':15000,'xatol':1e-11,'fatol':1e-14,'adaptive':True}); v0=res.x
        if -res.fun>best[0]: best=(-res.fun,res.x)
    print(f"n={n} s={s} width {N}: c1(collision-free) >= {best[0]:.5f}",flush=True)
    if best[0]>overall[0]: overall=(best[0],N,best[1])
v=overall[2]; N=overall[1]; b,th,B=parts(v,N); A=b.sum()
print("best width",N," |beta|/|A|",np.round(np.abs(b)/abs(A),4)," arg(beta/A)",np.round(np.angle(b/A),4))
print(" xi rel atom0 (deg)\n",np.round(np.degrees(np.angle(np.exp(1j*(th-th[0])))),2))
print(" |beta_T|/|A|:",{Ts[k]:round(abs(B[k])/abs(A),4) for k in range(len(Ts))})
np.save(f"cf_n{n}_s{s}.npy",v)
