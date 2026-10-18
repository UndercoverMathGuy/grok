import sys, numpy as np
from lib import *
from direction import dirobj
p,n,b,l,m,starts=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
nu=n+1; s=nu/b; rng=np.random.default_rng(1); mask=freq_proj(p,1); ka=1-np.cos(2*np.pi*l/p)
dels=[l%p,(-l)%p]; wts=np.array([1/ka]*2); ref=nu**(1-s)*Bp(p,n)
vals=[]
for st in range(starts):
    U=rng.normal(size=(m,n,p)); W=rng.normal(size=(m,p)); x=pack(U,W); nit=0
    for tau in (3e-2,1e-2,3e-3,1e-3,3e-4,1e-4):
        res=minimize(lambda z: tuple(-v for v in dirobj(z,m,n,p,b,tau*ref,dels,wts,mask)),x,jac=True,method='L-BFGS-B',options={'maxiter':4000,'ftol':1e-15,'gtol':1e-12})
        x=res.x; nit+=res.nit
    U,W=unpack(x,m,n,p); W=W@mask; net=Net(U,W,b)
    G=net.gaps()[:,[d-1 for d in dels]]/ka; val=G.min()/net.cost()**s/ref
    A,Q,a=net.cheapness()
    vals.append(val); print(f"start {st}: val {val:.5f} iters {nit} cheap {A[0]/a[0]:.4f} |beta1| {A[0]:.3e}",flush=True)
print("sorted",np.round(sorted(vals),4))
