# Direction-resolved constant used by Corollary cor:quad after averaging +-delta:
#   kR(n,s) = sup_{atoms, alpha in [0,pi/2)} (Re A - nu^{1-s}Q^s) cos(alpha) / (sigma * (f(g+)+f(g-))/2),
#   g+- = (3/4) Re(tau e^{+-i alpha}) / sigma^3,  alpha = arg c_l(delta) (kappa = 2 cos^2 alpha).
# kR < 1  <=>  payment <= detection in every direction pair +-delta.
import numpy as np, sys
from scipy.optimize import minimize
n=int(sys.argv[1]); s=float(sys.argv[2]) if len(sys.argv)>2 else (n+1)/2; nu=n+1; r=1/s
masks=list(range(1,2**n)); V=np.array([[(m>>t)&1 for t in range(n)] for m in masks]); ix={m:i for i,m in enumerate(masks)}
trip=np.array([(ix[a],ix[b],ix[a|b]) for a in masks for b in masks if a&b==0])
f=lambda g:(np.sqrt(g*g+4)-g)/2
def val(v,N):
    b=np.abs(v[:N])*np.exp(1j*v[N:2*N]); th=v[2*N:2*N+N*n].reshape(N,n); al=(np.pi/2)*np.sin(v[-1])**2*0.999
    A=b.sum(); BT=np.exp(1j*(th@V.T)).T@b
    bonus=A.real-nu**(1-s)*np.sum(np.abs(b)**r)**s
    sig=np.sqrt(0.5*np.sum(np.abs(BT)**2))
    if sig<1e-9: return -9
    tau=np.sum(BT[trip[:,0]]*BT[trip[:,1]]*np.conj(BT[trip[:,2]]))
    gp=0.75*np.real(tau*np.exp(1j*al))/sig**3; gm=0.75*np.real(tau*np.exp(-1j*al))/sig**3
    return bonus*np.cos(al)/(sig*0.5*(f(gp)+f(gm)))
rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 1); best=-9; arg=None
for N in range(2,nu+3):
    for t in range(25):
        v0=np.concatenate([rng.uniform(0.1,1,N),rng.uniform(0,2*np.pi,N+N*n),[rng.uniform(0,1)]])
        res=minimize(lambda v:-val(v,N),v0,method='Nelder-Mead',options={'maxiter':10000,'xatol':1e-10,'fatol':1e-13})
        if -res.fun>best: best=-res.fun; arg=(N,res.x[-1])
print(f"n={n} s={s}: kR >= {best:.4f}  (N={arg[0]}, alpha={(np.pi/2)*np.sin(arg[1])**2*0.999:.3f})",flush=True)
