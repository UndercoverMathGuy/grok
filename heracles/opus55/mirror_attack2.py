# Attack 2: c = c_block + beta*d + eps*e, where d kills clean+all junk (sum d = sum d v = sum d w = sum d v w = 0)
# but has nonzero mirrors (sum d vbar, ...), and e injects small junk. Minimize Psi/(3 m^r) over beta, phases.
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space
r=2/3; rng=np.random.default_rng(7); om=np.exp(2j*np.pi/3)
G=100; th=np.linspace(0,2*np.pi,G,endpoint=False); S,T=np.meshgrid(np.exp(1j*th),np.exp(1j*th),indexing='ij'); S=S.ravel(); T=T.ravel()
def stats(c,v,w):
    m=((1+np.outer(S,v))*(1+np.outer(T,w))*c).sum(1).real.min()
    ml=np.stack([np.ones(len(c)),v,v*w],1); M=np.einsum('l,li,lj->ij',c,ml,ml.conj())
    lam=np.linalg.eigvals(M); psi=(np.abs(lam)**r*np.cos(r*np.angle(lam))).sum()
    return m,lam,psi
worst=(np.inf,)
for trial in range(400):
    N=int(rng.integers(5,9))
    v=np.r_[om**np.arange(3),np.exp(2j*np.pi*rng.random(N-3))]; w=np.r_[om**np.arange(3),np.exp(2j*np.pi*rng.random(N-3))]
    c0=np.r_[np.ones(3)/3,np.zeros(N-3)]
    K=null_space(np.stack([np.ones(N),v,w,v*w]))          # N x (N-4) complex
    def f(z):
        d=K@(z[:K.shape[1]]+1j*z[K.shape[1]:2*K.shape[1]])
        e=z[2*K.shape[1]:2*K.shape[1]+N]+1j*z[2*K.shape[1]+N:]
        c=c0+d+e
        m,lam,psi=stats(c,v,w)
        if m<=1e-6: return 10.0
        return psi/(3*m**r)
    z0=np.r_[rng.normal(0,rng.choice([0.3,1,3]),2*K.shape[1]),rng.normal(0,0.02,2*N)]
    z=minimize(f,z0,method='Nelder-Mead',options={'maxiter':4000}).x
    val=f(z)
    if val<worst[0]:
        d=K@(z[:K.shape[1]]+1j*z[K.shape[1]:2*K.shape[1]]); e=z[2*K.shape[1]:2*K.shape[1]+N]+1j*z[2*K.shape[1]+N:]
        c=c0+d+e; m,lam,psi=stats(c,v,w)
        worst=(val,N,m,np.abs(d).max(),np.round(lam,3),(np.abs(3*c)**r).sum()/(3*m**r))
        print(f"trial {trial}: Psi/(3m^r)={val:.5f}  N={N}  m={m:.4f}  max|d|={worst[3]:.3f}  eig={worst[4]}  (L)-ratio={worst[5]:.3f}",flush=True)
