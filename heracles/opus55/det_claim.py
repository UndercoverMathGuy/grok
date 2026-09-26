# Pointwise (r-free) claim (D), n=2:  |det(I + tau M)| >= (1 + tau m)^3 for all tau >= 0,
# M = sum_l c_l m_l m_l^*, m_l=(1,v_l,v_l w_l), m = min_{T^2} Re P.  (D) => (E_r) for every r => (L).
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(5)
G=90; th=np.linspace(0,2*np.pi,G,endpoint=False); S,T=np.meshgrid(np.exp(1j*th),np.exp(1j*th),indexing='ij'); S=S.ravel(); T=T.ravel()
taus=np.logspace(-3,4,240)
def score(z,N):
    c=z[:N]*np.exp(1j*z[N:2*N]); v=np.exp(1j*z[2*N:3*N]); w=np.exp(1j*z[3*N:4*N])
    m=((1+np.outer(S,v))*(1+np.outer(T,w))*c).sum(1).real.min()
    if m<=1e-9: return 10.0,None
    ml=np.stack([np.ones(N),v,v*w],1); M=np.einsum('l,li,lj->ij',c,ml,ml.conj())
    lam=np.linalg.eigvals(M)
    f=np.log(np.abs(np.prod(1+np.outer(taus,lam),1)))-3*np.log1p(taus*m)
    q=f/np.log1p(taus*m)                     # >=0 iff claim (D); scale-free
    i=np.argmin(q); return q[i],taus[i]
worst=np.inf
for trial in range(300):
    N=int(rng.integers(3,8))
    base=np.r_[np.ones(3)/3,0.1*rng.random(N-3)]; om=2*np.pi*np.arange(N)/3
    z=np.concatenate([base*(1+rng.normal(0,0.3,N)).clip(0.05),rng.normal(0,0.3,N),om+rng.normal(0,0.6,N),om+rng.normal(0,0.6,N)])
    z=minimize(lambda z:score(z,N)[0],z,method='Nelder-Mead',options={'maxiter':3000}).x
    s,tau=score(z,N)
    if s<worst: worst=s; print(f"trial {trial} N={N}: min_tau [log|det(I+tau M)| - 3 log(1+tau m)]/log(1+tau m) = {s:.6f} at tau={tau}",flush=True)
print("min found:",worst)
