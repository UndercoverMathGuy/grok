# Claim (E_r), n=2: Psi(M) := sum_j |lam_j|^r cos(r arg lam_j) >= 3 m^r,  m = min_{T^2} Re P.
# ELEGANT: sum_l (3|c_l|)^r >= Psi(M) (Cauchy-Binet + Hadamard + Bernstein). So (E_r) => (L).
# Random + adversarial (Nelder-Mead minimising Psi/(3 m^r)).
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(4); r=2/3
G=90; th=np.linspace(0,2*np.pi,G,endpoint=False); S,T=np.meshgrid(np.exp(1j*th),np.exp(1j*th),indexing='ij'); S=S.ravel(); T=T.ravel()
def unpack(z,N):
    c=z[:N]*np.exp(1j*z[N:2*N]); v=np.exp(1j*z[2*N:3*N]); w=np.exp(1j*z[3*N:4*N]); return c,v,w
def score(z,N):
    c,v,w=unpack(z,N)
    m=((1+np.outer(S,v))*(1+np.outer(T,w))*c).sum(1).real.min()
    if m<=1e-9: return 10.0
    ml=np.stack([np.ones(N),v,v*w],1); M=np.einsum('l,li,lj->ij',c,ml,ml.conj())
    lam=np.linalg.eigvals(M); psi=(np.abs(lam)**r*np.cos(r*np.angle(lam))).sum()
    Lrat=(np.abs(3*c)**r).sum()/(3*m**r)      # (L) ratio, must be >=1 too
    return psi/(3*m**r)
worst=np.inf
for trial in range(300):
    N=int(rng.integers(3,8))
    base=np.r_[np.ones(3)/3,0.1*rng.random(N-3)]; om=2*np.pi*np.arange(N)/3
    z=np.concatenate([base*(1+rng.normal(0,0.3,N)).clip(0.05),rng.normal(0,0.3,N),om+rng.normal(0,0.6,N),om+rng.normal(0,0.6,N)])
    z=minimize(lambda z:score(z,N),z,method='Nelder-Mead',options={'maxiter':3000}).x
    s=score(z,N)
    if s<worst:
        worst=s; c,v,w=unpack(z,N); print(f"trial {trial} N={N}: Psi/(3m^r) = {s:.6f}   (L)-ratio sum(3|c|)^r/(3m^r) = {(np.abs(3*c)**r).sum()/(3*((((1+np.outer(S,v))*(1+np.outer(T,w))*c).sum(1).real.min()))**r):.6f}",flush=True)
print("min Psi/(3 m^r) found:",worst)
