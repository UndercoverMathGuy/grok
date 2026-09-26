# Deliberate attack on (D)/(E_r): big mirror (upper-triangle) entries times small junk.
# Population on block phases v_l=w_l=omega^l: c = alpha*(1,1,1) + beta*(1,omega,omega^2) [+ eps-perturbation].
import numpy as np
r=2/3; om=np.exp(2j*np.pi/3)
G=120; th=np.linspace(0,2*np.pi,G,endpoint=False); S,T=np.meshgrid(np.exp(1j*th),np.exp(1j*th),indexing='ij'); S=S.ravel(); T=T.ravel()
def stats(c,v,w):
    m=((1+np.outer(S,v))*(1+np.outer(T,w))*c).sum(1).real.min()
    ml=np.stack([np.ones(len(c)),v,v*w],1); M=np.einsum('l,li,lj->ij',c,ml,ml.conj())
    lam=np.linalg.eigvals(M); psi=(np.abs(lam)**r*np.cos(r*np.angle(lam))).sum()
    return m,M,lam,psi
v=om**np.arange(3); 
for beta in (0,1,3,10,30):
    for eps in (0.0,0.05,0.2):
        best=None
        for phase in np.linspace(0,2*np.pi,24,endpoint=False):
          for ph2 in np.linspace(0,2*np.pi,12,endpoint=False):
            c=np.array([1,1,1])/3+beta*np.exp(1j*phase)*np.array([1,om,om**2])/3
            w=v*np.exp(1j*eps*np.array([0,1,-1])*np.exp(1j*ph2).real)   # small junk via w-phase perturbation
            m,M,lam,psi=stats(c,v,w)
            if m<=1e-9: continue
            q=psi/(3*m**r); L=(np.abs(3*c)**r).sum()/(3*m**r)
            if best is None or q<best[0]: best=(q,L,m,np.round(lam,3))
        print(f"beta={beta:>4} eps={eps}: min Psi/(3m^r)={best[0]:.4f}  (L)-ratio={best[1]:.3f}  m={best[2]:.3f}  eig={best[3]}" if best else f"beta={beta} eps={eps}: no feasible",flush=True)
