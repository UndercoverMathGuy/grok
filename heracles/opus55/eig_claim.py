# Claim (E): for chain matrix M = sum_l c_l m_l m_l^*, m_l = (1, v_l, v_l w_l)  (n=2),
# every eigenvalue lambda of M has Re(lambda) >= m := min over T^2 of Re P(s,t), P = sum c_l (1+v_l s)(1+w_l t).
# If true, |det(I+tau M)| >= (1+tau m)^3 and (L) follows from ELEGANT's Cauchy-Binet/Bernstein argument.
import numpy as np
rng=np.random.default_rng(3)
G=120; th=np.linspace(0,2*np.pi,G,endpoint=False); S,T=np.meshgrid(np.exp(1j*th),np.exp(1j*th),indexing='ij'); S=S.ravel(); T=T.ravel()
worst=np.inf; cnt=0; ex=None
for trial in range(20000):
    N=rng.integers(1,7)
    mode=rng.integers(3)
    if mode==0:   # near block + extras
        c=np.r_[np.ones(3)/3,0.1*rng.random(max(N-3,0))][:max(N,3)]; N=len(c)
        c=c*(1+rng.normal(0,rng.choice([0.02,0.2,0.6]),N))*np.exp(1j*rng.normal(0,0.3,N))
        v=np.exp(1j*(2*np.pi*np.arange(N)/3+rng.normal(0,0.5,N))); w=np.exp(1j*(2*np.pi*np.arange(N)/3+rng.normal(0,0.5,N)))
    else:
        c=rng.random(N)*np.exp(1j*rng.normal(0,0.4,N)); v=np.exp(2j*np.pi*rng.random(N)); w=np.exp(2j*np.pi*rng.random(N))
    P=((1+np.outer(S,v))*(1+np.outer(T,w))*c).sum(1).real; m=P.min()
    if m<=1e-9: continue
    cnt+=1
    ml=np.stack([np.ones(N),v,v*w],1)                     # N x 3
    M=np.einsum('l,li,lj->ij',c,ml,ml.conj())
    lam=np.linalg.eigvals(M); gap=lam.real.min()-m
    if gap<worst: worst=gap; ex=(N,m,lam)
print(f"feasible populations tested: {cnt};  min over them of [min Re(lambda) - m] = {worst:.3e}")
print("worst case: N, m, eigenvalues =",ex)
