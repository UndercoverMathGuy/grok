# Ratio forms of (L) at n=2 (analogues of the n=1 step m e1 <= 4 e2):
#   (R1)  m*e1 <= 3*e2      (R2)  m*e2 <= 9*e3        [both tight at the 3-atom block]
# with x_l=|c_l|, m=min_{T^2} Re P.  Adversarial search from block and M_m-type starts.
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(11)
G=72; th=np.linspace(0,2*np.pi,G,endpoint=False); S,T=np.meshgrid(np.exp(1j*th),np.exp(1j*th),indexing='ij'); S=S.ravel(); T=T.ravel()
def es(x):
    p1=x.sum();p2=(x**2).sum();p3=(x**3).sum()
    return p1,(p1**2-p2)/2,(p1**3-3*p1*p2+2*p3)/6
def f(z,N,which):
    c=z[:N]*np.exp(1j*z[N:2*N]); v=np.exp(1j*z[2*N:3*N]); w=np.exp(1j*z[3*N:4*N])
    m=((1+np.outer(S,v))*(1+np.outer(T,w))*c).sum(1).real.min()
    if m<=1e-7: return 10.0
    e1,e2,e3=es(np.abs(c))
    return 3*e2/(m*e1) if which==1 else 9*e3/(m*e2)
for which in (1,2):
    worst=np.inf
    for trial in range(160):
        N=int(rng.integers(3,8)); kind=trial%3
        if kind==0:   # block + extras
            base=np.r_[np.ones(3)/3,0.08*rng.random(N-3)]; om=2*np.pi*np.arange(N)/3
            z=np.r_[base*(1+rng.normal(0,.3,N)).clip(.05),rng.normal(0,.3,N),om+rng.normal(0,.5,N),om+rng.normal(0,.5,N)]
        elif kind==1: # M_2-like: pairs with antipodal phases in slot 1, random elsewhere
            z=np.r_[rng.random(N)+.3,rng.normal(0,.3,N),np.pi*np.arange(N)+rng.normal(0,.4,N),2*np.pi*rng.random(N)]
        else:         # random with large amplitudes (cancellation)
            z=np.r_[rng.random(N)*3+.1,2*np.pi*rng.random(N),2*np.pi*rng.random(N),2*np.pi*rng.random(N)]
        for _ in range(2): z=minimize(lambda z:f(z,N,which),z,method='Nelder-Mead',options={'maxiter':4000}).x
        val=f(z,N,which)
        if val<worst: worst=val; print(f"(R{which}) trial {trial} kind {kind} N={N}: ratio {val:.6f}",flush=True)
    print(f"(R{which}) minimum found {worst:.6f} -> {'VIOLATED' if worst<1-1e-5 else 'holds'}",flush=True)
