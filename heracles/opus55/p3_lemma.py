# (p3) lemma: for a single-frequency population (c_l, v_l, w_l):
#   b := min{ Re G - |Im G|/sqrt3 ,  min_{T in {1},{2},{12}} min_{zeta in mu_3} Re(G - J_T zeta) }
#   claim  b <= ( sum|c_l|^r / 3^{1-r} )^{1/r},   r = 2/3   (ratio b/RHS <= 1; equality at the 3-atom block)
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(2); r=2/3; mu3=np.exp(2j*np.pi*np.arange(3)/3)
def b_of(c,v,w):
    G=c.sum(); J=[(c*v).sum(),(c*w).sum(),(c*v*w).sum()]
    vals=[G.real-abs(G.imag)/np.sqrt(3)]+[min((G-JT*z).real for z in mu3) for JT in J]
    return min(vals)
def ratio(z,N):
    c=z[:N]*np.exp(1j*z[N:2*N]); v=np.exp(1j*z[2*N:3*N]); w=np.exp(1j*z[3*N:4*N])
    return b_of(c,v,w)/((np.abs(c)**r).sum()/3**(1-r))**(1/r)
best=-np.inf
for trial in range(3000):
    N=int(rng.integers(1,9)); kind=trial%3
    if kind==0:
        base=np.r_[np.ones(3)/3,0.1*rng.random(max(N-3,0))][:max(N,3)]; N=len(base); om=2*np.pi*np.arange(N)/3
        z=np.r_[base*(1+rng.normal(0,.3,N)).clip(.05),rng.normal(0,.3,N),om+rng.normal(0,.6,N),om+rng.normal(0,.6,N)]
    elif kind==1: z=np.r_[rng.random(N)+.1,rng.normal(0,.5,N),2*np.pi*rng.random(2*N)]
    else: z=np.r_[rng.random(N)*2+.1,2*np.pi*rng.random(3*N)]
    z=minimize(lambda z:-ratio(z,N),z,method='Nelder-Mead',options={'maxiter':3000}).x
    val=ratio(z,N)
    if val>best: best=val; print(f"trial {trial} N={N} kind={kind}: b/RHS = {val:.7f}",flush=True)
print("max b/RHS found:",best)
