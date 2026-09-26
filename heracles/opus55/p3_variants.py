# Simplified sufficient forms of (p3), all with r=2/3, claim ratio <= 1:
#  V1: b1 = ReG - |ImG|/sqrt3 wedge dropped -> b1 = min(ReG - max_T h(J_T))       (h(J)=max_{zeta in mu3} Re(J zeta))
#  V2: b2 = ReG - max_T |J_T|/2
#  V3: b3 = |G| - max_T |J_T|/2
#  V4: b4 = ReG - mean_T |J_T|/2      (stronger than V2)
#  V5: b5 = ReG - sum_T |J_T|/4       (stronger still?)
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(5); r=2/3; mu3=np.exp(2j*np.pi*np.arange(3)/3)
def data(z,N):
    c=z[:N]*np.exp(1j*z[N:2*N]); v=np.exp(1j*z[2*N:3*N]); w=np.exp(1j*z[3*N:4*N])
    G=c.sum(); J=np.array([(c*v).sum(),(c*w).sum(),(c*v*w).sum()]); rhs=((np.abs(c)**r).sum()/3**(1-r))**(1/r)
    return G,J,rhs
h=lambda J: max((J*z).real for z in mu3)
V={'V1':lambda G,J: G.real-max(h(j) for j in J),
   'V2':lambda G,J: G.real-np.abs(J).max()/2,
   'V3':lambda G,J: abs(G)-np.abs(J).max()/2,
   'V4':lambda G,J: G.real-np.abs(J).mean()/2,
   'V5':lambda G,J: G.real-np.abs(J).sum()/4}
for name,fb in V.items():
    best=-np.inf
    for trial in range(900):
        N=int(rng.integers(1,9)); kind=trial%3
        if kind==0:
            base=np.r_[np.ones(3)/3,0.1*rng.random(max(N-3,0))][:max(N,3)]; N=len(base); om=2*np.pi*np.arange(N)/3
            z=np.r_[base*(1+rng.normal(0,.3,N)).clip(.05),rng.normal(0,.3,N),om+rng.normal(0,.6,N),om+rng.normal(0,.6,N)]
        elif kind==1: z=np.r_[rng.random(N)+.1,rng.normal(0,.5,N),2*np.pi*rng.random(2*N)]
        else: z=np.r_[rng.random(N)*2+.1,2*np.pi*rng.random(3*N)]
        f=lambda z: -fb(*data(z,N)[:2])/data(z,N)[2]
        z=minimize(f,z,method='Nelder-Mead',options={'maxiter':3000}).x
        best=max(best,-f(z))
    print(f"{name}: max ratio {best:.6f}  -> {'holds' if best<=1+1e-6 else 'FAILS'}",flush=True)
