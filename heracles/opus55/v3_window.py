# True worst V3 ratio restricted to junk level delta = max_T |J_T|/|G| in a window.
# ratio = (|G| - max|J|/2) / ((sum|c|^r)/3^{1-r})^{1/r}; penalise leaving the window.
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(9); r=2/3
def parts(z,N):
    c=z[:N]*np.exp(1j*z[N:2*N]); v=np.exp(1j*z[2*N:3*N]); w=np.exp(1j*z[3*N:4*N])
    G=c.sum(); J=np.abs([(c*v).sum(),(c*w).sum(),(c*v*w).sum()]); rhs=((np.abs(c)**r).sum()/3**(1-r))**(1/r)
    return abs(G),J.max(),rhs
for lo,hi in [(0.0,0.1),(0.1,0.2),(0.2,0.25),(0.25,0.3),(0.3,0.35),(0.35,0.5)]:
    best=-np.inf
    for trial in range(500):
        N=int(rng.integers(3,9))
        base=np.r_[np.ones(3)/3,0.1*rng.random(N-3)]; om=2*np.pi*np.arange(N)/3
        z=np.r_[base*(1+rng.normal(0,.4,N)).clip(.05),rng.normal(0,.4,N),om+rng.normal(0,.8,N),om+rng.normal(0,.8,N)]
        def f(z):
            G,d,rhs=parts(z,N); d/=G
            pen=max(0,lo-d)+max(0,d-hi)
            return -(G*(1-d/2))/rhs+50*pen
        z=minimize(f,z,method='Nelder-Mead',options={'maxiter':4000}).x
        G,d,rhs=parts(z,N)
        if lo-1e-4<=d/G<=hi+1e-4: best=max(best,G*(1-d/G/2)/rhs)
    print(f"delta in [{lo},{hi}]: max true V3 ratio {best:.4f}",flush=True)
