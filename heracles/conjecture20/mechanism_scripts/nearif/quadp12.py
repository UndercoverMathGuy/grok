# Is the relations-case bonus <= c12 |D|^2 / L  (sharp at two aligned atoms)?  Adversarial search, B=C=0 by exact projection.
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(7)
def sysfrom(z,m):
    rho=np.exp(z[:m]); p=np.exp(1j*z[m:2*m]); xi=np.exp(1j*z[2*m:3*m]); eta=np.exp(1j*z[3*m:4*m]); return rho*p,xi,eta,rho
def obj(z,m,s,pen):
    b,xi,eta,rho=sysfrom(z,m); A=b.sum(); B=(b*xi).sum(); C=(b*eta).sum(); D=(b*xi*eta).sum(); L=rho.sum(); Q=(rho**(1/s)).sum()
    bonus=abs(A)-3**(1-s)*Q**s
    return -(bonus*L/max(abs(D)**2,1e-30)) + pen*(abs(B)**2+abs(C)**2)/L**2
for s in [1.1,1.25,1.4,1.5]:
    c12=1-(2/3)**(s-1); best=-1
    for t in range(600):
        m=int(rng.integers(2,9)); z=rng.normal(size=4*m)
        for pen in [1e2,1e4,1e6]:
            z=minimize(obj,z,args=(m,s,pen),method='BFGS').x
        b,xi,eta,rho=sysfrom(z,m); B=(b*xi).sum(); C=(b*eta).sum(); L=rho.sum()
        if (abs(B)+abs(C))/L<1e-5:
            A=b.sum(); D=(b*xi*eta).sum(); Q=(rho**(1/s)).sum()
            if abs(D)/L>1e-3: best=max(best,(abs(A)-3**(1-s)*Q**s)*L/abs(D)**2)
    print(f"s={s}: sup bonus*L/|D|^2 = {best:.5f}   c12 = {c12:.5f}",flush=True)
