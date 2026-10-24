# Check the two quadratic payment constants (n=2) by adversarial search over atom systems.
#  general atoms:   bonus = |A| - 3^{1-s} Q^s <= K2 * (|B|^2+|C|^2+|D|^2)/L,   K2 = 2(s-1)/(7-4s)
#  relations case (B=C=0): bonus <= Kq * |D|^2/L,  Kq = 9/(10+8r)
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(0)
def unpack(z,m):
    rho=np.exp(z[:m]); p=np.exp(1j*z[m:2*m]); xi=np.exp(1j*z[2*m:3*m]); eta=np.exp(1j*z[3*m:4*m]); return rho,p,xi,eta
def stats(z,m,s):
    rho,p,xi,eta=unpack(z,m); b=rho*p; r=1/s
    A=b.sum(); B=(b*xi).sum(); C=(b*eta).sum(); D=(b*xi*eta).sum(); L=rho.sum(); Q=(rho**r).sum()
    return A,B,C,D,L,Q
def ratio_gen(z,m,s):
    A,B,C,D,L,Q=stats(z,m,s); bonus=abs(A)-3**(1-s)*Q**s; E=abs(B)**2+abs(C)**2+abs(D)**2
    return -bonus*L/max(E,1e-300)
def ratio_rel(z,m,s,pen=1e4):
    A,B,C,D,L,Q=stats(z,m,s); bonus=abs(A)-3**(1-s)*Q**s
    # enforce B=C=0 by penalty on the scale-free quantities
    return -bonus*L/max(abs(D)**2,1e-300) + pen*(abs(B)**2+abs(C)**2)/L**2
for s in [1.25,1.5]:
    r=1/s; K2=2*(s-1)/(7-4*s); Kq=9/(10+8*r)
    bg=-np.inf; br=-np.inf
    for t in range(400):
        m=rng.integers(1,7); z=rng.normal(size=4*m)
        o=minimize(ratio_gen,z,args=(m,s),method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-10,fatol=1e-12))
        bg=max(bg,-o.fun)
        if m>=2:
            o=minimize(ratio_rel,z,args=(m,s),method='Nelder-Mead',options=dict(maxiter=6000,xatol=1e-10,fatol=1e-12))
            A,B,C,D,L,Q=stats(o.x,m,s)
            if (abs(B)+abs(C))/L<1e-4: br=max(br,(abs(A)-3**(1-s)*Q**s)*L/abs(D)**2)
    print(f"s={s}: general sup bonus*L/E ~ {bg:.4f} (claimed bound K2={K2:.4f});  relations sup bonus*L/|D|^2 ~ {br:.4f} (claimed Kq={Kq:.4f})")
