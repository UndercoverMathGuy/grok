"""First feasibility scan of the chain-matrix + mirror-control certificate at depth A (n=2).
Pi(1+x_l t) >= 1 + S t + eps2 t^2 + eps3 t^3, eps_k = |E_k(M)|/3^k lower bounds,
|L_I| = delta_I <= d/A^{|I|}, |U_I| <= min(S, delta_I + 2 sqrt(S^2-1)), |A0| = 1, mu = 1-d.
LB = sum_i Re(y_i^r), P(t) = prod(1+y_i t). Target 3^{1-r} mu^r."""
import numpy as np, sys
A=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
r=2/3; k0=3**((1-r)/r)
def LB(S,e2,e3):
    e2=max(e2,0); e3=max(e3,0)
    co=[e3,e2,S,1.0]
    while co[0]==0: co=co[1:]
    rt=np.roots(co); y=-1/rt
    return float(np.sum((y.astype(complex)**r).real))
worst=(9,None)
for d in np.linspace(0.0,1-1/k0,200):
    mu=1-d; d1=d/A; d12=d/A**2
    for S in np.linspace(1,k0*mu,200) if k0*mu>1 else [1.0]:
        w=np.sqrt(S*S-1)
        u1=min(S,d1+2*w); u12=min(S,d12+2*w)
        E2=3-2*d1*u1-d12*u12
        E3=1-2*d1*u1-d12*u12-d1*d1*u12-u1*u1*d12
        val=max(LB(S,E2/9,E3/27),S**r)/(3**(1-r)*mu**r)
        if val<worst[0]: worst=(val,(d,S,E2,E3))
print(A,worst)
