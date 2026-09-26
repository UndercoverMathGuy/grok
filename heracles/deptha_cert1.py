"""Certificate scan v1 (n=2): chain-matrix/mirror bound + one-atom peel + S^r.  Normalised |A0|=1.
usage: deptha_cert1.py A r [grid]"""
import numpy as np, sys
A=float(sys.argv[1]); r=float(sys.argv[2]); G=int(sys.argv[3]) if len(sys.argv)>3 else 300
n=2; k0=3**((1-r)/r)
def LB(S,e2,e3):
    e2=max(e2,0); e3=max(e3,0)
    co=[e3,e2,S,1.0]
    while co[0]==0: co=co[1:]
    y=-1/np.roots(co)
    return float(np.sum((y.astype(complex)**r).real))
def peel(S,mu):
    b=(mu-S*(1-A)**n)/((1+A)**n-(1-A)**n)
    b=min(max(b,0),S/2)
    return (S-b)**r+b**r
worst=(9,None)
for d in np.linspace(0.0,1-1/k0,G):
    mu=1-d; d1=d/A; d12=d/A**2
    for S in (np.linspace(1,k0*mu,G) if k0*mu>1 else [1.0]):
        w=np.sqrt(S*S-1)
        u1=min(S,d1+2*w); u12=min(S,d12+2*w)
        E2=3-2*d1*u1-d12*u12
        E3=1-2*d1*u1-d12*u12-d1*d1*u12-u1*u1*d12
        val=max(LB(S,E2/9,E3/27),S**r,peel(S,mu))/(3**(1-r)*mu**r)
        if val<worst[0]: worst=(val,(d,S,E2,E3))
print(A,r,worst)
