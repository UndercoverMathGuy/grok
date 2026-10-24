import numpy as np
from p52lib import *
from hs_cert import B_HS
s=1.5; G=Gamma(s); cR=3**(2-2*s)/4; P=3**(1-s)
for al in [0.0, 0.2, 0.4, 0.6, 0.8, 1.2, -0.3]:
    A=np.exp(1j*al); bT=[1.0,1.0,np.exp(-1j*al)]
    best=(0,None)
    for R in np.concatenate([[0],np.geomspace(0.01,6,30)]):
        b=B_HS(A,bT,R); v=b/(G*np.sqrt(P**2+cR*R**2))
        if v>best[0]: best=(v,R)
    print(f"single atom alpha={al}: HS ratio {best[0]:.4f} at R={best[1]:.3f}", flush=True)
