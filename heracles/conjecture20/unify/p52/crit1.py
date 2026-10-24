import numpy as np
from p52lib import *
from twisted_pay import twisted_lb
from tw2 import pay_tw2
from hs_elim import elim_ratio
from l2relax import h_L2
s=1.5; G=Gamma(s); cR=3**(2-2*s)/4
al,m1,m2,m12,phi = 0.0,0.555,0.582,0.856,0.269
A=np.exp(1j*al); B=m1; C=m2; D=m12*np.exp(1j*phi)
full=twisted_lb(A,B,C,D,s,tries=20)
dec=pay_tw2(1.0,m1,m2,m12,np.cos(phi+al),s,nL=400)
c1=1-3**(1-s); M=max(m1,m2,m12); RT=np.sqrt(m1**2+m2**2+m12**2)
plb=max(1-c1*M,1-RT/(2*np.sqrt(3)),3**(1-s)*max(1,M))
print("payments: P_lb",plb," decoupled tw",dec," full tw",full)
for F in [dec, full]:
    print(" HS elim ratio with F=%.4f: %.4f"%(F, elim_ratio(al,m1,m2,m12,phi,F)))
    best=0
    for R in np.concatenate([[0],np.geomspace(0.01,4,25)]):
        best=max(best, h_L2(A,[B,C,D],R)/(G*np.sqrt(F**2+cR*R**2)))
    print(" full SOCP (L2) ratio with F=%.4f: %.4f"%(F,best))
