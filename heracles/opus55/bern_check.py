import numpy as np
from scipy.integrate import quad
r=2/3; Cr=r*np.sin(np.pi*r)/np.pi
for R_,th in [(1,0),(0.7,1.0),(0.3,2.0),(2,2.9)]:
    rho=R_*np.exp(1j*th)
    num=2*Cr*quad(lambda u: np.log(abs(1+np.exp(u)*rho))*np.exp(-r*u),-80,80,limit=800)[0]
    print(f"R={R_} th={th}: numeric {num:.10f}   2 R^r cos(r th) = {2*R_**r*np.cos(r*th):.10f}")
