# check: bonus ~ (1-r)/(8+r) |beta_1+beta_2+conj(beta_12)|^2 / a  near the 3-atom IF block (xi_i=(w^i,w^i))
import numpy as np
rng=np.random.default_rng(1); w=np.exp(2j*np.pi/3)
for s in (1.5,1.2):
    r=1/s; rat=[]
    for t in range(2000):
        eps=1e-3
        x=eps*rng.standard_normal(3); y=eps*rng.standard_normal(3); th=eps*rng.standard_normal(3); ph=eps*rng.standard_normal(3)
        b=(1/3)*(1+x)*np.exp(1j*y); i=np.arange(3)
        A=b.sum(); a=3**(1-s)*np.sum(np.abs(b)**r)**s
        # remove global phase: bonus measured with |A|
        bon=abs(A)-a
        b1=b@(w**i*np.exp(1j*th)); b2=b@(w**i*np.exp(1j*ph)); b12=b@(w**(2*i)*np.exp(1j*(th+ph)))
        Sb=b1+b2+np.conj(b12)*np.exp(2j*np.angle(A))  # gauge with global phase of A
        rat.append(bon*a/abs(Sb)**2)
    print(s, 'max ratio', max(rat), ' predicted (1-r)/(8+r) =',(1-r)/(8+r))
