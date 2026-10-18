import numpy as np
from freeatoms import FA
p,n,b=3,3,2.0; s=(n+1)/b
A=FA(p,n,s,[1],{1:3})
al=2*np.pi*0.2324
b0=0.37468; bp=0.44919+0.10100j
for name,beta,xi in [('as given',np.array([b0,bp,np.conj(bp)]),np.array([[1,1,1],[np.exp(1j*al)]*3,[np.exp(-1j*al)]*3])),
                     ('conj phases',np.array([b0,bp,np.conj(bp)]),np.conj(np.array([[1,1,1],[np.exp(1j*al)]*3,[np.exp(-1j*al)]*3]))),
                     ('conj beta',np.conj(np.array([b0,bp,np.conj(bp)])),np.array([[1,1,1],[np.exp(1j*al)]*3,[np.exp(-1j*al)]*3]))]:
    ph=np.angle(xi)
    th=A.pack({1:(beta.astype(complex),ph)})
    r,h,Q,g=A.ratio_and_grad(th)
    print(f"{name}: ratio {r:.6f}  h {h:.6f}  Q {Q:.6f}  (barrier claims 1/(nu^(1-s)Q^s) = {1/(4**(-1)*Q**2):.6f})")
    val,th2=A.optimise(th)
    print("   after local optimisation:",round(val,6))
