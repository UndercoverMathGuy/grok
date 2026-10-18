import numpy as np
from relax3 import Spikes
p,n,b=3,3,2.0; s=(n+1)/b
kap=0.27306; N=27
for pts,conj in [([(2,2,2),(1,1,1)],False),([(2,2,2),(1,1,1)],True)]:
    Sp=Spikes(p,n,s,pts)
    v0=np.array([1.0+0j])
    w=np.array([[27*kap*np.exp(-1j*np.pi/3)],[27*kap*np.exp(1j*np.pi/3)]])
    if conj: w=np.conj(w)
    U=np.zeros((1,len(Sp.up)),dtype=complex)
    th=Sp.pack(v0,w,U)
    print('start ratio (U=0)',Sp.ratio(th))
    val,th2=Sp.optimise(th)
    print('conj',conj,'optimised ratio',val)
