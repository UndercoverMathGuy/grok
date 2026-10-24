import numpy as np
from fb import gstar, funcs
s=1.5
for rho in [0.4,0.45,0.5]:
    rng=np.random.default_rng(7)
    g,N,v=gstar(s,(rho,rho,rho),rng,trials=25)
    am,ph,t1,t2=v[:N],v[N:2*N],v[2*N:3*N],v[3*N:]
    b=am*np.exp(1j*ph)
    BT=[b@np.exp(1j*t1),b@np.exp(1j*t2),b@np.exp(1j*(t1+t2))]
    print(rho,g,N,'amps',np.round(am,4),'A',np.round(b.sum(),4),'|BT|',np.round(np.abs(BT),4),'t1',np.round(np.mod(t1-t1[0],2*np.pi),3),'t2',np.round(np.mod(t2-t2[0],2*np.pi),3))
