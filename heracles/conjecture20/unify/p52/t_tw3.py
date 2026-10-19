import numpy as np, time
from tw3_rig import tw3_bound
m=np.array([[0.524,0.497,0.635],[0.628,0.628,0.628],[1,1,1]]); ph=np.array([0.117,0.022,0.0])
for nL,npsi in [(40,36),(80,48),(160,72),(320,96)]:
    t0=time.time(); v=tw3_bound(m,m,ph,ph,nL=nL,npsi=npsi); print(nL,npsi,np.round(v,4),round(time.time()-t0,2))
