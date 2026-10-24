import time, numpy as np
from hs2 import elim_ratio
from cert_hs import cert_data, eval_cells
from iv import dn
F=3**-0.5
t0=time.time(); val,Gs=elim_ratio(0.0,1.0,1.0,1.0,0.0,F,deg=2,ret=True); t1=time.time()
cd=cert_data(Gs); t2=time.time()
r=eval_cells(cd,np.array([0.0]),np.array([0.0]),np.array([[1.0,1,1]]),np.array([[1.0,1,1]]),np.array([0.0]),np.array([0.0]),np.array([dn(F)]))
print("SDP",round(t1-t0,2),"s  rigorise",round(t2-t1,3),"s  ratio",val,r)
t0=time.time()
for k in range(5):
    val,Gs=elim_ratio(0.01*k,0.9,0.9,0.95,0.05,0.62,deg=2,ret=True)
print("5 more SDPs", round(time.time()-t0,2))
