import numpy as np
from p52lib import *
s=1.5; G=Gamma(s)
best=(0,None)
K=48
vals=np.zeros((K,K))
for i in range(K):
    for j in range(K):
        xi=np.array([[np.exp(2j*np.pi*i/K),np.exp(2j*np.pi*j/K)]])
        A,bT=own_from_atoms(np.array([1.0+0j]),xi)
        v=h_free(A,bT)/(G*3**(1-s))
        vals[i,j]=v
        if v>best[0]: best=(v,(i,j))
print("max",best, "min",vals.min())
# phase of beta too
for ph in np.linspace(-0.3,0.3,7):
    i,j=best[1]
    xi=np.array([[np.exp(2j*np.pi*i/K),np.exp(2j*np.pi*j/K)]])
    A,bT=own_from_atoms(np.array([np.exp(1j*ph)]),xi)
    print(ph, h_free(A,bT)/(G*3**(1-s)))
np.save('scan_xi.npy',vals)
