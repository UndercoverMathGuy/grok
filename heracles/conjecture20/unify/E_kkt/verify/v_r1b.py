import numpy as np
from v_r1 import local_max, kkt_lp
from vlib import *
p,n,b,m=7,2,2.5,7
nu=n+1; s=nu/b; X,Y=grid(p,n)
best=None
for seed in range(4):
    U,W=local_max(p,n,b,m,seed)
    G=gaps(U,W,p,n,X,Y); P,sq=cost(U,W,b); gam=G.min()/P**s
    print(seed, gam, np.sort(sq/sq.max()))
    if best is None or gam>best[0]: best=(gam,U,W)
gam,U,W=best
_,sq=cost(U,W,b)
for thr in [1e-10,1e-6,1e-3]:
    keep=sq>thr*sq.max(); U2,W2=U[keep],W[keep]
    Q,resid,h,P=kkt_lp(U2,W2,p,n,b)
    _,sq2=cost(U2,W2,b); wsh=sq2**(b/2)/P
    M=dual_tensor(Q,p,n,X,Y)
    t=[]
    for i in range(len(sq2)):
        v=[U2[i,tt]/np.linalg.norm(U2[i,tt]) for tt in range(n)]+[W2[i]/np.linalg.norm(W2[i])]
        t.append(float(contract(M,v)))
    t=np.array(t); law=t/(nu**(nu/2)*(h/P**s)*wsh**(1-s))-1
    print(thr,len(sq2),resid,'gamma',h/P**s,'law per neuron',np.round(law,8),'shares',np.round(wsh,5))
