import numpy as np, sys, time
sys.path.insert(0,'..')
from gget import gamma_table
from relmodel import *
p=31; s=1.5
T=gamma_table()[p]; G,S,y=T['Gamma'],T['S'],np.array(T['y']); print(G,S,y)
for c in [0.1835,0.2887,0.6]:
    M=Model(p,s,c,G)
    a=np.zeros(M.F); a[np.array(S)-1]=y
    print('IF ratio',M.ratio(a,0*a,0*a))
    rng=np.random.default_rng(0); best=-9
    t=time.time()
    for trial in range(10):
        if trial<5: a0=a+1e-3*rng.random(M.F)*a.max()
        else: a0=rng.exponential(1,M.F)
        v0=M.cap*a0*rng.random(M.F); th0=rng.uniform(0,2*np.pi,M.F)
        aa,vv,tt=optimise(M,a0,v0,th0)
        R=M.ratio(aa,vv,tt)
        R2,a2,v2=lp_polish(M,aa,vv,tt)
        inf=M.info(a2,v2,tt)
        print(c,trial,round(R,4),round(R2,4),inf,flush=True)
    print(time.time()-t)
