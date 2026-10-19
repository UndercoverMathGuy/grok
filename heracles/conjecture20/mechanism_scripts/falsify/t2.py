import numpy as np, sys, time
from gget import gamma_table
from relmodel import *
p=int(sys.argv[1]); s=1.5
T=gamma_table()[p]; G,S,y=T['Gamma'],T['S'],np.array(T['y']); print(p,G,S,y)
M=Model(p,s,0.1835,G)
a=np.zeros(M.F); a[np.array(S)-1]=y
rng=np.random.default_rng(0); t=time.time()
for trial in range(12):
    if trial<4: a0=a+1e-2*rng.random(M.F)*a.max(); mask=None
    elif trial<8: a0=a.copy()+1e-9; mask=(a>0).astype(float)
    else: a0=rng.exponential(1,M.F); mask=None
    v0=M.cap*a0*rng.uniform(0.5,1,M.F); th0=rng.uniform(0,2*np.pi,M.F)
    aa,vv,tt=optimise_crit(M,a0,v0,th0,mask=mask)
    cs,d=cstar(M,aa,vv,tt)
    M.c=0.1835; r1=M.ratio(aa,vv,tt); M.c=0.2887; r2=M.ratio(aa,vv,tt)
    print(trial,'c*=',round(cs,4),'delta',d,'supp',np.sum(aa>1e-6*aa.max()),'cheap',np.sum(vv>1e-3*aa.max()),'ratio@.1835',round(r1,4),'@.2887',round(r2,4),f'{time.time()-t:.0f}s',flush=True)
