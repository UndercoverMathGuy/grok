import numpy as np, sys, time
from gget import gamma_table
from relmodel import *
p=int(sys.argv[1]); s=1.5
G=gamma_table()[p]['Gamma']; M=Model(p,s,0.1835,G)
R=np.load(f'res/run_p{p}.npy',allow_pickle=True).item()
val,name,cfg=R['cstar']; a,v,th=[np.array(x) for x in cfg]
print('stored',val,name, M.cstar(a,v,th))
t=time.time()
cs,a2,v2=crit_polish(M,a,v,th)
print('polished',cs,time.time()-t, 'supp',np.sum(a2>1e-6*a2.max()),'cheap',np.sum(v2>1e-6*a2.max()))
print('a',np.round(a2/a2.max(),3)); print('v/a',np.round(v2/np.maximum(a2,1e-300),3))
# alternate: re-optimise phases by gradient then polish again
for k in range(3):
    S=np.nonzero(a2>1e-9*a2.max())[0]
    a3,v3,t3=optimise(M,S,a2,np.maximum(v2,1e-12*a2.max()),th,mode='crit',stages=(0.003,0.001,0.0003))
    c3,_=M.cstar(a3,v3,t3)
    c4,a4,v4=crit_polish(M,a3,v3,t3)
    print(k,'grad',c3,'polish',c4,flush=True)
    a2,v2,th=a4,v4,t3
