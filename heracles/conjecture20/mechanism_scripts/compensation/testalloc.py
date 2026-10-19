import json,numpy as np,time
from alloc import sigma_y, load_grid
s=1.5; ax,G=load_grid(s)
D={json.loads(l)['p']:json.loads(l) for l in open('prog15.out')}
d=D[17]; S=d['S']; yv=np.array(d['y']); l=5
t=time.time()
b1,_,_=sigma_y(17,s,S,yv,l,d['Gamma'],ax,G,Dhalf=d['Bhalf']); print('B only',b1,time.time()-t)
b2,_,_=sigma_y(17,s,S,yv,l,d['Gamma'],ax,G); print('all dirs',b2)
mu={dd:lam for dd,lam in zip(d['Bhalf'],d['lam'])}
b3,_,_=sigma_y(17,s,S,yv,l,d['Gamma'],ax,G,mu_dir_half=mu); print('mu* fixed',b3,time.time()-t)
