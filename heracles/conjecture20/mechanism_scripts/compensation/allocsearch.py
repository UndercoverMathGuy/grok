# Task 2: does moving the allocation help the network?  Minimise sigma(y) over y on the budget sphere (support S, also S+{k}).
import json,numpy as np,sys,time
from scipy.optimize import minimize
from alloc import sigma_y, load_grid
s=float(sys.argv[1]); p=int(sys.argv[2]); l=int(sys.argv[3]); budget=int(sys.argv[4]) if len(sys.argv)>4 else 60
ax,G=load_grid(s)
D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out' if s==1.5 else 'prog12.out')}
d=D[p]; S=d['S']; y0=np.array(d['y']); r=1/s
def proj(v,Sx,y0x):
    y=y0x*np.exp(v); return y/np.sum(y**r)**s
def f(v,Sx,y0x):
    y=proj(v,Sx,y0x)
    if l not in Sx: return np.inf
    return sigma_y(p,s,Sx,y,l,d['Gamma'],ax,G)[0][0]
t0=time.time(); base=f(np.zeros(len(S)),S,y0)
print(f"p={p} l={l} sigma(y*) all-dirs = {base:.4f}",flush=True)
rng=np.random.default_rng(0); best=(base,0)
for eps in [0.02,0.05,0.1,0.2,0.4]:
    for t in range(4):
        v=eps*rng.standard_normal(len(S)); val=f(v,S,y0)
        if val<best[0]: best=(val,eps)
        print(f"  random eps={eps}: sigma={val:.4f}",flush=True)
res=minimize(lambda v:f(v,S,y0),0.05*rng.standard_normal(len(S)),method='Nelder-Mead',options={'maxfev':budget,'xatol':1e-3,'fatol':1e-4})
print(f"  Nelder-Mead: sigma={res.fun:.4f} at y={np.round(proj(res.x,S,y0),4)} (y*={np.round(y0,4)})  t={time.time()-t0:.0f}",flush=True)
