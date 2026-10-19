# joint-cost Riesz safety ratio at a given allocation y (all directions, coverage term), local grid search over (rs,rt) near the critical region
import json,numpy as np,sys,glob
from ded import run
from fb import gstar
s=float(sys.argv[1]); p=int(sys.argv[2]); l=int(sys.argv[3]); yv=np.array(json.loads(sys.argv[4]))
D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out' if s==1.5 else 'prog12.out')}
d=D[p]; S=d['S']; r=1/s; yv=yv/np.sum(yv**r)**s; y=dict(zip(S,yv)); a={j:3**(1-s)*y[j] for j in S}; F=(p-1)//2
cov={dd:3**(1-s)*(d['Gamma']-sum(y[j]*(1-np.cos(2*np.pi*j*dd/p)) for j in S)) for dd in range(1,F+1)}
G={}
for f in glob.glob(f'grid_s{s}_part*.json'): G.update(json.load(open(f))['G'])
best=(np.inf,None)
pts=[(rs,rt) for rs in (0.4,0.5,0.6,0.7) for rt in (0.5,0.6,0.7,0.8,0.9,1.1)]
for rs,rt in pts:
    g=G[f"{rs},{rt}"]
    f=lambda t: run(p,s,S,a,l,list(range(1,F+1)),d['Gamma'],rs*a[l],rt*a[l],t*g*a[l],starts=(0.0,),joint=True,cov=cov)[0]
    lo,hi=0.3,4.0
    for it in range(13):
        mid=0.5*(lo+hi)
        if f(mid)>=0: lo=mid
        else: hi=mid
    if lo<best[0]: best=(lo,(rs,rt))
print(f"p={p} l={l} y={np.round(yv,4).tolist()}: joint-cost sigma (all dirs) = {best[0]:.3f} at {best[1]}",flush=True)
