# L^2 (quadratic test, joint costs) safety ratio at given allocations, all directions with coverage
import json,numpy as np,glob,sys
from gameL2 import GameL2
s=float(sys.argv[1]); p=int(sys.argv[2]); l=int(sys.argv[3])
G={}
for f in glob.glob(f'grid_s{s}_part*.json'): G.update(json.load(open(f))['G'])
D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out' if s==1.5 else 'prog12.out')}
d=D[p]; S=d['S']; r=1/s; F=(p-1)//2; Gam=d['Gamma']
axL=[0,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.1,1.3,1.5,round(3**(s-1),4)]
for ys in sys.argv[4:]:
    yv=np.array(json.loads(ys)); yv=yv/np.sum(yv**r)**s; y=dict(zip(S,yv)); a={j:3**(1-s)*y[j] for j in S}
    Dh=list(range(1,F+1)); B=np.array(sorted(set(Dh)|{p-x for x in Dh}))
    cov=3**(1-s)*(Gam-sum(y[j]*(1-np.cos(2*np.pi*j*B/p)) for j in S))
    g=GameL2(p,S,Dh,l,a,cov=cov); best=(np.inf,None)
    for rs in axL:
        for rt in axL:
            gg=G.get(f"{float(rs)},{float(rt)}") or G.get(f"{rs},{rt}")
            if gg is None or gg<1e-7: continue
            v=g.Pi({'1':rs*a[l],'2':rs*a[l],'12':rt*a[l]})/(a[l]*gg)
            if v<best[0]: best=(v,(rs,rt))
    print(f"p={p} l={l} y={np.round(yv,4).tolist()}: sigmaL2(all dirs)={best[0]:.3f} at {best[1]}",flush=True)
