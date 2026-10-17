# Allocation moves against the L^2 (quadratic) test with joint costs: sigmaL2(y) on the coarse grid, all directions.
import json,numpy as np,glob,sys,time
from gameL2 import GameL2
s=float(sys.argv[1]); p=int(sys.argv[2]); l=int(sys.argv[3]); eps_list=[float(e) for e in sys.argv[4].split(',')]; ntr=int(sys.argv[5])
G={}
for f in glob.glob(f'grid_s{s}_part*.json'): G.update(json.load(open(f))['G'])
D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out' if s==1.5 else 'prog12.out')}
d=D[p]; S=d['S']; y0=np.array(d['y']); r=1/s; F=(p-1)//2; Gam=d['Gamma']
rmax=3**(s-1); axL=np.unique(np.round(np.concatenate([[0,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.1,1.3,1.5],[rmax]]),4)); axL=axL[axL<=rmax+1e-9]
def sig(yv,Dh):
    y=dict(zip(S,yv)); a={j:3**(1-s)*y[j] for j in S}
    B=np.array(sorted(set(Dh)|{p-x for x in Dh}))
    cov=3**(1-s)*(Gam-sum(y[j]*(1-np.cos(2*np.pi*j*B/p)) for j in S))
    g=GameL2(p,S,Dh,l,a,cov=cov); best=(np.inf,None)
    for rs in axL:
        for rt in axL:
            gg=G.get(f"{rs},{rt}")
            if gg is None or gg<1e-7: continue
            v=g.Pi({'1':rs*a[l],'2':rs*a[l],'12':rt*a[l]})/(a[l]*gg)
            if v<best[0]: best=(v,(rs,rt))
    return best
t0=time.time()
print(f"p={p} l={l}: sigmaL2(y*) B-only={sig(y0,d['Bhalf'])}  all-dirs={sig(y0,list(range(1,F+1)))}  t={time.time()-t0:.0f}",flush=True)
rng=np.random.default_rng(3)
for eps in eps_list:
    for t in range(ntr):
        v=eps*rng.standard_normal(len(S)); y=y0*np.exp(v); y=y/np.sum(y**r)**s
        print(f"  eps={eps}: sigmaL2={sig(y,list(range(1,F+1)))}",flush=True)
