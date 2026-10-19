# Local refinement of the joint-cost Riesz safety ratio around a centre (rs0,rt0): fresh G* (more starts) and bisection.
import json,numpy as np,sys,time
from fb import gstar
from ded import run
s=float(sys.argv[1]); p=int(sys.argv[2]); l=int(sys.argv[3]); rs0=float(sys.argv[4]); rt0=float(sys.argv[5]); h=float(sys.argv[6])
D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out' if s==1.5 else 'prog12.out')}
d=D[p]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
rng=np.random.default_rng(11); best=(np.inf,None)
for rs in (rs0-h,rs0,rs0+h):
    for rt in (rt0-h,rt0,rt0+h):
        g,N,v=gstar(s,(rs,rs,rt),rng,trials=10)
        f=lambda t: run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],t*g*a[l],starts=(0.0,),joint=True)[0]
        lo,hi=0.5,3.0
        for it in range(14):
            mid=0.5*(lo+hi)
            if f(mid)>=0: lo=mid
            else: hi=mid
        print(f"p={p} l={l} ({rs:.3f},{rt:.3f}) G*={g:.5f} sigma_joint={lo:.4f}",flush=True)
        if lo<best[0]: best=(lo,(rs,rt))
print(f"p={p} l={l} refined sigma_joint={best[0]:.4f} at {best[1]}",flush=True)
