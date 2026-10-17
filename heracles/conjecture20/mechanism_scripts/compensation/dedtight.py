# Dedicated compensators at given tight points: value V/a_l without and with dedicated (models a,b). V<0 => network wins.
import json,numpy as np,sys,time
from ded import run
s=float(sys.argv[1]); D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out' if s==1.5 else 'prog12.out')}
pts=json.loads(sys.argv[2])   # list of [p,l,rs,rt,g]
for p,l,rs,rt,g in pts:
    d=D[p]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    t=time.time()
    v0=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g*a[l],starts=(0.0,))[0]
    vb=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g*a[l],model='b',starts=(0.01,0.05,0.2,0.5),iters=20)[0]
    va=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g*a[l],model='a',starts=(0.01,0.05,0.2),iters=15)[0]
    print(f"p={p} l={l} rs={rs} rt={rt} gain={g:.4f}: V/a_l none={v0/a[l]:.4f}  block(b)={vb/a[l]:.4f}  single+siblings(a)={va/a[l]:.4f}  t={time.time()-t:.0f}",flush=True)
