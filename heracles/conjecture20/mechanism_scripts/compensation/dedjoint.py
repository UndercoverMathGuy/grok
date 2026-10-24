import json,numpy as np,sys,time
from ded import run
s=1.5; D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out')}
for p,l,rs,rt,G,sig in [(17,5,0.545,0.6,0.21335,1.0511),(19,8,0.5,0.7,0.21574,1.219),(11,2,0.6,0.6,0.22429,1.262)]:
    d=D[p]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    g=0.98*sig*G*a[l]; t=time.time()
    v0=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g,starts=(0.0,),joint=True)[0]
    vb=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g,model='b',starts=(0.005,0.02,0.1,0.4),iters=20,joint=True)[0]
    va=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g,model='a',starts=(0.005,0.02,0.1),iters=15,joint=True)[0]
    print(f"p={p} l={l} ({rs},{rt}) gain=0.98*sigma_joint*G*: V/a_l none={v0/a[l]:.5f} block={vb/a[l]:.5f} single+siblings={va/a[l]:.5f} t={time.time()-t:.0f}",flush=True)
