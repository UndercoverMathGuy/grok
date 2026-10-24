# Dedicated compensators with a pooled norm budget c (sum of mode norms <= c, cost D((f c)^r) <= true concave cost): scan c.
import json,numpy as np,sys,time
from ded import run
s=1.5; D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out')}
for p,l,rs,rt,G,sig in [(17,5,0.545,0.6,0.21335,1.0511),(19,8,0.5,0.7,0.21574,1.219),(11,2,0.6,0.6,0.22429,1.262)]:
    d=D[p]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    g=0.98*sig*G*a[l]; t=time.time()
    v0=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g,starts=(0.0,),joint=True)[0]
    out=[]
    for model in ('b','a'):
        for c in [1e-4,1e-3,3e-3,0.01,0.03,0.1]:
            cc=c*a[l]
            v=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g,model=model,starts=(0.01,),iters=1,joint=True,cpool=cc)[0]
            out.append((model,c,v/a[l]))
    print(f"p={p} l={l}: V/a_l without={v0/a[l]:.5f}; with pooled dedicated budget c (units a_l): "+" ".join(f"{m}:{c:g}:{v:.4f}" for m,c,v in out)+f" t={time.time()-t:.0f}",flush=True)
