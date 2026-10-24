import json,numpy as np,time
from ded import run
D={json.loads(l)['p']:json.loads(l) for l in open('prog15.out')}
s=1.5
for p,l,rs,rt,g in [(19,8,0.0,1.2247,0.2247),(19,8,0.3,1.0,0.25)]:
    d=D[p]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    for model in ['none','b','a']:
        t=time.time()
        if model=='none': v=run(p,s,S,a,l,d['Bhalf'],d['Gamma']*1,rs*a[l],rt*a[l],g*a[l],starts=(0.0,))
        else: v=run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],g*a[l],model=model,starts=(0.02,0.1,0.3),iters=15)
        print(p,l,rs,rt,g,model,'V/a_l=',round(v[0]/a[l],4),v[1],round(time.time()-t,1),flush=True)
