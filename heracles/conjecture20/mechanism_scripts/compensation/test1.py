import json,numpy as np
from game import *
D=[json.loads(l) for l in open('prog15.out')]
s=1.5; c1=1-3**(1-s)
for d in D:
    p=d['p']; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    out=[]
    for l in S:
        hs={m:slope(p,S,d['Bhalf'],l,a,m)[0] for m in ('none','partner','riding','full')}
        out.append((l,{m:round(v,4) for m,v in hs.items()}))
    print(p,S,d['Bhalf'],out,flush=True)
