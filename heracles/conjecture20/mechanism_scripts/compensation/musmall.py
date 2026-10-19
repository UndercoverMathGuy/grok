import json,numpy as np
from game2 import Game,family
s=1.5; out=[]
for line in open('prog15.out'):
    d=json.loads(line); p=d['p']; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    B=np.array(sorted(set(d['Bhalf'])|{p-x for x in d['Bhalf']})); lam={x:v for x,v in zip(d['Bhalf'],d['lam'])}
    mu=np.array([lam[min(x,p-x)] for x in B])
    for l in S:
        g1=Game(p,S,d['Bhalf'],l,a,family(p,S,l,T12=False),mu_dir=mu); g2=Game(p,S,d['Bhalf'],l,a,family(p,S,l,T12=True),mu_dir=mu)
        R=0.005*a[l]
        out.append((p,l,g1.Pi(R)/R,g2.Pi(R)/R**2*a[l]))
print('min h1(mu*) =',min(o[2] for o in out),' min kappa2(mu*) =',min(o[3] for o in out), [o for o in out if o[3]<1.0])
