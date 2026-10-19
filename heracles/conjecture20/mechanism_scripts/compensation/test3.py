import json,numpy as np
from game2 import *
from game import slope, Pi
D={json.loads(l)['p']:json.loads(l) for l in open('prog15.out')}
s=1.5
for p in [11,17,19]:
    d=D[p]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    for l in S:
        g1=Game(p,S,d['Bhalf'],l,a,family(p,S,l,T12=False))
        g2=Game(p,S,d['Bhalf'],l,a,family(p,S,l,T12=True))
        R=0.5*a[l]
        print(p,l,'h1',round(g1.slope(),4),'old',round(slope(p,S,d['Bhalf'],l,a,'full')[0],4),'h12',round(g2.slope(),4),
              'Pi1',round(g1.Pi(R)/R,4),'old',round(Pi(p,S,d['Bhalf'],l,a,R)/R,4),'Pi12',round(g2.Pi(R)/R,4),'shared?',has_shared(p,S,l), family(p,S,l,T12=True))
