import json,numpy as np,sys
from fb import gstar
from gameL2 import GameL2
s=float(sys.argv[1]); p=int(sys.argv[2]); l=int(sys.argv[3]); var=sys.argv[4]
D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out' if s==1.5 else 'prog12.out')}
d=D[p]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
g=GameL2(p,S,d['Bhalf'],l,a,partner_free=(var=='pfree'),shared=(var=='shared'))
rng=np.random.default_rng(5); best=(np.inf,None)
for rs in np.arange(0.40,0.66,0.05):
    for rt in np.arange(0.5,0.96,0.05):
        G,N,v=gstar(s,(rs,rs,rt),rng,trials=6)
        val=g.Pi({'1':rs*a[l],'2':rs*a[l],'12':rt*a[l]})/(a[l]*G)
        if val<best[0]: best=(val,(round(rs,3),round(rt,3)),G)
print(f"p={p} l={l} var={var}: refined sigmaL2={best[0]:.4f} at {best[1]} G*={best[2]:.4f}",flush=True)
