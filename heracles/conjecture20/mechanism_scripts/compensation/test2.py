import json,numpy as np
from primal import solve_primal
from game import Pi
D={json.loads(l)['p']:json.loads(l) for l in open('prog15.out')}
s=1.5; d=D[19]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}; l=8
rho=0.361; R=rho*a[l]
pi=Pi(19,S,d['Bhalf'],l,a,R)
print('Pi/(rho a)=',pi/(rho*a[l]))
for g in [0.38,0.39,0.40]:
    v,t,_=solve_primal(19,s,S,a,l,R,g*rho*a[l],d['Bhalf'])
    print(g,v/a[l])
