# Relation mass of supports vs the independent-frequency (random) prediction:
#   R(x) = rho(x)/(e^{E(x)}/p),  rho = Z/Pi - 1 = sum_{k in Rel\0} prod I_|k|/I_0 = P(Skellam in Rel\0)/P(Skellam=0).
# R<1: fewer (weighted) relations than for independent phases.  Control: random supports of the same size.
import numpy as np, sys, json
sys.path.insert(0,'.')
from core import *
def R_of(p,S,x):
    C=cosmat(p,S); r0,eps,E=relation_split(x,C,p); return (r0+eps)/r0, E
def stats(p,S,y,rng,ntest=6):
    S=np.array(S); y=np.array(y); B,cov=binding(p,S,y); out=[]
    for d in B[:4]:
        s=np.abs(np.sin(np.pi*S*d/p)); b=y*s
        for xm in [0.5,1.0,2.0]:
            out.append(R_of(p,S,xm*b/b.max())[0])
    return np.array(out)
D=load(); rng=np.random.default_rng(0)
for fn in ['large_2003.json','large_5003.json','large_10007.json']:
    d=json.load(open(fn)); p=int(fn.split('_')[1].split('.')[0]); bst=d['best']; D[p]=bst
print("  p   K   R(prog support): min  med  max   |  R(random K-sets, same x-profile): med  [q10,q90]")
for p in sorted(D):
    if p<23: continue
    S=D[p]['S']; y=D[p]['y']; R=stats(p,S,y,rng)
    # control: random supports with the same weights / s-profile (use the vertex's b-profile but random frequencies)
    Sa=np.array(S); ya=np.array(y); B,cov=binding(p,Sa,ya); d=B[0]; s=np.abs(np.sin(np.pi*Sa*d/p)); b=ya*s
    F=(p-1)//2; Rr=[]
    for t in range(40):
        T=rng.choice(np.arange(1,F+1),len(S),replace=False)
        for xm in [0.5,1.0,2.0]: Rr.append(R_of(p,T,xm*b/b.max())[0])
    Rr=np.array(Rr)
    print(f"{p:5d} {len(S):3d}   {R.min():5.2f} {np.median(R):5.2f} {R.max():5.2f}   |   {np.median(Rr):5.2f}  [{np.quantile(Rr,0.1):5.2f},{np.quantile(Rr,0.9):5.2f}]",flush=True)
