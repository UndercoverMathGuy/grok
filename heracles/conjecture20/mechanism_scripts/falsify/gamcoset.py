# augment Gamma tables with the best multiplicative-coset support (restricted programme), saved as gam_s{s}_coset.json
import sys, json, numpy as np
sys.path.insert(0,'..')
from prog3 import restricted
from prog2 import kappa
from sympy import primitive_root
s=float(sys.argv[1]); out={}
for p in [int(a) for a in sys.argv[2:]]:
    g=primitive_root(p); K=kappa(p); F=K.shape[1]; best=(0,None,None)
    for d in [d for d in range(2,p) if (p-1)%d==0]:
        h=pow(g,(p-1)//d,p); seen=set()
        for co in range(1,p):
            ls=tuple(sorted(set(min((co*pow(h,j,p))%p, p-(co*pow(h,j,p))%p) for j in range(d))))
            if ls in seen or len(ls)>min(F,60): continue
            seen.add(ls)
            if len(seen)>15: break
            Q,y=restricted(K,1/s,[l-1 for l in ls],np.random.default_rng(0),tries=4)
            if Q<np.inf and Q**(-s)>best[0]:
                yy=np.array(y)*(np.array(y)>1e-12)/Q**s; best=(Q**(-s),ls,yy)
    G,ls,yy=best; S=[int(l) for l,v in zip(ls,yy) if v>0]
    cov=K@np.bincount(np.array(ls)-1,weights=yy,minlength=F)
    out[str(p)]={'Gamma':float(G),'S':S,'y':[float(v) for v in yy if v>0],'B':[int(d) for d in np.nonzero(cov<G*(1+1e-6))[0]+1]}
    print(s,p,round(G,6),S,flush=True)
json.dump(out,open(f'gam_s{s}_coset.json','w'))
