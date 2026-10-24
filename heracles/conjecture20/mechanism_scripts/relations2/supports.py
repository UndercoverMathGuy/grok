# Heuristic programme maximisers (prog3) for a list of primes; saves y, support, binding set.
import numpy as np, sys, json
sys.path.insert(0,'..')
from prog3 import solve
s=float(sys.argv[1]); out={}
for p in [int(a) for a in sys.argv[2:]]:
    G,y,cov=solve(p,s,starts=int(8 if p>60 else 16))
    S=[int(j) for j in np.nonzero(y)[0]+1]; B=[int(d) for d in np.nonzero(cov<G*(1+1e-6))[0]+1]
    out[p]={'Gamma':float(G),'S':S,'y':[float(y[j-1]) for j in S],'B':B}
    print(p,G,S,B,flush=True)
    json.dump(out,open(f'supports_s{s}.json','w'))
