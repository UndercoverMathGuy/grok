# Programme Gamma_s(p) via prog3 local search (heuristic lower bound on Gamma); saves to gam_s{s}.json
import sys, json, os, time, numpy as np
sys.path.insert(0,'..')
from prog3 import solve
s=float(sys.argv[1]); starts=int(sys.argv[2]); ps=[int(a) for a in sys.argv[3:]]
fn=f'gam_s{s}_{ps[0]}_'+os.environ.get('SEED','0')+'.json'
out=json.load(open(fn)) if os.path.exists(fn) else {}
for p in ps:
    t=time.time()
    G,y,cov=solve(p,s,starts=starts,seed=p+1000*int(os.environ.get('SEED','0')))
    S=[int(j) for j in np.nonzero(y)[0]+1]; B=[int(d) for d in np.nonzero(cov<G*(1+1e-6))[0]+1]
    out[str(p)]={'Gamma':float(G),'S':S,'y':[float(y[j-1]) for j in S],'B':B}
    json.dump(out,open(fn,'w'))
    print(p,round(G,6),len(S),S,f'{time.time()-t:.0f}s',flush=True)
