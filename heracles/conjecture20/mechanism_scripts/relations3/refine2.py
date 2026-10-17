# Independent refinement from the best runs of each K (different starting supports), to bound the uncertainty of Gamma.
import numpy as np, sys, json
sys.path.insert(0,'.')
from refine import refine
from largep import restricted
p=int(sys.argv[1]); tmax=float(sys.argv[2]); Ks=[int(k) for k in sys.argv[3].split(',')]
d=json.load(open(f'large_{p}.json')); rng=np.random.default_rng(11)
for K in Ks:
    r=max([r for r in d['runs'] if r['K']==K and r['rep']!='refined'],key=lambda r:r['Gamma'])
    G,S=refine(p,r['S'],r['Gamma'],rng,rounds=60,tmax=tmax); Gs,S2,y=restricted(p,S)
    print(f"p={p} from K={K} G={r['Gamma']:.5f} -> K={len(S2)} G={Gs:.5f}",flush=True)
