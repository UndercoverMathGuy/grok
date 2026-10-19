# 2-D sharp bonus grid G*(rho_s, rho_12): |beta_1|,|beta_2| <= rho_s, |beta_12| <= rho_12  (a = 1).  Rows split by argv.
import numpy as np, sys, json, time
from fb import gstar
s=float(sys.argv[1]); part=int(sys.argv[2]); nparts=int(sys.argv[3])
rmax=3**(s-1)
ax=np.unique(np.round(np.concatenate([[0,0.02,0.05,0.1,0.15,0.2],np.arange(0.3,rmax,0.1),[rmax]]),4))
rng=np.random.default_rng(100+part); out={}; t0=time.time()
for i,rs in enumerate(ax):
    if i%nparts!=part: continue
    warm=[]
    for rt in ax:
        g,N,v=gstar(s,(rs,rs,rt),rng,warm,trials=5)
        if v is not None: warm=[(N,v)]
        out[f"{rs},{rt}"]=g
    print(f"row rho_s={rs} done: "+" ".join(f"{g:.4f}" for g in [out[f'{rs},{rt}'] for rt in ax])+f" t={time.time()-t0:.0f}",flush=True)
    json.dump(dict(ax=list(ax),G=out),open(f'grid_s{s}_part{part}.json','w'))
