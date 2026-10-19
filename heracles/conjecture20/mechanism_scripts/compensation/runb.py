# Bonus curves: mode sym (all three |beta_T|<=rho), t12 (beta_1=beta_2=0, |beta_12|<=rho), t1 (only |beta_1|<=rho)
import numpy as np, sys, json, time
from fb import gstar
s=float(sys.argv[1]); mode=sys.argv[2]; rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 0)
rmax=3**(s-1); rhos=np.concatenate([[0.02,0.05,0.1],np.arange(0.15,rmax,0.05),[rmax]])
out={}; warm=[]; t0=time.time()
for rho in rhos:
    rb={'sym':(rho,rho,rho),'t12':(0,0,rho),'t1':(rho,0,0)}[mode]
    g,N,v=gstar(s,rb,rng,warm)
    if v is not None: warm=[(N,v)]
    out[float(rho)]=g
    print(f"s={s} mode={mode} rho={rho:.4f} G={g:.6f} G/rho={g/rho:.4f} G/rho^2={g/rho**2:.4f} N={N} t={time.time()-t0:.0f}",flush=True)
json.dump(out,open(f'G_{mode}_s{s}.json','w'))
