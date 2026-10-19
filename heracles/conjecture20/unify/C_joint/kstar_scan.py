# K*(s,p), binding kappa-mass, and the single-T Parseval threshold 1/c_n(s)^2 at b=2 (s=nu/2).
import sys, os, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
from prog import solve, kappa
for n in [2,3,4,5,7,9]:
    nu=n+1; s=nu/2; c=1-(n/nu)**(s-1)
    print(f"n={n} b=2 s={s:.2f} c_n={c:.4f} 1/c^2={1/c**2:.2f}", flush=True)
    for p in [31,61,101,211,401]:
        G,y,cov=solve(p,s,restarts=4); S=np.nonzero(y)[0]+1; K=kappa(p)
        B=np.nonzero(cov<G*(1+1e-6))[0]+1
        kb=[K[d-1,S-1].sum() for d in B]
        print(f"   p={p} Gamma={G:.5f} K*={len(S)} |Bind|={len(B)} binding kappa-mass max {max(kb):.2f}", flush=True)
