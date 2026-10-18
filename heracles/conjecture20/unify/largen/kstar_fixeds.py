# k* at FIXED s (b = nu/s grows with n): slot-symmetric search (random+structured starts) and arc family.
import sys, math, numpy as np, warnings, mpmath as mp
warnings.filterwarnings("ignore")
sys.path.insert(0,'.')
from kstar_sym import run
from kstar_exact import eval_arc
s=float(sys.argv[1])
for n in [int(v) for v in sys.argv[2].split(',')]:
    best=(-np.inf,None)
    for m in range(2, n+2):
        v,z,info=run(n,m,s,8,seed=13*n+m)
        if v>best[0]: best=(v,m)
    # arc family
    arcb=-np.inf
    for f in [0.5,0.6,0.7,0.8,0.9,1.0]:
        m=max(2,int(round(f*(n+1))))
        for c in [0.9,1.1,1.3,1.5,1.7]:
            d=eval_arc(n,m,2*math.pi/(c*n+2),s,dps=30+n)
            r=d["ratio"]
            if mp.im(r)==0 and d["E"]>0: arcb=max(arcb,float(mp.re(r)))
    print(f"s={s} n={n}: k*_sym (search) >= {best[0]:.4f} (m={best[1]})   arc family {arcb:.4f}",flush=True)
