# Arc family at large n (exact mpmath), beta = G^{-1}1, scan (m, c); save best atom system for certification.
import sys, math, numpy as np, mpmath as mp
sys.path.insert(0,'.')
from kstar_exact import eval_arc
for n in [int(v) for v in sys.argv[1].split(',')]:
    s=(n+1)/2; best=None
    for f in [0.70,0.73,0.76,0.79]:
        m=int(round(f*n))
        for c in [1.2,1.27,1.34,1.41]:
            d=eval_arc(n,m,2*math.pi/(c*n+2),s,dps=30+n)
            if best is None or d['ratio']>best[0]['ratio']: best=(d,m,c)
    d,m,c=best
    beta=[complex(b) for b in d['beta']]; phis=[float(x) for x in d['phis']]
    z=np.concatenate([np.log(np.abs(beta)),np.angle(beta),phis])
    np.save(f"arcbig_n{n}.npy",np.concatenate([[m],z]))
    print(f"n={n}: ratio={mp.nstr(d['ratio'],6)} m={m} c={c} IFfree={mp.nstr(d['ratio_IFfree'],5)} IF/A={mp.nstr(d['IF'],4)} L/A={mp.nstr(d['L'],5)} E/A2={mp.nstr(d['E'],5)}",flush=True)
