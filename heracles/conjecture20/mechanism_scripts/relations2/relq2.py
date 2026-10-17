# As relq.py but with the actual payment s_m=sin(pi m delta/p) at the best binding direction of the programme support,
# and leakage counted per feature (worst-case phases).  Per-feature surplus (per unit b_m):
#   det - Lam_m - (1+W0) s_m/(2 sqrt3)  ; report min over m, max over delta in B and eps.
import numpy as np, sys, json
sys.path.insert(0,'..')
from relq import genfun
from prog2 import kappa
def run(p,S,B):
    out={}
    for kind in ['lin','quad']:
        best=(-9,None,None)
        for eps in np.linspace(0.05,1,96):
            if kind=='lin': wts={1:eps/2}; det=eps/2
            else: wts={1:eps/np.sqrt(2),2:eps/4}; det=eps/np.sqrt(2)
            G=genfun(p,S,wts); W0=G[0]-1; Lam=np.array([G[m%p]-wts[1] for m in S])
            for d in B:
                sm=np.abs(np.sin(np.pi*np.array(S)*d/p))
                v=np.min(det-Lam-(1+W0)*sm/(2*np.sqrt(3)))/det
                if v>best[0]: best=(v,eps,d)
        out[kind]=best
    return out
sup={}
for f in sys.argv[1:]: sup.update(json.load(open(f)))
for p,d in sorted(sup.items(),key=lambda t:int(t[0])):
    p=int(p); S=d['S']; B=d.get('B')
    if B is None:
        # recompute binding set from a programme solve restricted to S
        from prog3 import restricted
        K=kappa(p); Q,y=restricted(K,1/1.5,[s-1 for s in S],np.random.default_rng(0)); yy=np.zeros(K.shape[1]); yy[np.array(S)-1]=y
        cov=K@yy; B=list(np.nonzero(cov<cov.min()*(1+1e-6))[0]+1)
    o=run(p,S,B)
    print(f"p={p} K={len(S)} |B|={len(B)}  lin: surplus {o['lin'][0]*100:+.0f}% of det (eps={o['lin'][1]:.2f})   quad: {o['quad'][0]*100:+.0f}% (eps={o['quad'][1]:.2f})",flush=True)
