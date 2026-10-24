# Sup of (Phi(a)+tau g(a))/(Gamma ||a||_r) over the region where the Parseval gain is substantial: tau g >= eta*Phi.
import numpy as np, sys, cma, warnings, json
warnings.filterwarnings('ignore'); sys.path.insert(0,'..')
from cl2f import g_all, kappa, restricted
s=1.5; r=1/s; tau=3**(s-1); eta=float(sys.argv[1])
sup={}
for fn in ['supports_small.json','supports_s1.5.json']: sup.update(json.load(open(fn)))
for p in [int(x) for x in sys.argv[2:]]:
    S=sup[str(p)]['S']; K=kappa(p); F=K.shape[1]
    Q,y=restricted(K,r,[l-1 for l in S],np.random.default_rng(0)); G0=Q**(-s)
    def vals(a):
        g=g_all(a,K); H=K@a+tau*g; d=np.argmin(H); return H[d], np.min(K@a), tau*g[d]
    def f(x):
        a=np.exp(np.clip(x-np.max(x),-30,0)); H,Phi,gain=vals(a); n=np.sum(a**r)**s
        pen=max(0,eta*Phi-gain)/n/G0*20
        return -(H/n/G0)+pen
    rng=np.random.default_rng(0); best=(-9,None)
    starts=[np.zeros(F)]+[rng.normal(0,s_,F) for s_ in [0.5,1,2]*6]
    for x0 in starts:
        es=cma.CMAEvolutionStrategy(x0,0.6,{'verbose':-9,'maxfevals':5000,'seed':int(rng.integers(1e9))}); es.optimize(f)
        a=np.exp(es.result.xbest-np.max(es.result.xbest)); H,Phi,gain=vals(a); n=np.sum(a**r)**s
        if gain>=eta*Phi*0.999 and H/n/G0>best[0]: best=(H/n/G0,a)
    a=best[1]; an=a/np.sum(a**r)**s
    print(f"p={p} eta={eta}: sup ratio over substantial-gain region = {best[0]:.4f}; argmax support (>=1% of max) size {np.sum(an>0.01*an.max())}",flush=True)
