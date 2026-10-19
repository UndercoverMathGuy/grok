# Box-aware certificate at the delta's with smallest water-filling value:
#   W_delta <= sup_{0<=b<=bbar} [c sum b s - max(Pars(b), TiltMean(b))],  bbar = 2 tau a s.
import numpy as np, sys, json, cma, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from cert import kapS, g_waterfill, Tilt, norm, CA, C12, TAU
p=int(sys.argv[1]); K=int(sys.argv[2]); c=CA if sys.argv[3]=='A' else C12
d=json.load(open(f'large_{p}.json')); Gref=max([r['Gamma'] for r in d['runs']]+[float(sys.argv[5]) if len(sys.argv)>5 else 0])
r=max([r for r in d['runs'] if len(r['S'])==K],key=lambda r:r['Gamma'])
S=np.array(r['S']); a=np.array(r['y']); Km=kapS(p,S); cov=Km@a; n=norm(a)
gw=g_waterfill(a,Km,c); H=(cov+TAU*gw)/(Gref*n); order=np.argsort(H)[:int(sys.argv[4])]
tl=Tilt(p,S); best=9
for i in order:
    sv=np.sqrt(Km[i]/2); bbar=2*TAU*a*sv
    def W(z):
        b=bbar/(1+np.exp(-z)); return -(c*(b*sv).sum()-tl.T(b))
    rng=np.random.default_rng(0); bestW=0
    for x0 in [np.full(K,3.0),np.zeros(K),rng.normal(0,2,K)]:
        es=cma.CMAEvolutionStrategy(x0,1.5,{'verbose':-9,'maxfevals':600,'seed':1}); es.optimize(W); bestW=max(bestW,-es.result.fbest)
    val=(cov[i]+bestW)/(Gref*n); best=min(best,val)
    print(f"delta={i+1}  (Ka)/G||a||={cov[i]/(Gref*n):.4f}  CL2 value={H[i]:.4f}  box-tilt value={val:.4f}",flush=True)
print(f"p={p} K={K} c={sys.argv[3]}: CL2={H.min():.4f}  box-tilt (over tested deltas) = {best:.4f}")
