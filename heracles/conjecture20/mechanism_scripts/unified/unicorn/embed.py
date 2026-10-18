# Step 2: build the cheater network and compute its exact margin against the clean optimum.
#  - R: programme support at p (clean features, weights y scaled to budget share 1-f)
#  - B: block of m frequencies (antipodal progression at a binding offset d0 of R, or controls), budget share f split equally,
#       built at maximal dirtiness: clean amplitude a/(1-c), interference v = a/(1-c), phases chosen to flatten the block wave.
#  margin h = min_{delta!=0} [ sum_l Aclean_l kappa_l(delta) + min_w (phi(w) - phi(w-delta)) ],  phi = block wave.
#  Compared with Gamma(p) (clean optimum, unit budget).  Ratio > 1 would be a network beating the prediction.
import numpy as np, json, sys
from scipy.optimize import minimize
from scipy.special import logsumexp
s=1.5; c=1-(2/3)**(s-1)
p=int(sys.argv[1]); m=int(sys.argv[2]); kind=sys.argv[3]
runs=json.load(open(f'/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/large_{p}.json'))['runs']
r0=max(runs,key=lambda r:r['Gamma']); SR=np.array(r0['S']); yR=np.array(r0['y']); G=r0['Gamma']
d=np.arange(1,p); w=np.arange(p)
def cov(S,a): return (a[:,None]*(1-np.cos(2*np.pi*np.outer(S,d)/p))).sum(0)
covR=cov(SR,yR); d0=int(d[np.argmin(covR)])
inv=pow(d0,-1,p); rng=np.random.default_rng(1)
def fold(l): l%=p; return min(l,p-l)
if kind=='antipodal': B=[fold(((p-1)//2-j)*inv) for j in range(m)]
elif kind=='prog':    k=int(rng.integers(1,p)); B=[fold(k*(j+1)) for j in range(m)]
else:                 B=list(rng.choice(np.arange(1,(p-1)//2+1),m,replace=False))
B=np.array(sorted(set(B)-set(SR.tolist()))); m=len(B)
# flatten the block wave: minimise peak-to-trough of phi over phases (equal amplitudes)
E=2*np.pi*np.outer(w,B)/p
def osc_s(ps,T):
    ph=np.cos(E+ps).sum(1); return (logsumexp(T*ph)+logsumexp(-T*ph))/T
best=None
for t in range(4):
    ps=rng.uniform(0,2*np.pi,m)
    for T in [0.5,2,8,32]:
        ps=minimize(osc_s,ps,args=(T/np.sqrt(m),),method='L-BFGS-B',options=dict(maxiter=300)).x
    ph=np.cos(E+ps).sum(1); o=ph.max()-ph.min()
    if best is None or o<best[0]: best=(o,ps)
osc_unit,psi=best; phi_unit=np.cos(E+psi).sum(1)
print(f"p={p} K*={len(SR)} block={kind} m={m}: block wave half peak-to-trough / total size = {osc_unit/2/m:.3f}  (cheater needs < {c:.3f})",flush=True)
for f in [0.02,0.05,0.1,0.2,0.3,0.5]:
    aR=(1-f)**s*yR; aB=np.full(m,(f/m)**s)
    clean=cov(SR,aR)+cov(B,aB)                       # paid coverage (block clean)
    cheat=cov(SR,aR)+cov(B,aB/(1-c))                 # block bought at max discount
    phi=phi_unit*(f/m)**s/(1-c)                      # block interference, v = a/(1-c) each
    o=phi.max()-phi.min()
    order=np.argsort(cheat); h=np.inf
    for idx in order:
        if cheat[idx]-o>=h: break
        dl=d[idx]; h=min(h, cheat[idx]+(phi-np.roll(phi,dl)).min())
    print(f"   block share f={f:4.2f}: clean efficiency {clean.min()/G:.3f} | cheater margin / clean optimum {h/G:.3f} "
          f"| without the interference loss it would be {cheat.min()/G:.3f}",flush=True)
