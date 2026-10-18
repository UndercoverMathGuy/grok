# The cheater's best allocation with the block in place: re-optimise ALL weights on R u B (budget sum b_l = 1, amplitude b^s),
# block features bought at the maximal discount (amplitude x 1/(1-c)), then charge the exact interference loss.
#   value_discount_only : max_a min_delta coverage with the discount, interference ignored (most generous to the cheater)
#   value_exact         : same allocation, exact margin including the block's interference (flattened phases)
import numpy as np, json, sys
from scipy.optimize import minimize
from scipy.special import logsumexp
s=1.5; c=1-(2/3)**(s-1); p=int(sys.argv[1]); m=int(sys.argv[2]); kind=sys.argv[3] if len(sys.argv)>3 else 'antipodal'
runs=json.load(open(f'/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/large_{p}.json'))['runs']
r0=max(runs,key=lambda r:r['Gamma']); SR=np.array(r0['S']); yR=np.array(r0['y']); G=r0['Gamma']
d=np.arange(1,p); w=np.arange(p); rng=np.random.default_rng(2)
K=lambda S: 1-np.cos(2*np.pi*np.outer(S,d)/p)
d0=int(d[np.argmin(yR@K(SR))]); inv=pow(d0,-1,p); fold=lambda l: min(l%p,p-l%p)
if kind=='antipodal': B=sorted({fold(((p-1)//2-j)*inv) for j in range(m)}-set(SR.tolist()))
else: B=sorted(set(rng.choice(np.arange(1,(p-1)//2+1),m,replace=False).tolist())-set(SR.tolist()))
B=np.array(B); S=np.concatenate([SR,B]); KS=K(S); boost=np.concatenate([np.ones(len(SR)),np.full(len(B),1/(1-c))])
def negsoftmin(z,T):
    b=np.exp(z); b=b/b.sum(); a=b**s*boost; cv=a@KS
    lse=logsumexp(-T*cv); wts=np.exp(-T*cv-lse)
    # d/dz of softmin: d cv/d b_l = s b_l^{s-1} boost_l K_l ; chain through normalisation
    g_b=s*b**(s-1)*boost*(KS@wts); g_z=b*(g_b-(b@g_b))
    return lse/T, g_z            # minimising (-softmin) = lse/T
best=None
for start in range(3):
    b0=np.concatenate([yR**(1/s)*(1-0.1*start), np.full(len(B),0.1*start/len(B)+1e-6)])
    z=np.log(b0/b0.sum())
    for T in [200,1000,5000,20000]:
        z=minimize(negsoftmin,z,args=(T/G,),jac=True,method='L-BFGS-B',options=dict(maxiter=500)).x
    b=np.exp(z); b/=b.sum(); a=b**s*boost; v=(a@KS).min()
    if best is None or v>best[0]: best=(v,b)
v,b=best; fB=b[len(SR):].sum(); aB=(b**s*boost)[len(SR):]
print(f"p={p} K*={len(SR)} block={kind} m={len(B)}: best cheater allocation puts {fB:.3f} of budget on the block")
print(f"   discount only (interference free of charge): {v/G:.4f} x clean optimum")
# exact margin with the block's interference, phases flattened for this allocation's amplitudes
E=2*np.pi*np.outer(w,B)/p
def osc_s(ps,T):
    ph=np.cos(E+ps)@aB; return (logsumexp(T*ph)+logsumexp(-T*ph))/T
ps=rng.uniform(0,2*np.pi,len(B))
if aB.sum()>0:
    for T in [0.5,2,8,32]: ps=minimize(osc_s,ps,args=(T/np.sqrt(len(B))/aB.max(),),method='L-BFGS-B',options=dict(maxiter=300)).x
phi=np.cos(E+ps)@aB; o=phi.max()-phi.min(); cheat=(b**s*boost)@KS; h=np.inf
for idx in np.argsort(cheat):
    if cheat[idx]-o>=h: break
    h=min(h,cheat[idx]+(phi-np.roll(phi,d[idx])).min())
print(f"   exact, with interference:                   {h/G:.4f} x clean optimum   (block wave flatness {o/2/max(aB.sum(),1e-300):.3f})")
