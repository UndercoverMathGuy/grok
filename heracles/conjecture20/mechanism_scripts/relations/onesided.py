# Multi-direction one-sided constant of a frequency set S (1-D interference profile, single T):
#   C1 = min over amplitudes a>=0, phases th of  max_{delta in B}  (-min_w D_delta(w)) / sum_l a_l kappa_l(delta),
#   D_delta(w) = sum_l a_l |c_l(delta)| cos(2 pi l w/p + th_l + arg c_l(delta)),  c_l(delta)=1-rho^{-l delta}.
# For fixed phases the inner min over a is a quasi-convex problem: bisection on t with an LP feasibility test.
import numpy as np, sys
from scipy.optimize import linprog
from prog import solve, kappa
def setup(p,s):
    G,y,cov=solve(p,s); S=np.nonzero(y)[0]+1; F=(p-1)//2
    B=np.nonzero(cov<G*(1+1e-6))[0]+1; B=np.concatenate([B,p-B])
    return G,y,S,B
def inner(p,S,B,th,tol=1e-4):
    w=np.arange(p); rows=[]; kap=[]
    for d in B:
        c=1-np.exp(-2j*np.pi*S*d/p)
        M=np.abs(c)[None,:]*np.cos(2*np.pi*np.outer(w,S)/p+th[None,:]+np.angle(c)[None,:])  # p x K, D=M@a
        rows.append(M); kap.append(1-np.cos(2*np.pi*S*d/p))
    K=len(S)
    def feas(t):
        # -M a <= t*kap.a  for all rows ; sum a = 1 ; a>=0
        A_ub=np.vstack([-M-t*k[None,:] for M,k in zip(rows,kap)])
        res=linprog(np.zeros(K),A_ub=A_ub,b_ub=np.zeros(A_ub.shape[0]),A_eq=np.ones((1,K)),b_eq=[1],bounds=(0,None),method='highs')
        return res.status==0, (res.x if res.status==0 else None)
    lo,hi=0.0,2.0
    while hi-lo>tol:
        mid=(lo+hi)/2; ok,_=feas(mid)
        if ok: hi=mid
        else: lo=mid
    return hi
if __name__=="__main__":
    p=int(sys.argv[1]); s=float(sys.argv[2]); R=int(sys.argv[3]) if len(sys.argv)>3 else 20
    G,y,S,B=setup(p,s); K=len(S); rng=np.random.default_rng(0)
    best=(9,None)
    for r in range(R):
        th=rng.uniform(0,2*np.pi,K); v=inner(p,S,B,th,1e-3); step=1.0
        for it in range(150):
            thn=th+rng.normal(0,step,K)*(rng.random(K)<0.3); vn=inner(p,S,B,thn,1e-3)
            if vn<v: th,v=thn,vn
            else: step*=0.97
            if step<0.02: break
        if v<best[0]: best=(v,th.copy())
        print(f"  restart {r}: {v:.4f}",flush=True)
    vfin=inner(p,S,B,best[1],1e-5)
    # single-direction version at the best delta for comparison
    print(f"p={p} s={s} K={K} |B|={len(B)}  multi-direction one-sided constant C1 <= {vfin:.4f}",flush=True)
    np.save(f"os_p{p}.npy",best[1])
