# Relation weights for the eps-Riesz product nu = prod (1 - eps cos(<chi_l,x>+psi_l)):
#  W0 = sum_{sigma != 0, sum sigma chi = 0} (eps/2)^{|sigma|},  Wl = sum_{sigma != e_l, sum sigma chi = chi_l} (eps/2)^{|sigma|}.
# quantitative Riesz: one-sided constant >= min_l (eps/2 - W_l)/(1+W0).  Exact DP over the group Z_p^n.
import numpy as np, sys, itertools
from prog import solve
def weights(chars,shape,eps):
    f=np.zeros(shape); f[(0,)*len(shape)]=1.0
    for c in chars:
        f=f+(eps/2)*(np.roll(f,c,axis=tuple(range(len(shape))))+np.roll(f,[-x for x in c],axis=tuple(range(len(shape)))))
    W0=f[(0,)*len(shape)]-1
    Wl=[f[tuple(np.mod(c,shape[0]))]-eps/2 for c in chars]
    return W0,max(Wl)
if __name__=="__main__":
    s=float(sys.argv[1])
    for p in [int(a) for a in sys.argv[2:]]:
        G,y,cov=solve(p,s); S=list(np.nonzero(y)[0]+1); K=len(S)
        for eps in (0.3,0.6,1.0):
            # 1-D: all features use the same T (worst assignment for us)
            W0,Wl=weights([(2*l%p,) for l in S],(p,),eps)
            # 2-D: greedy T assignment in Z_p^2 minimising W0
            Ts=[(1,0),(0,1),(1,1)]; ch=[]
            for l in S:
                best=None
                for T in Ts:
                    c=(2*l*T[0]%p,2*l*T[1]%p); w=weights(ch+[c],(p,p),eps)
                    if best is None or w[0]+w[1]<best[0]: best=(w[0]+w[1],c)
                ch.append(best[1])
            W0b,Wlb=weights(ch,(p,p),eps)
            c1d=(eps/2-Wl)/(1+W0); c2d=(eps/2-Wlb)/(1+W0b)
            print(f"p={p} K={K} eps={eps}: 1-D W0={W0:.3g} Wl={Wl:.3g} const>={c1d:.3f} | 2-D greedy W0={W0b:.3g} Wl={Wlb:.3g} const>={c2d:.3f}",flush=True)
