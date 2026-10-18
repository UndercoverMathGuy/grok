# Step 1: where does the detection side turn?  Best cheater block of m frequencies at a binding offset d0.
# Cheater gain at d0 (max dirtiness): c * sum_l v_l kappa_l(d0).   Loss: half peak-to-trough of D(w) = phi(w) - phi(w-d0),
# phi(w) = sum_l v_l cos(2 pi l w/p + psi_l).  Turn <=> gain > loss for some weights v and phases psi.
# Block = "antipodal progression": l*d0 = (p-1)/2 - j (mod p), j=0..m-1  (every element sees d0 with kappa ~ 2, and the
# block is a progression, so its wave can be made flat).  WLOG d0 = 1 (dilation invariance).
import numpy as np, sys
from scipy.optimize import minimize
from scipy.special import logsumexp
p=10007; c=1-(2/3)**0.5; w=np.arange(p); rng=np.random.default_rng(0)
def net(m, starts=2):
    L=np.array([(p-1)//2 - j for j in range(m)]); kap=1-np.cos(2*np.pi*L/p)
    E=2*np.pi*np.outer(w,L)/p; Em=2*np.pi*np.outer((w-1)%p,L)/p
    def loss_smooth(z,T):
        v=np.exp(z[:m]); ps=z[m:]
        D=(np.cos(E+ps)-np.cos(Em+ps))@v
        hi=logsumexp(T*D)/T; lo=-logsumexp(-T*D)/T
        return 0.5*(hi-lo)/ (c*(v*kap).sum())
    best=np.inf
    for t in range(starts):
        z=np.concatenate([np.zeros(m), rng.uniform(0,2*np.pi,m)])
        for T in [2,8,32,128]:
            z=minimize(loss_smooth,z,args=(T*1.0/m,),method='L-BFGS-B',options=dict(maxiter=400)).x
        v=np.exp(z[:m]); ps=z[m:]; D=(np.cos(E+ps)-np.cos(Em+ps))@v
        best=min(best, 0.5*(D.max()-D.min())/(c*(v*kap).sum()))
    return best   # loss / gain ; turn when < 1
for m in [int(a) for a in sys.argv[1:]]:
    r=net(m); print(f"block size m={m:3d}: loss/gain at d0 = {r:.3f}  ({'TURNS' if r<1 else 'no turn'});  Parseval floor {1/np.sqrt(2*m)/c*np.sqrt(2)/2*np.sqrt(2):.3f}",flush=True)
