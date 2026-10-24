# Task 3 (gradient version): general networks (per-slot inputs), all output frequencies, b=2.
# maximise soft-min margin / P^{nu/2} over all parameters with Adam, anneal temperature; then exact SOCP over outputs.
import math, sys, time, itertools, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)
sys.path.insert(0,'/home/user/grok/heracles/conjecture20'); sys.path.insert(0,'/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/forkC'); sys.path.insert(0,'/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/cu')
from famp import log_gamma_if, Gamma
from gen import Gen
from prog2 import solve as psolve
def setup(p,n):
    X=torch.tensor(list(itertools.product(range(p),repeat=n))); y=X.sum(1)%p
    return X,y
def margin(U,W,X,y,p,n):
    G=torch.ones(len(X),U.shape[1])
    for t in range(n): G=G*U[t][:,X[:,t]].T
    Wc=W-W.mean(1,keepdim=True)
    f=G@Wc                                   # |X| x p
    fy=f.gather(1,y[:,None]); g=fy-f         # zero at the correct class
    mask=torch.ones_like(g,dtype=torch.bool); mask[torch.arange(len(y)),y]=False
    return g[mask]
def run(p,n,U0,W0,iters,lr,seed):
    torch.manual_seed(seed); X,y=setup(p,n); nu=n+1
    U=torch.tensor(U0).clone().requires_grad_(); W=torch.tensor(W0).clone().requires_grad_()
    opt=torch.optim.Adam([U,W],lr=lr); lg=log_gamma_if(p,n,2.0)
    for it in range(iters):
        opt.zero_grad()
        g=margin(U,W,X,y,p,n); P=(U**2).sum()+((W-W.mean(1,keepdim=True))**2).sum()
        gn=g/P**(nu/2)/math.exp(lg)          # normalised gaps (units of gamma_IF)
        tau=0.02*(0.995**it)+2e-4
        hs=-tau*torch.logsumexp(-gn/tau,0)+tau*math.log(len(gn))
        (-hs).backward(); opt.step()
    g=margin(U,W,X,y,p,n); P=(U**2).sum()+((W-W.mean(1,keepdim=True))**2).sum()
    return (g.min()/P**(nu/2)).item()/math.exp(lg), U.detach().numpy(), W.detach().numpy()
def ifnet(p,n,S,shares,extra,rng,noise):
    nu=n+1; a=np.arange(p); U=[[] for _ in range(n)]; W=[]
    for j,yj in zip(S,shares):
        lam=(2**n*yj/nu)**(1/nu)
        for l in range(nu):
            al=math.pi*l/nu
            for t in range(n): U[t].append(lam*np.cos(2*math.pi*j*a/p+al))
            W.append(lam*np.cos(2*math.pi*j*a/p+n*al))
    for e in range(extra):
        for t in range(n): U[t].append(noise*rng.standard_normal(p))
        W.append(noise*rng.standard_normal(p))
    U=np.array(U); W=np.array(W)
    return U+noise*rng.standard_normal(U.shape)*np.abs(U).max()*0.5, W
if __name__=="__main__":
    p,n=int(sys.argv[1]),int(sys.argv[2]); s=(n+1)/2; t0=time.time(); rng=np.random.default_rng(0)
    G,y,cov=psolve(p,s,restarts=200); S=list(np.nonzero(y)[0]+1); sh=list(y[np.array(S)-1])
    if 1-math.cos(2*math.pi/p)>G+1e-9: S=[1]; sh=[1.0]
    print(f"p={p} n={n} programme support {S} shares {np.round(sh,4)}",flush=True)
    gen=Gen(p,n); F=(p-1)//2; best=(-1,None)
    configs=[(S,sh,0,0.0),(S,sh,2,0.05),(S,sh,n+1,0.1),(S,sh,n+1,0.3)]
    allF=list(range(1,F+1)); configs+=[(allF,[1.0/F]*F,2,0.1)]
    for (supp,shs,extra,noise) in configs:
        for rep in range(2):
            U0,W0=ifnet(p,n,supp,shs,extra,rng,noise)
            e,U,W=run(p,n,U0,W0,iters=int(sys.argv[3]) if len(sys.argv)>3 else 1500,lr=0.003,seed=rep)
            v=gen.solve(U,rew=6)        # exact outputs for these inputs
            ex=v[0]
            if v[1] is not None:
                an=gen.analyse(*v[1]); mx=max(b for _,b,_ in an)
                feats=[(l,round(b/mx,4),round(b/a,4) if a>0 else None) for l,b,a in an]
            else: feats=None
            print(f"  supp={supp} extra={extra} noise={noise} rep={rep}: grad eff {e:.5f}, exact-output eff {ex:.5f}; (l, rel amp, |beta|/IF price) {feats} ({time.time()-t0:.0f}s)",flush=True)
            if ex>best[0]: best=(ex,(U,W))
    print(f"p={p} n={n} BEST exact eff {best[0]:.6f}",flush=True)
    np.save(f"gt_p{p}_n{n}.npy",np.concatenate([best[1][0].ravel(),best[1][1].ravel()]))
