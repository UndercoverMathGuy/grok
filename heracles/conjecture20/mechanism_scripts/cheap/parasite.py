# Parasitic-orbit test (n=2, b=2, s=1.5): programme allocation; feature(s) in CHEAP realised by 2 neurons (IF block minus one,
# rescaled to the same cost: the optimal 2-atom shape), other features by IF blocks; off-frequency input components
# (struct: 2l-j and 2l+j on one slot of every neuron of every other block; random: noise everywhere).
# Adam on soft-min normalised margin over all parameters, then exact SOCP outputs; report eff (gamma/gamma_IF) and spurious features.
import math, sys, time, itertools, numpy as np, torch, warnings, json
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)
D='/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/'
sys.path.insert(0,'/home/user/grok/heracles/conjecture20'); sys.path.insert(0,D+'forkC'); sys.path.insert(0,D+'cu')
from famp import log_gamma_if
from gen import Gen
from gt import margin, setup
from prog2 import solve as psolve
def build(p,S,sh,cheap,mode,eps,rng):
    n=2; nu=3; a=np.arange(p); U=[[],[]]; W=[]; owner=[]
    for j,yj in zip(S,sh):
        lam=(2**n*yj/nu)**(1/nu); ls=range(nu)
        if j in cheap: ls=range(2); lam=lam*math.sqrt(1.5)
        for l in ls:
            al=math.pi*l/nu
            for t in range(n): U[t].append(lam*np.cos(2*math.pi*j*a/p+al))
            W.append(lam*np.cos(2*math.pi*j*a/p+n*al)); owner.append(j)
    U=np.array(U); W=np.array(W)
    if mode=='struct':
        for i,j in enumerate(owner):
            if j in cheap: continue
            for l in cheap:
                t=rng.integers(0,2); scale=np.abs(U[t,i]).max()
                for fr in ((2*l-j)%p,(2*l+j)%p):
                    U[t,i]+=eps*scale*np.cos(2*math.pi*fr*a/p+rng.uniform(0,2*math.pi))
    elif mode=='random':
        U=U+eps*np.abs(U).max()*rng.standard_normal(U.shape)
    elif mode.startswith('extra'):
        k=int(mode[5:]); sc=np.abs(U).max()
        U=np.concatenate([U+0.03*sc*rng.standard_normal(U.shape),eps*sc*rng.standard_normal((2,k,p))],1)
        W=np.concatenate([W,eps*sc*rng.standard_normal((k,p))],0)
    return U,W,owner
def opt(p,U0,W0,iters,lr,lg):
    X,y=setup(p,2); nu=3
    U=torch.tensor(U0).clone().requires_grad_(); W=torch.tensor(W0).clone().requires_grad_()
    o=torch.optim.Adam([U,W],lr=lr)
    for it in range(iters):
        o.zero_grad(); g=margin(U,W,X,y,p,2); P=(U**2).sum()+((W-W.mean(1,keepdim=True))**2).sum()
        gn=g/P**(nu/2)/math.exp(lg); tau=0.02*(0.997**it)+1e-4
        hs=-tau*torch.logsumexp(-gn/tau,0)+tau*math.log(len(gn)); (-hs).backward(); o.step()
    g=margin(U,W,X,y,p,2); P=(U**2).sum()+((W-W.mean(1,keepdim=True))**2).sum()
    return (g.min()/P**(nu/2)).item()/math.exp(lg),U.detach().numpy(),W.detach().numpy()
if __name__=="__main__":
    p=int(sys.argv[1]); iters=int(sys.argv[2]); t0=time.time(); rng=np.random.default_rng(1)
    G,y,cov=psolve(p,1.5,restarts=200); S=list(np.nonzero(y)[0]+1); sh=list(y[np.array(S)-1])
    print(f"p={p} K*={len(S)} support {S} shares {np.round(sh,4)}",flush=True)
    gen=Gen(p,2); lg=gen.lgif; best=-9; rec=[]
    order=sorted(range(len(S)),key=lambda k:-sh[k])
    cases=[(c,m,e) for c,m,e in [([S[order[-1]]],'extra3',0.2),([S[order[-1]]],'extra6',0.2),([S[order[0]]],'extra6',0.2),([S[order[-1]]],'extra3',0.5)]] if len(sys.argv)>3 else [([S[order[0]]],'none',0.0),([S[order[0]]],'struct',0.05),([S[order[0]]],'struct',0.15),([S[order[0]]],'random',0.05),
           ([S[order[-1]]],'struct',0.1),([S[order[0]],S[order[1]]],'struct',0.1),([S[order[0]],S[order[1]]],'random',0.1)]
    for cheap,mode,eps in cases:
        for rep in range(2):
            U0,W0,owner=build(p,S,sh,cheap,mode,eps,rng)
            v0=gen.solve(U0,rew=4)[0]
            e,U,W=opt(p,U0,W0,iters,0.002,lg)
            v=gen.solve(U,rew=6); ex=v[0]
            an=gen.analyse(*v[1]) if v[1] is not None else []
            mx=max([b for _,b,_ in an],default=1)
            spur=[(l,round(b/mx,5)) for l,b,a in an if l not in S and b/mx>1e-4]
            cheapness=[(l,round(b/a,4)) for l,b,a in an if a>0 and b/mx>1e-3]
            print(f"  cheap={cheap} mode={mode} eps={eps} rep={rep}: start eff {v0:.4f} -> grad {e:.5f}, exact {ex:.5f}; |beta|/IF-price {cheapness}; spurious {spur} ({time.time()-t0:.0f}s)",flush=True)
            rec.append(dict(cheap=[int(c) for c in cheap],mode=mode,eps=eps,rep=rep,start=v0,grad=e,exact=ex,cheapness=[(int(l),float(r)) for l,r in cheapness],spur=[(int(l),float(r)) for l,r in spur]))
            if ex>best: best=ex; np.save(f"parasite_best_p{p}{sys.argv[3] if len(sys.argv)>3 else ''}.npy",np.concatenate([U.ravel(),W.ravel()]))
    print(f"p={p} BEST exact eff {best:.6f}",flush=True)
    json.dump(rec,open(f"parasite_p{p}{sys.argv[3] if len(sys.argv)>3 else ''}.json","w"))
