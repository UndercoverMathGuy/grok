# normalized margin m/P^{3/2} of real n=2 gated nets on the full table:
# dense junk-free 3-neuron blocks vs ONE random-phase neuron per frequency (junky)
import numpy as np, sys
def margin(p,E1,E2,W):
    y=np.arange(p); m=np.inf
    for x1 in range(p):
        Z=(E1[:,x1][:,None]*E2).T@W      # x2 x c
        yy=(x1+y)%p
        corr=Z[y,yy].copy(); Z[y,yy]=-np.inf
        m=min(m,(corr-Z.max(1)).min())
    P=(E1**2).sum()+(E2**2).sum()+(W**2).sum()
    return m/P**1.5
def dense_block(p):
    h=(p-1)//2; x=np.arange(p); E1=[];E2=[];W=[]
    for k in range(1,h+1):
        for l in range(3):
            ph=np.pi*l/3
            E1.append(np.cos(2*np.pi*k*x/p+ph)); E2.append(np.cos(2*np.pi*k*x/p+ph)); W.append(np.cos(2*np.pi*k*x/p+2*ph))
    return [np.array(a) for a in (E1,E2,W)]
def dense_single(p,rng):
    h=(p-1)//2; x=np.arange(p); a,b=rng.uniform(0,2*np.pi,(2,h))
    k=np.arange(1,h+1)[:,None]
    return np.cos(2*np.pi*k*x/p+a[:,None]),np.cos(2*np.pi*k*x/p+b[:,None]),np.cos(2*np.pi*k*x/p+(a+b)[:,None])
rng=np.random.default_rng(1)
for p in [int(a) for a in sys.argv[1:]]:
    gb=margin(p,*dense_block(p))
    gs=max(margin(p,*dense_single(p,rng)) for _ in range(3))
    print(f"p={p}: dense junk-free block {gb:.4e}   best of 3 random single-neuron dense {gs:.4e}   ratio {gs/gb:.3f}",flush=True)
