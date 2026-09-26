import numpy as np
rng=np.random.default_rng(0)
def budget(p,n,us,w):
    # h(x)=relu(sum_t u_t(a_t)); beta_j = 2 p^{-(n+1)} hhat(j..j) conj(what(j)); return sum_{j in F}|beta_j| / ||omega||^2
    grids=np.meshgrid(*[np.arange(p)]*n,indexing='ij')
    z=sum(us[t][grids[t]] for t in range(n))
    h=np.maximum(z,0)
    s=sum(grids)%p
    tot=0.0
    for j in range(1,(p-1)//2+1):
        hh=(h*np.exp(-2j*np.pi*j*s/p)).sum()
        wh=(w*np.exp(-2j*np.pi*j*np.arange(p)/p)).sum()
        tot+=abs(2*p**(-(n+1))*hh*np.conj(wh))
    nrm=sum((u**2).sum() for u in us)+(w**2).sum()
    return tot/nrm
for p in [7,11,13]:
    for n in [1,2,3]:
        best=0
        a=np.arange(p)
        # cosine neuron
        us=[np.cos(2*np.pi*a/p) for _ in range(n)]; w=np.cos(2*np.pi*a/p)
        c=budget(p,n,us,w)
        for _ in range(200):
            us=[rng.standard_normal(p) for _ in range(n)]; w=rng.standard_normal(p)
            best=max(best,budget(p,n,us,w))
        print(f"p={p} n={n} cos={c:.3e} randbest={best:.3e} claimed p^-n/2={p**(-n)/2:.3e} 1/(2p)={1/(2*p):.3e}")
