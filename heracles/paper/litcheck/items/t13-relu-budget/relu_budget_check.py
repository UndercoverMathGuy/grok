import numpy as np, itertools
rng=np.random.default_rng(0)
def ratio(us,w,p,n):
    # h(x)=relu(sum_t u_t(a_t)), grid Z_p^n
    grids=np.meshgrid(*[np.arange(p)]*n,indexing='ij')
    s=sum(us[t][grids[t]] for t in range(n))
    h=np.maximum(s,0)
    H=np.fft.fftn(h)  # hat h(k) = sum_x h(x) rho^{-k.x}
    W=np.fft.fft(w)
    tot=0
    for j in range(1,(p-1)//2+1):
        G=H[(j,)*n]*np.conj(W[j])
        tot+=abs(2*p**(-(n+1))*G)
    nrm=sum((u**2).sum() for u in us)+(w**2).sum()
    lhs=sum(abs(H[(j,)*n]*W[j]) for j in range(1,(p-1)//2+1))
    return tot/nrm, lhs/nrm
for p in [7,11,23]:
  for n in [1,2,3]:
    best=0;bestl=0
    a=np.arange(p)
    for A in [0.3,0.5,1,2,3]:
      us=[A*np.cos(2*np.pi*a/p)]*n; w=np.cos(2*np.pi*a/p)
      r,l=ratio(us,w,p,n); best=max(best,r); bestl=max(bestl,l)
    for _ in range(300):
      us=[rng.normal(size=p) for _ in range(n)]; w=rng.normal(size=p)*rng.uniform(0.2,3)
      r,l=ratio(us,w,p,n); best=max(best,r); bestl=max(bestl,l)
    print(f"p={p} n={n}  C2_lower={best:.4g}  claimed C2<= p^-n/2={p**(-n)/2:.4g}  (1/(2p)={1/(2*p):.4g})   lhs/||w||^2={bestl:.4g} vs claimed p/4={p/4:.4g}, p^n/4={p**n/4:.4g}")
