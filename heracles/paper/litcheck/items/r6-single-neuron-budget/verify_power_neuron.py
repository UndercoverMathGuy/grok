import numpy as np, math
rng=np.random.default_rng(1)
def beta(p,n,us,w,act):
    g=np.meshgrid(*[np.arange(p)]*n,indexing='ij'); z=sum(us[t][g[t]] for t in range(n))
    h=act(z); s=sum(g)%p; out=[]
    for j in range(1,(p-1)//2+1):
        hh=(h*np.exp(-2j*np.pi*j*s/p)).sum(); wh=(w*np.exp(-2j*np.pi*j*np.arange(p)/p)).sum()
        out.append(2*p**(-(n+1))*hh*np.conj(wh))
    return np.array(out)
def prodbeta(p,n,us,w):
    out=[]
    for j in range(1,(p-1)//2+1):
        e=np.exp(-2j*np.pi*j*np.arange(p)/p)
        pr=np.prod([(u*e).sum() for u in us]); wh=(w*e).sum()
        out.append(2*p**(-(n+1))*pr*np.conj(wh))
    return np.array(out)
p=11
for n in [2,3]:
    us=[rng.standard_normal(p) for _ in range(n)]; w=rng.standard_normal(p)
    bpow=beta(p,n,us,w,lambda z:z**n); bpr=prodbeta(p,n,us,w)
    print(n, np.max(np.abs(bpow-math.factorial(n)*bpr)), np.max(np.abs(bpr)))
    nu=n+1; Bp=2*(2*p*nu)**(-nu/2)
    a=np.arange(p); c=np.cos(2*np.pi*a/p); us=[c]*n; w=c.copy()
    nrm=n*(c**2).sum()+(w**2).sum()
    for b in [2,2.5,nu]:
        val=(np.abs(beta(p,n,us,w,lambda z:z**n))**(b/nu)).sum()/nrm**(b/2)
        print(f"  b={b}: C_b^(nu/b) witness={val**(nu/b):.6e}  n!B_p={math.factorial(n)*Bp:.6e}")
