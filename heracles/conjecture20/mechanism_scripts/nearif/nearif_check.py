# Spot-check Theorem nearif(a): h <= sum_l a_l k_l(d*) + sum_{L'} min(c12, Kq t_l) v_l k_l(d*) - X  on random relations-case networks.
import numpy as np
rng=np.random.default_rng(3); s=1.5; r=1/s; th=3**(1-s); c12=1-(2/3)**(s-1); Kq=5/(5+4*r)
worst=np.inf; ncheap=0
for trial in range(3000):
    p=int(rng.choice([7,11,13,17,19,23,29,31])); F=np.arange(1,(p-1)//2+1)
    S=rng.choice(F,size=int(rng.integers(1,len(F)+1)),replace=False)
    rows=[]
    for l in S:
        # atoms with B=C=0: random pairs (b, xi, eta),(b',-xi,-eta) with b'=b, plus an IF triple of random size
        atoms=[]
        for _ in range(int(rng.integers(0,3))):
            b=rng.exponential()*np.exp(1j*rng.normal(0,0.3)); xi=np.exp(1j*rng.uniform(0,2*np.pi)); eta=np.exp(1j*rng.uniform(0,2*np.pi))
            atoms+= [(b,xi,eta),(b,-xi,-eta)]
        w=np.exp(2j*np.pi/3); bt=rng.exponential()*rng.uniform(0,2)
        atoms+= [(bt,w**k,w**k) for k in range(3)] if bt>0 else []
        if not atoms: continue
        b=np.array([a[0] for a in atoms]); xi=np.array([a[1] for a in atoms]); eta=np.array([a[2] for a in atoms])
        A=b.sum(); B=(b*xi).sum(); C=(b*eta).sum(); D=(b*xi*eta).sum(); L=abs(b).sum(); Q=(abs(b)**r).sum()
        assert abs(B)<1e-9*L and abs(C)<1e-9*L
        rows.append((l,A,D*np.exp(1j*rng.uniform(0,2*np.pi)),L,th*Q**s))
    if not rows: continue
    rho=np.exp(2j*np.pi/p); ws=np.arange(p); ds=np.arange(1,p)
    G=np.zeros((p,p-1))
    for (l,A,W,L,a) in rows:
        c=1-rho**(-l*ds)
        G+=np.real((np.conj(A)+W*rho**(l*ws))[:,None]*c[None,:])
    h=G.min()
    kap=lambda l,d: 1-np.cos(2*np.pi*l*d/p)
    dstar=min(ds,key=lambda d:sum(a*kap(l,d) for (l,A,W,L,a) in rows))
    X=np.sqrt(sum(abs(W)**2*kap(l,dstar) for (l,A,W,L,a) in rows))
    bonus=sum(min(c12,Kq*abs(W)/L)*abs(W)*kap(l,dstar) for (l,A,W,L,a) in rows if abs(A)>a)
    ncheap+=any(abs(A)>a for (l,A,W,L,a) in rows)
    bound=sum(a*kap(l,dstar) for (l,A,W,L,a) in rows)+bonus-X
    worst=min(worst,bound-h)
print("trials with a cheap feature:",ncheap,"  min(bound - h) =",worst)
