import numpy as np
rng=np.random.default_rng(1); worst=np.inf
for trial in range(200000):
    n=rng.integers(1,5); nu=n+1; m=rng.integers(1,7); r=rng.uniform(0.05,0.99)
    beta=rng.normal(size=m)+1j*rng.normal(size=m)
    if rng.random()<0.3: beta=np.abs(beta)  # aligned
    xi=np.exp(1j*rng.uniform(0,2*np.pi,(m,n)))
    if rng.random()<0.3: xi=np.exp(2j*np.pi*rng.integers(0,nu,(m,n))/nu)
    M=np.concatenate([np.ones((m,1)),np.cumprod(xi,axis=1)],axis=1)  # chain vectors
    H=sum(beta[l]*np.outer(M[l],M[l].conj()) for l in range(m))
    lam=np.linalg.eigvals(H); lam=lam[np.abs(lam)>1e-9*max(1,np.abs(lam).max())]
    rhs=np.sum(np.abs(lam)**r*np.cos(r*np.angle(lam)))
    lhs=nu**r*np.sum(np.abs(beta)**r)
    worst=min(worst,(lhs-rhs)/lhs)
print("min relative slack (should be >= 0):",worst)
