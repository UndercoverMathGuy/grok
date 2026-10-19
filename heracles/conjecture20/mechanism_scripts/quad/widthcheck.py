# Check: single-feature class-C field with N<=n atoms, Phi(z)=sum_i conj(b_i) prod_t (1+conj(xi_it) z_t), z in mu_p^n:
# min_x cm(Phi) <= 2^n sin(pi/2p) sum|b_i|  (kill each atom in its own slot).
import numpy as np, itertools
rng=np.random.default_rng(0)
for n,p in [(2,11),(3,13),(3,31),(4,17)]:
    worst=0
    for trial in range(30):
        N=n; b=rng.standard_normal(N)+1j*rng.standard_normal(N); xi=np.exp(2j*np.pi*rng.random((N,n)))
        Z=np.exp(2j*np.pi*np.array(list(itertools.product(range(p),repeat=n)))/p)   # all z in mu_p^n
        Phi=sum(np.conj(b[i])*np.prod(1+np.conj(xi[i])*Z,axis=1) for i in range(N))
        cm=Phi.real-abs(Phi.imag)/np.tan(np.pi/p)
        bound=2**n*np.sin(np.pi/(2*p))*np.abs(b).sum()
        worst=max(worst,cm.min()/bound)
    print(f"n={n} p={p}: max over trials of min_x cm / bound = {worst:.4f} (<=1 expected)")
