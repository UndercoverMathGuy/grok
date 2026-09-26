import numpy as np, itertools, math
rng=np.random.default_rng(0)
def beta_table(fun, p, n):
    # f(a_1..a_n, c) table -> coefficient at (j,..,j; -j) times 2 p^{-nu}
    nu=n+1
    grids=np.meshgrid(*([np.arange(p)]*(n+1)),indexing='ij')
    T=fun(*grids)
    F=np.fft.fftn(T)  # sum f rho^{-(k.a)}
    out=[]
    for j in range(1,(p-1)//2+1):
        idx=tuple([j]*n+[(-j)%p])
        out.append(2*p**(-nu)*F[idx])
    return np.array(out)
p,n=5,3; nu=n+1
Bp=2*(2*p*nu)**(-nu/2)
for trial in range(3):
    U=[rng.standard_normal(p) for _ in range(n)]; w=rng.standard_normal(p)
    prod=lambda *g: np.prod([U[t][g[t]] for t in range(n)],axis=0)*w[g[n]]
    powr=lambda *g: (sum(U[t][g[t]] for t in range(n)))**n*w[g[n]]
    bp=beta_table(prod,p,n); bw=beta_table(powr,p,n)
    uh=[np.fft.fft(u) for u in U]; wh=np.fft.fft(w)
    formula=np.array([2*p**(-nu)*np.prod([uh[t][j] for t in range(n)])*np.conj(wh[j]) for j in range(1,(p-1)//2+1)])
    print('prod==formula',np.allclose(bp,formula),' power/prod ratio',np.round(bw/bp,6))
# cosine witness at unit norm
zeta=1; A=1.0
cosv=lambda ph: np.cos(2*np.pi*zeta*np.arange(p)/p+ph)
U=[cosv(0.3),cosv(1.1),cosv(0.2)]; w=cosv(1.6)
nrm=math.sqrt(sum((u**2).sum() for u in U)+(w**2).sum()); U=[u/nrm for u in U]; w=w/nrm
prod=lambda *g: np.prod([U[t][g[t]] for t in range(n)],axis=0)*w[g[n]]
powr=lambda *g: (sum(U[t][g[t]] for t in range(n)))**n*w[g[n]]
bp=beta_table(prod,p,n); bw=beta_table(powr,p,n)
for b in [2,3,4]:
    print('b',b,'prod sum^(nu/b)',(np.abs(bp)**(b/nu)).sum()**(nu/b),' Bp',Bp,' power',(np.abs(bw)**(b/nu)).sum()**(nu/b),' n!Bp',math.factorial(n)*Bp)
