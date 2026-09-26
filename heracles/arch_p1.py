"""arch_p1.py -- check of the one-line P1 proof and of Proposition S0.
(1) junk-free populations (random phases, c solved from (KL) by least-norm): eigenvalues of M all equal A;
    D|A|^r <= ||M||_{S_r}^r <= D^r sum|c|^r.
(2) arbitrary populations: D^r sum|c|^r >= tr(M_+^r) >= ||M||_{S_r}^r >= sum|lambda(M)|^r."""
import numpy as np, itertools, json
rng=np.random.default_rng(0); worst=dict(eig=0,weyl=9,rot=9,s0a=9,s0b=9,s0c=9)
for n in [1,2,3,4]:
    D=n+1; pats=[T for k in range(n+1) for T in itertools.combinations(range(n),k)]
    for N in [2**n,2**n+2,2**n+6]:
        for rep in range(200):
            v=np.exp(2j*np.pi*rng.random((N,n)))
            ch=np.concatenate([np.ones((N,1)),np.cumprod(v,1)],1)
            for mode in ['jf','gen']:
                if mode=='jf':
                    B=np.array([[np.prod(v[l,list(T)]) for l in range(N)] for T in pats]); rhs=np.zeros(len(pats),complex); rhs[0]=1
                    c=np.linalg.lstsq(B,rhs,rcond=None)[0]
                    if np.abs(B@c-rhs).max()>1e-9: continue
                else: c=rng.normal(size=N)+1j*rng.normal(size=N)
                M=(ch.T*c)@ch.conj(); Mp=(ch.T*np.abs(c))@ch.conj()
                s=np.linalg.svd(M,compute_uv=False); ev=np.linalg.eigvals(M); evp=np.clip(np.linalg.eigvalsh(Mp),0,None)
                for r in [0.2,0.5,2/3,0.9]:
                    cost=D**r*(np.abs(c)**r).sum()
                    if mode=='jf':
                        worst['eig']=max(worst['eig'],np.abs(ev-1).max() if np.abs(ev-1).max()<1e-3 else worst['eig'])
                        worst['weyl']=min(worst['weyl'],(s**r).sum()/D); worst['rot']=min(worst['rot'],cost/(s**r).sum())
                    worst['s0a']=min(worst['s0a'],cost/(evp**r).sum()); worst['s0b']=min(worst['s0b'],(evp**r).sum()/(s**r).sum()); worst['s0c']=min(worst['s0c'],(s**r).sum()/(np.abs(ev)**r).sum())
print(worst); json.dump({k:float(x) for k,x in worst.items()},open('FINAL_RUNS/ARCH/p1_check.json','w'))
