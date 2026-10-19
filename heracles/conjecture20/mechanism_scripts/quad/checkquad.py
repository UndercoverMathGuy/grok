# Sanity check of Theorem thm:quad on random class-C fields, n=2: h <= sum_k Re(conj(beta_k) c_k) - sum_{l in L'} sigma_l |c_l| f(gamma_l)
import numpy as np, itertools
f=lambda g:(np.sqrt(g*g+4)-g)/2
def d3_ok(L,p):
    for eps in itertools.product(range(-3,4),repeat=len(L)):
        if all(e==0 for e in eps): continue
        if sum(abs(e)==3 for e in eps)>1: continue
        if sum(e*l for e,l in zip(eps,L))%p==0: return False
    return True
rng=np.random.default_rng(1); p=31; n=2
x=np.array(list(itertools.product(range(p),repeat=n)))
Ts=[(1,0),(0,1),(1,1)]; worst=1e9
for trial in range(200):
    S=sorted(rng.choice(np.arange(1,(p+1)//2),3,replace=False)); L=S[:2]
    if not d3_ok(L,p): continue
    if any(any((k-sum(e*l for e,l in zip(eps,L)))%p==0 for eps in itertools.product(range(-2,3),repeat=2)) for k in S[2:]): continue
    beta={k:abs(rng.normal())+0.5 for k in S}; a={k:(rng.normal(size=3)+1j*rng.normal(size=3))*0.4 for k in S}
    for d in range(1,p):
        g=np.zeros(len(x)); rhs=0
        for k in S:
            c=1-np.exp(-2j*np.pi*k*d/p)
            W=sum(a[k][i]*np.exp(2j*np.pi*2*k*(x@np.array(T))/p) for i,T in enumerate(Ts))
            g+=np.real((np.conj(beta[k])+W)*c); rhs+=np.real(np.conj(beta[k])*c)
            if k in L:
                sig=np.sqrt(0.5*np.sum(abs(a[k])**2)); tau=2*a[k][0]*a[k][1]*np.conj(a[k][2])
                gam=0.75*np.real(tau*c)/(sig**3*abs(c)); rhs-=sig*abs(c)*f(gam)
        h=g.min(); worst=min(worst,rhs-h)
print("min over trials/directions of RHS - min_x g(x,delta):",worst,"(must be >= 0)")
# exact check: E[w g] with the optimal product test equals RHS under (D3),(I2)
maxerr=0; cnt=0
for trial in range(100):
    S=sorted(rng.choice(np.arange(1,(p+1)//2),3,replace=False)); L=S[:2]
    if not d3_ok(L,p): continue
    if any(any((k-sum(e*l for e,l in zip(eps,L)))%p==0 for eps in itertools.product(range(-2,3),repeat=2)) for k in S[2:]): continue
    beta={k:abs(rng.normal())+0.5 for k in S}; a={k:(rng.normal(size=3)+1j*rng.normal(size=3))*0.4 for k in S}
    d=int(rng.integers(1,p)); g=np.zeros(len(x)); rhs=0; w=np.ones(len(x))
    for k in S:
        c=1-np.exp(-2j*np.pi*k*d/p)
        W=sum(a[k][i]*np.exp(2j*np.pi*2*k*(x@np.array(T))/p) for i,T in enumerate(Ts))
        g+=np.real((np.conj(beta[k])+W)*c); rhs+=np.real(np.conj(beta[k])*c)
        if k in L:
            X=np.real(W*c); V=np.mean(X**2); gam=np.mean(X**3)/V**1.5
            t=0.5*(gam+np.sqrt(gam**2+4))*np.sqrt(V); w*=(X-t)**2/(V+t*t)
            rhs-=np.sqrt(V)*f(gam)
    cnt+=1; maxerr=max(maxerr,abs(np.mean(w*g)-rhs),abs(np.mean(w)-1))
print("cases",cnt,"max |E[wg]-RHS|, |Ew-1|:",maxerr)
