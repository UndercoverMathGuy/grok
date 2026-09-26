import numpy as np
rng=np.random.default_rng(1)
p,n=7,2
def score(x):
    U=[x[t*p:(t+1)*p] for t in range(n)]; w=x[n*p:]
    g=np.meshgrid(*([np.arange(p)]*n),indexing='ij')
    h=np.maximum(sum(U[t][g[t]] for t in range(n)),0)
    H=np.fft.fftn(h); wh=np.fft.fft(w)
    s=sum(abs(2*p**(-(n+1))*H[tuple([j]*n)]*np.conj(wh[j])) for j in range(1,(p-1)//2+1))
    return s/(x@x)
best=0
for r in range(200):
    x=rng.standard_normal(p*(n+1)); s=score(x); step=0.5
    for it in range(400):
        y=x+step*rng.standard_normal(x.size); sy=score(y)
        if sy>s: x,s=y,sy
        else: step*=0.995
    best=max(best,s)
print('best found',best,' bound p^-n/2',p**(-n)/2, 'ratio',best/(p**(-n)/2))
