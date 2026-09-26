import numpy as np
rng=np.random.default_rng(1)
p,n=7,2
def parts(x):
    return [x[t*p:(t+1)*p] for t in range(n)], x[n*p:]
def score_fft(x):
    U,w=parts(x); g=np.meshgrid(*([np.arange(p)]*n),indexing='ij')
    h=np.maximum(sum(U[t][g[t]] for t in range(n)),0); H=np.fft.fftn(h); wh=np.fft.fft(w)
    return sum(abs(2*p**(-(n+1))*H[tuple([j]*n)]*np.conj(wh[j])) for j in range(1,(p-1)//2+1))/(x@x)
def score_direct(x):
    # eta(delta)=E_a h(a) w(sum a - delta); amplitudes from DFT of eta over delta
    U,w=parts(x); g=np.meshgrid(*([np.arange(p)]*n),indexing='ij')
    h=np.maximum(sum(U[t][g[t]] for t in range(n)),0); s=sum(g)%p
    eta=np.array([np.mean(h*w[(s-d)%p]) for d in range(p)])
    E=np.fft.fft(eta)/p   # eta(d)=sum_j c_j rho^{jd}, c_j = E[-j]... amplitudes |2c_j|
    return sum(2*abs(E[j]) for j in range(1,(p-1)//2+1))/(x@x)
x=rng.standard_normal(p*(n+1)); print('fft',score_fft(x),'direct',score_direct(x))
best=0;bx=None
for r in range(100):
    x=rng.standard_normal(p*(n+1)); s=score_direct(x); step=0.5
    for it in range(300):
        y=x+step*rng.standard_normal(x.size); sy=score_direct(y)
        if sy>s: x,s=y,sy
        else: step*=0.995
    if s>best: best,bx=s,x
print('best direct',best,'fft on same',score_fft(bx),' paper bound p^-n/2',p**(-n)/2,' corrected 1/(2p)',1/(2*p))
