# Check: (S2) frequency split never raises cost and keeps own-frequency data; (S3) sign-flip average of tables = split table.
import numpy as np, itertools
rng=np.random.default_rng(1)
def table(U,W):  # U: m x n x p inputs, W: m x p outputs -> f[x1..xn, c]
    m,n,p=U.shape; f=np.zeros((p,)*n+(p,))
    for i in range(m):
        T=W[i]
        for t in reversed(range(n)): T=np.multiply.outer(U[i,t],T)
        f+=T
    return f
def proj(v,k,p):  # orthogonal projection of real vector v onto frequency-k (k=0: constants)
    a=np.arange(p)
    if k==0: return np.full(p,v.mean())
    c=np.cos(2*np.pi*k*a/p); s=np.sin(2*np.pi*k*a/p)
    return c*(v@c)/(c@c)+s*(v@s)/(s@s)
def margin(f,p,n):
    idx=np.indices((p,)*n).reshape(n,-1).T; h=np.inf
    for x in idx:
        y=x.sum()%p; row=f[tuple(x)]
        h=min(h,min(row[y]-row[(y-d)%p] for d in range(1,p)))
    return h
def cost(U,W,b): return sum((np.sum(U[i]**2)+np.sum(W[i]**2))**(b/2) for i in range(len(W)))
def own(f,p,n):  # clean and sign-flipped coefficients via FFT
    F=np.fft.fftn(f); out={}
    for j in range(1,(p-1)//2+1):
        for T in itertools.product([0,1],repeat=n):
            k=tuple((-j if T[t] else j)%p for t in range(n))+((-j)%p,)
            out[(j,T)]=F[k]
    return out
for p,n,m,b in [(5,2,4,2.0),(5,2,4,2.7),(7,2,3,2.0),(5,3,3,2.5),(3,2,4,2.0)]:
    U=rng.normal(size=(m,n,p)); W=rng.normal(size=(m,p)); F=(p-1)//2
    Us=np.array([[ [proj(U[i,t],k,p) for t in range(n)] for k in range(1,F+1)] for i in range(m)]).reshape(m*F,n,p)
    Ws=np.array([[proj(W[i],k,p) for k in range(1,F+1)] for i in range(m)]).reshape(m*F,p)
    f=table(U,W); fs=table(Us,Ws)
    o1=own(f,p,n); o2=own(fs,p,n); dev=max(abs(o1[k]-o2[k]) for k in o1)
    # sign-flip group average over sigma in {+-1}^{F x n}
    acc=np.zeros_like(f); cnt=0
    for sig in itertools.product([1,-1],repeat=F*n):
        sig=np.array(sig).reshape(F,n)
        U2=U.copy(); W2=W.copy()
        for i in range(m):
            for k in range(1,F+1):
                for t in range(n):
                    if sig[k-1,t]<0:
                        U2[i,t]-=2*proj(U[i,t],k,p)
                        W2[i]-=2*proj(W2[i],k,p) if False else 0  # output flipped once per (k,t) below
            # output: flipped by product over t of sigma_{k,t}
        for i in range(m):
            W2[i]=W[i].copy()
            for k in range(1,F+1):
                if np.prod(sig[k-1])<0: W2[i]-=2*proj(W[i],k,p)
        acc+=table(U2,W2); cnt+=1
    avg=acc/cnt
    dc=np.zeros_like(f)  # all-DC term
    Ud=np.array([[proj(U[i,t],0,p) for t in range(n)] for i in range(m)]); Wd=np.array([proj(W[i],0,p) for i in range(m)])
    dc=table(Ud,Wd)
    print(f"p={p} n={n} b={b}: cost {cost(U,W,b):.3f} split {cost(Us,Ws,b):.3f} | own-data dev {dev:.2e} | |avg-(split+DC)| {np.abs(avg-fs-dc).max():.2e} | h(theta) {margin(f,p,n):.3f} h(split) {margin(fs,p,n):.3f}")
