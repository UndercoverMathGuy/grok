# Best degree-2 (Hermitian-square) per-feature test: w = |P|^2/E|P|^2, P in span{1, rho^{+-chi_T}}.
# Detection (|c|=1) D(alpha) = -lambda_min(A), A_{v,u} = Xhat(v-u), X = Re(W e^{i alpha}), W = sum_T a_T e_{chi_T}, a_T = conj(beta_T).
# kflat(n,s) = sup_{atoms, alpha in [0,pi/2)} bonus*cos(alpha) / ((D(alpha)+D(-alpha))/2).   (< 1 needed; <= kR)
import numpy as np, sys, itertools
from scipy.optimize import minimize
n=int(sys.argv[1]); s=float(sys.argv[2]) if len(sys.argv)>2 else (n+1)/2; nu=n+1; r=1/s
masks=list(range(1,2**n)); V=np.array([[(m>>t)&1 for t in range(n)] for m in masks])
U=[np.zeros(n,int)]+[V[i] for i in range(len(masks))]+[-V[i] for i in range(len(masks))]
key={tuple(u):i for i,u in enumerate(U)}
# for each pair (v,u): index of character v-u among +-chi_T (or -1)
P=len(U); pos=-np.ones((P,P),int); sgn=np.zeros((P,P),int)
for a in range(P):
    for b in range(P):
        d=tuple(U[a]-U[b])
        for i,m in enumerate(masks):
            if d==tuple(V[i]): pos[a,b]=i; sgn[a,b]=1
            elif d==tuple(-V[i]): pos[a,b]=i; sgn[a,b]=-1
def D(aT,alpha):
    c=np.exp(1j*alpha); A=np.zeros((P,P),complex)
    m=pos>=0
    vals=np.where(sgn==1,0.5*aT[pos]*c,0.5*np.conj(aT[pos]*c))
    A[m]=vals[m]
    return -np.linalg.eigvalsh(A)[0]
def val(v,N):
    b=np.abs(v[:N])*np.exp(1j*v[N:2*N]); th=v[2*N:2*N+N*n].reshape(N,n); al=(np.pi/2)*np.sin(v[-1])**2*0.999
    A=b.sum(); BT=np.exp(1j*(th@V.T)).T@b
    bonus=A.real-nu**(1-s)*np.sum(np.abs(b)**r)**s
    if np.sum(np.abs(BT)**2)<1e-16: return -9
    aT=np.conj(BT)
    return bonus*np.cos(al)/(0.5*(D(aT,al)+D(aT,-al)))
rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 2); best=-9; arg=None
for N in range(2,nu+3):
    for t in range(20):
        v0=np.concatenate([rng.uniform(0.1,1,N),rng.uniform(0,2*np.pi,N+N*n),[rng.uniform(0,1)]])
        res=minimize(lambda v:-val(v,N),v0,method='Nelder-Mead',options={'maxiter':10000,'xatol':1e-10,'fatol':1e-13})
        if -res.fun>best: best=-res.fun; arg=(N,res.x[-1])
print(f"n={n} s={s}: kflat >= {best:.4f}  (N={arg[0]}, alpha={(np.pi/2)*np.sin(arg[1])**2*0.999:.3f})",flush=True)
