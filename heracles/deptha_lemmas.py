"""fp64 validation of every ingredient of the depth-A certificate on random populations (n=2).
L1  delta_T <= (a0 - m_A)/A^{|T|}           (single-pattern aiming)
L2  |U_I| <= delta_I + 2 sqrt(S^2-a0^2)      (mirror control)
L3  e_k(x) >= |E_k(M)|/3^k                   (Cauchy-Binet + Hadamard)
L4  tails: S-x1 >= b1, S-x1-x2 >= b2         (depth-A peels)
L5  final: cost >= 3^{1-r} m_A^r
"""
import numpy as np, itertools, json, sys
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
A=0.5; r=2/3; n=2
def coeffs(c,v):
    return (c.sum(), (c*v[:,0]).sum(), (c*v[:,1]).sum(), (c*v[:,0]*v[:,1]).sum())
def mA(C,A,starts=24):
    C0,C1,C2,C12=C; best=1e9
    # exact: for fixed t2, min over t1 of Re(C0+C2 t2 + t1 (C1+C12 t2)) = Re(C0+C2 t2) - A|C1+C12 t2|
    ph=np.exp(1j*np.linspace(0,2*np.pi,20001))*A
    f=(C0+C2*ph).real-A*np.abs(C1+C12*ph)
    return f.min()
def esym(x,k): return sum(np.prod(cmb) for cmb in itertools.combinations(x,k))
worst=dict(L1=9,L2=9,L3=9,L4=9,L5=9); cnt=0
for trial in range(4000):
    N=int(rng.integers(2,8))
    mode=trial%4
    if mode==0:
        c=rng.normal(size=N)+1j*rng.normal(size=N); v=np.exp(2j*np.pi*rng.random((N,2)))
    else:
        # perturbed block padded with small atoms
        w3=np.exp(2j*np.pi*np.arange(3)/3); eps=10**rng.uniform(-3,0)
        c=np.concatenate([np.ones(3)/3,np.zeros(max(N-3,0))])[:max(N,3)]; N=len(c)
        c=c+eps*(rng.normal(size=N)+1j*rng.normal(size=N))/3
        v=np.concatenate([np.stack([w3,w3],1),np.exp(2j*np.pi*rng.random((N-3,2)))])
        v=v*np.exp(1j*eps*rng.normal(size=v.shape))
    C=coeffs(c,v); a0=abs(C[0]); x=np.abs(c); S=x.sum()
    m=mA(C,A)
    if m<=1e-3*a0: continue
    cnt+=1
    dl=[abs(C[1]),abs(C[2]),abs(C[3])]
    for d_,k in zip(dl,(1,1,2)): worst['L1']=min(worst['L1'],((a0-m)/A**k+1e-12)/(d_+1e-12))
    ml=np.stack([np.ones(N),v[:,0],v[:,0]*v[:,1]],1)   # chain vectors
    M=np.einsum('l,lk,lj->kj',c,ml,ml.conj())
    U=[M[0,1],M[1,2],M[0,2]]; Lw=[M[1,0],M[2,1],M[2,0]]
    assert abs(Lw[0]-C[1])<1e-9 and abs(Lw[1]-C[2])<1e-9 and abs(Lw[2]-C[3])<1e-9
    w=np.sqrt(max(S*S-a0*a0,0))
    for u_,d_ in zip(U,dl): worst['L2']=min(worst['L2'],(d_+2*w+1e-12)/(abs(u_)+1e-12))
    ev=np.linalg.eigvals(M); E=[ev.sum(), (ev.sum()**2-(ev**2).sum())/2, np.prod(ev)]
    for k in (1,2,3):
        ek=esym(x,k) if N>=k else 0
        worst['L3']=min(worst['L3'],(ek*3**k+1e-9)/(abs(E[k-1])+1e-9))
    xs=np.sort(x)[::-1]
    b1=(m-S*(1-A)**2)/((1+A)**2-(1-A)**2); b2=(m-S*(1-A*A))/((1+A)**2-(1-A*A))
    worst['L4']=min(worst['L4'],(S-xs[0])-b1+1,(S-xs[0]-xs[1])-b2+1)
    worst['L5']=min(worst['L5'],(x**r).sum()/(3**(1-r)*m**r))
print(cnt,worst)
json.dump(dict(count=cnt,worst={k:float(v) for k,v in worst.items()}),open('FINAL_RUNS/DEPTHA/lemmas_seed%s.json'%(sys.argv[1] if len(sys.argv)>1 else 0),'w'))
