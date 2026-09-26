# Can junk-bearing PURE networks beat the ansatz at larger p?  N atoms (neurons) per frequency, N<=2 means
# every feature carries same-|k| junk (Feature Cost Theorem: junk-free needs >=3). N=3 is the optimizer control.
# usage: purejunk.py p N seeds [Kmin Kmax]
import torch, math, sys
torch.set_default_dtype(torch.float64)
p=int(sys.argv[1]); N=int(sys.argv[2]); seeds=int(sys.argv[3]); h=(p-1)//2
Kmin=int(sys.argv[4]) if len(sys.argv)>4 else 2; Kmax=int(sys.argv[5]) if len(sys.argv)>5 else min(h,18)
x=torch.arange(p)
k=torch.arange(1,h+1)
A=1-torch.cos(2*math.pi*torch.outer(k,k)/p)            # A[D,k], D,k=1..h
# free-share Gamma lower bound by multistart (junk-free programme): max_b min_D sum_k A_Dk b_k^{3/2}
def gamma_prog(trials=300):
    best=0;bestS=None
    g=torch.Generator().manual_seed(0)
    for t in range(trials):
        z=(torch.randn(h,generator=g)*2).requires_grad_(); opt=torch.optim.Adam([z],lr=0.05)
        for s in range(600):
            tau=0.05*(0.01**(s/600)); b=torch.softmax(z,0)
            v=A@b**1.5; loss=tau*torch.logsumexp(-v/tau,0); opt.zero_grad();loss.backward();opt.step()
        b=torch.softmax(z,0).detach(); val=(A@b**1.5).min().item()
        if val>best: best=val;bestS=[i+1 for i in range(h) if b[i]>1e-3]
    return best,bestS
Gam,GS=gamma_prog()
target=3**-0.5*2*(6*p)**-1.5*Gam
print(f"p={p}: free-share Gamma (multistart lower bound) {Gam:.6f} on {GS}",flush=True)
Y=(x[:,None]+x[None,:])%p; idx=(Y[:,:,None]+x[None,None,:])%p
def gaps(E1,E2,W):
    Z=torch.einsum('ia,ib,ic->abc',E1,E2,W); Zc=torch.gather(Z,2,idx); return (Zc[:,:,:1]-Zc)[:,:,1:]
best=-9
for sd in range(seeds):
    g=torch.Generator().manual_seed(sd)
    if sd==0: S=torch.tensor(GS)
    else:
        K=int(torch.randint(Kmin,Kmax+1,(1,),generator=g)); S=torch.randperm(h,generator=g)[:K]+1
    ks=S.repeat_interleave(N)[:,None].double()
    C=torch.cos(2*math.pi*ks*x/p); Sn=torch.sin(2*math.pi*ks*x/p)
    par=torch.randn(len(S)*N,3,2,generator=g).requires_grad_()
    opt=torch.optim.Adam([par],lr=1e-2)
    for step in range(3000):
        tau=0.3*(0.01**(step/3000))
        E=[par[:,j,0:1]*C+par[:,j,1:2]*Sn for j in range(3)]
        P=sum((e**2).sum() for e in E); En=[e/P.sqrt() for e in E]
        gq=gaps(*En).reshape(-1)/target
        loss=tau*torch.logsumexp(-gq/tau,0); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        E=[par[:,j,0:1]*C+par[:,j,1:2]*Sn for j in range(3)]; P=sum((e**2).sum() for e in E)
        val=(gaps(*E).min()/P**1.5/target).item()
    best=max(best,val)
    print(f"seed {sd} N={N} K={len(S)} S={S.tolist()}: margin/ansatz {val:.4f}  best {best:.4f}",flush=True)
