# X-blind relaxation R: maximize, over PURE networks (per-frequency atoms), the best upper bound on the margin
# obtainable from the X-blind family  { row-uniform, any offset weights }  U  { half-offset type-T lines, uniform D },
# normalized by P^{3/2}. If sup R == ansatz value, impure neurons are harmless (the family is blind to cross terms).
# usage: xblind_relax.py p Gamma seeds
import torch, math, sys
torch.set_default_dtype(torch.float64)
p=int(sys.argv[1]); Gam=float(sys.argv[2]); seeds=int(sys.argv[3]); h=(p-1)//2
target=3**-0.5*2*(6*p)**-1.5*Gam
x=torch.arange(p); inv2=pow(2,-1,p)
Y=(x[:,None]+x[None,:])%p
def gap_table(E1,E2,W):
    Z=torch.einsum('ia,ib,ic->abc',E1,E2,W)
    idx=(Y[:,:,None]+x[None,None,:])%p                 # class y+D
    Zc=torch.gather(Z,2,idx)                            # [x1,x2,D] logit of class y+D
    return Zc[:,:,:1]-Zc                                # gap(x,D), D=0 column is 0
def bounds(G):
    clean=G.mean((0,1))[1:]                             # row-average gap per D
    Ds=x[1:]
    out=[clean]
    # type 1: x1 = s - D/2 ; type 2: x2 = s - D/2 ; type 12: x1+x2 = s - D/2
    s=x[:,None]; sh=(s-Ds[None,:]*inv2)%p               # [s,D]
    t1=G[sh,:,Ds[None,:]].mean(-1).mean(1)              # mean over x2, then over D
    t2=G[:,sh,Ds[None,:]].mean(0).mean(1)
    x1=x[None,None,:]; x2=(sh[:,:,None]-x1)%p
    t12=G[x1.expand_as(x2),x2,Ds[None,:,None].expand_as(x2)].mean(-1).mean(1)
    out+= [t1,t2,t12]
    return torch.cat(out)
def build(par,K,Nper):
    a=par.view(K*Nper,3,2)
    ks=torch.arange(1,K+1).repeat_interleave(Nper)[:,None]
    C=torch.cos(2*math.pi*ks*x/p); S=torch.sin(2*math.pi*ks*x/p)
    E=[a[:,j,0:1]*C+a[:,j,1:2]*S for j in range(3)]
    return E
def P(E): return sum((e**2).sum() for e in E)
# calibration: build a junk-free dense block and check the bound equals its (true) margin
best=-1
for sd in range(seeds):
    g=torch.Generator().manual_seed(sd)
    Nper=[1,2,3][sd%3]; K=h
    par=torch.randn(K*Nper*6,generator=g)
    # random sparse support: zero most frequencies
    keep=torch.rand(K,generator=g)<torch.rand(1,generator=g).item()*0.8+0.1
    par=(par.view(K,Nper*6)*keep[:,None]).view(-1).requires_grad_()
    opt=torch.optim.Adam([par],lr=1e-2)
    for step in range(2500):
        tau=0.3*(0.01**(step/2500))
        E=build(par,K,Nper); Pn=P(E)
        b=bounds(gap_table(*E))/Pn**1.5/target
        loss=tau*torch.logsumexp(-b/tau,0)
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        E=build(par,K,Nper); val=(bounds(gap_table(*E)).min()/P(E)**1.5/target).item()
        G=gap_table(*E); true=(G[:,:,1:].min()/P(E)**1.5/target).item()
    best=max(best,val)
    print(f"seed {sd} Nper={Nper}: X-blind bound / ansatz = {val:.4f}  (true margin/ansatz {true:.4f})  best {best:.4f}",flush=True)
