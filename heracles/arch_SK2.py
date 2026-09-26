"""arch_SK2.py -- (S_K) test with a proper Gamma reference and structured starts. n=2.
See arch_SK.py for the statement. Starts: (a) perturbed Gamma-optimum with junk+mirror at several scales,
(b) random on random supports, (c) rank-one M_k (one-atom-per-frequency, maximal junk) and rank-two M_k."""
import torch, numpy as np, sys, json, math, itertools
torch.set_default_dtype(torch.float64)
p=int(sys.argv[1]); r=float(eval(sys.argv[2])); nstart=int(sys.argv[3]); seed=int(sys.argv[4]) if len(sys.argv)>4 else 0
H=(p-1)//2; ks=torch.arange(1,H+1); w=2*math.pi/p
a=torch.arange(p); Dv=torch.arange(1,p)
W=1-torch.exp(1j*w*ks[:,None]*Dv[None,:]); u=torch.exp(-1j*w*2*ks[:,None]*a[None,:])
cosm=(1-torch.cos(w*ks[:,None]*Dv[None,:]))   # H x (p-1)
def build(x):
    x=x.view(H,7,2); z=torch.complex(x[...,0],x[...,1]); return z[:,0],z[:,1:4],z[:,4:7]
def evalR(x,temp=None):
    A,C,U=build(x)
    h=(A[:,None,None]+C[:,0,None,None]*u[:,:,None]+C[:,1,None,None]*u[:,None,:]+C[:,2,None,None]*u[:,:,None]*u[:,None,:])
    F=torch.einsum('kd,kab->dab',W,h).real
    M=torch.zeros(H,3,3,dtype=torch.complex128); i=torch.arange(3); M[:,i,i]=A[:,None]
    M[:,1,0]=C[:,0];M[:,2,1]=C[:,1];M[:,2,0]=C[:,2];M[:,0,1]=U[:,0];M[:,1,2]=U[:,1];M[:,0,2]=U[:,2]
    s=torch.linalg.svdvals(M); cost=((s+1e-13)**r).sum()/3
    mn=F.min() if temp is None else -temp*torch.logsumexp(-F.flatten()/temp,0)
    return mn/cost**(1/r)
def ascend(x0,temps=(0.3,0.1,0.03,0.01,0.003,0.001,0.0003)):
    x=x0.clone().requires_grad_(True)
    for temp in temps:
        opt=torch.optim.LBFGS([x],max_iter=400,line_search_fn='strong_wolfe',tolerance_grad=1e-13,tolerance_change=1e-16)
        def cl():
            opt.zero_grad(); v=-evalR(x,temp)
            if not torch.isfinite(v): v=(x*0).sum()+1e6
            v.backward(); return v
        try: opt.step(cl)
        except Exception: break
        with torch.no_grad():
            sc=evalR(x).abs()  # scale-free objective; renormalise size
            x/= (x.norm()+1e-30)
    with torch.no_grad(): return float(evalR(x)),x.detach()
# exact Gamma: subsets x moduli
def gamma_ref():
    best=0;arg=None
    for m in range(1,2**H):
        S=[k for k in range(H) if m>>k&1]
        if len(S)<2: continue
        for t in range(3):
            y=torch.randn(len(S),requires_grad=True)
            opt=torch.optim.LBFGS([y],max_iter=200,line_search_fn='strong_wolfe')
            for temp in [0.1,0.01,0.001,0.0001]:
                def cl():
                    opt.zero_grad(); b=torch.softmax(y,0); v=(b[:,None]**(1/r)*cosm[S]).sum(0); f=temp*torch.logsumexp(-v/temp,0); f.backward(); return f
                opt.step(cl)
            b=torch.softmax(y,0).detach(); v=float((b[:,None]**(1/r)*cosm[S]).sum(0).min())
            if v>best: best=v;arg=(S,b)
    return best,arg
gam,(S,b)=gamma_ref()
print(f"p={p} r={r:.6f} Gamma_(1/r) = {gam:.10f} support k={[s+1 for s in S]} shares={b.numpy().round(4)}",flush=True)
g=torch.Generator().manual_seed(seed)
xg=torch.zeros(H,7,2); xg[S,0,0]=b**(1/r)
best=-1;xb=None;hist=[]
for t in range(nstart):
    kind=t%4
    if kind==0:
        x0=xg.clone(); sc=[0.01,0.05,0.2,0.5][(t//4)%4]; x0[:,1:]=torch.randn(H,6,2,generator=g)*sc*float(xg.abs().max())
    elif kind==1:
        x0=torch.randn(H,7,2,generator=g); msk=(torch.rand(H,generator=g)<0.6).double(); x0*=msk[:,None,None]
    elif kind==2: # rank-one per frequency: M_k = A m m^*, m=(1,v1,v1v2)
        x0=torch.zeros(H,7,2); ph=torch.rand(H,2,generator=g)*2*math.pi
        v1=torch.exp(1j*ph[:,0]);v2=torch.exp(1j*ph[:,1]); Ak=xg[:,0,0]+0.2*torch.rand(H,generator=g)*float(xg.abs().max())
        vals=torch.stack([Ak+0j,Ak*v1,Ak*v2,Ak*v1*v2,Ak*v1.conj(),Ak*v2.conj(),Ak*(v1*v2).conj()],1)
        x0[...,0]=vals.real;x0[...,1]=vals.imag; x0+=0.01*torch.randn(H,7,2,generator=g)
    else:
        x0=xg.clone()+0.3*float(xg.abs().max())*torch.randn(H,7,2,generator=g); x0[:,4:]*=3
    v,x=ascend(x0.flatten(),temps=((0.01,0.003,0.001,0.0003,0.0001) if kind in (0,3) else (0.05,0.02,0.01,0.003,0.001,0.0003))); hist.append(v)
    if v>best: best=v;xb=x
    if t%8==7: print(f"  start {t+1}: best R/Gamma = {best/gam:.10f}   (last 8: {np.round(np.array(hist[-8:])/gam,4)})",flush=True)
A,C,U=build(xb)
print(f"FINAL p={p} r={r:.4f}: best R/Gamma = {best/gam:.10f}; max junk/|A| {float(C.abs().max()/A.abs().max()):.2e}, max mirror/|A| {float(U.abs().max()/A.abs().max()):.2e}")
json.dump(dict(p=p,r=r,gamma=gam,best=best,ratio=best/gam,hist=hist,x=xb.tolist()),open(f'FINAL_RUNS/ARCH/SK2_p{p}_r{r:.3f}_s{seed}.json','w'))
