"""arch_SK.py -- test of the MATRIX REDUCTION (S_K), n=2 (all sign patterns are intervals).
Variables: per frequency k in 1..(p-1)/2 a 3x3 complex matrix M_k = [[A,U1,U12],[C1,A,U2],[C12,C2,A]]
 (A clean amplitude, C junk = lower part, U FREE 'mirror' completion = upper part).
Margin function (pure class profile identity): F(a,b,D) = sum_k Re[(1-w^{kD}) h_k(w^{-2ka}, w^{-2kb})],
 h_k(u) = A + C1 u1 + C2 u2 + C12 u1 u2.
Cost proxy (PROVED lower bound on true cost, any N, any junk): sum_l |c_lk|^r >= 3^{-r} ||M_k||_{S_r}^r.
Claim (S_K):  min F <= Gamma_{1/r}(p) * ( sum_k ||M_k||_{S_r}^r / 3 )^{1/r}.
We MAXIMISE  R = minF / (sum_k ||M_k||_r^r/3)^{1/r}  and compare with the junk-free value Gamma (same code, L=U=0)."""
import torch, numpy as np, sys, json, math
torch.set_default_dtype(torch.float64)
p=int(sys.argv[1]); r=float(sys.argv[2]); nstart=int(sys.argv[3]); seed=int(sys.argv[4]) if len(sys.argv)>4 else 0
H=(p-1)//2; ks=torch.arange(1,H+1)
w=2*math.pi/p
a=torch.arange(p)
Dv=torch.arange(1,p)
W=1-torch.exp(1j*w*ks[:,None]*Dv[None,:])            # H x (p-1)
u=torch.exp(-1j*w*2*ks[:,None]*a[None,:])            # H x p
def parts(x,free):
    x=x.view(H,7,2); z=torch.complex(x[...,0],x[...,1])
    A=z[:,0]; C=z[:,1:4]; U=z[:,4:7]
    if not free: C=C*0; U=U*0
    return A,C,U
def evalR(x,free=True,temp=None):
    A,C,U=parts(x,free)
    h=(A[:,None,None]+C[:,0,None,None]*u[:,:,None]+C[:,1,None,None]*u[:,None,:]+C[:,2,None,None]*u[:,:,None]*u[:,None,:]) # H,p,p
    F=torch.einsum('kd,kab->dab',W,h).real      # (p-1),p,p
    M=torch.zeros(H,3,3,dtype=torch.complex128)
    idx=torch.arange(3); M[:,idx,idx]=A[:,None]
    M[:,1,0]=C[:,0];M[:,2,1]=C[:,1];M[:,2,0]=C[:,2];M[:,0,1]=U[:,0];M[:,1,2]=U[:,1];M[:,0,2]=U[:,2]
    s=torch.linalg.svdvals(M)
    cost=((s+1e-14)**r).sum()/3
    if temp is None: mn=F.min()
    else: mn=-temp*torch.logsumexp(-F.flatten()/temp,0)
    return mn/cost**(1/r)
def ascend(x0,free):
    x=x0.clone().requires_grad_(True)
    for temp in [0.3,0.1,0.03,0.01,0.003,0.001]:
        opt=torch.optim.LBFGS([x],max_iter=300,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-15)
        def cl():
            opt.zero_grad(); v=-evalR(x,free,temp)
            if not torch.isfinite(v): v=torch.tensor(1e6,requires_grad=True)
            v.backward(); return v
        try: opt.step(cl)
        except Exception as e: break
        with torch.no_grad():
            nrm=x.abs().max()
            if nrm>0: x/= (x.view(H,7,2)[:,0,:].norm()+1e-30)
    with torch.no_grad(): return float(evalR(x,free,None)),x.detach()
g=torch.Generator().manual_seed(seed)
# junk-free reference (= Gamma_{1/r}(p) in these units)
gam=-1;xg=None
for t in range(max(40,nstart)):
    x0=torch.randn(H*14,generator=g)
    v,x=ascend(x0,False)
    if v>gam: gam=v;xg=x
print(f"p={p} r={r:.4f} junk-free Gamma (these units) = {gam:.10f}",flush=True)
best=-1;xb=None;hist=[]
for t in range(nstart):
    kind=t%3
    if kind==0: x0=torch.randn(H*14,generator=g)
    elif kind==1:
        x0=xg.clone().view(H,7,2); x0[:,1:]=torch.randn(H,6,2,generator=g)*[0.02,0.1,0.3][(t//3)%3]; x0=x0.flatten()
    else:
        x0=torch.randn(H*14,generator=g); x0=x0.view(H,7,2); x0[:,4:]*=3; x0=x0.flatten()
    v,x=ascend(x0,True); hist.append(v)
    if v>best: best=v;xb=x
    if t%10==9: print(f"  start {t+1}: best R/Gamma = {best/gam:.10f}",flush=True)
A,C,U=parts(xb,True)
print(f"FINAL p={p} r={r:.4f}: best R/Gamma = {best/gam:.10f}; junk norm at best {float(C.abs().max()):.2e}, mirror {float(U.abs().max()):.2e}, |A|={A.abs().numpy().round(4)}")
json.dump(dict(p=p,r=r,gamma=gam,best=best,ratio=best/gam,hist=hist,x=xb.tolist()),open(f'FINAL_RUNS/ARCH/SK_p{p}_r{r:.3f}_s{seed}.json','w'))
